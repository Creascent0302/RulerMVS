"""GaussianFormer-2 Prob-256 官方网络的四目数据薄适配。"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(__file__).resolve().parent / "upstream"
COMMIT = "b7e22bfc04cd6360cdee74be5af7fdace102f0a3"


def _native_modules():
    for path in [UPSTREAM, UPSTREAM / "model/head/localagg_prob", ROOT / "build_src/mmcv-2.0.1"]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # 官方包使用 model/loss/misc 绝对导入。各方法必须在独立命令进程运行。
    if "model" in sys.modules and not str(getattr(sys.modules["model"], "__file__", "")).startswith(
        str(UPSTREAM)
    ):
        raise RuntimeError("GaussianFormer 官方 model 包与已加载的其他方法冲突；请使用独立进程。")
    importlib.import_module("model")
    return importlib.import_module("loss")


def _grid(roi, voxel_size, device):
    bounds = torch.tensor(roi, dtype=torch.float32, device=device)
    size = torch.as_tensor(voxel_size, dtype=torch.float32, device=device).expand(3)
    count = (bounds[:, 1] - bounds[:, 0]) / size
    if not torch.allclose(count, count.round(), atol=1e-5):
        raise ValueError("GF2 的 ROI 必须是体素边长的整数倍")
    axes = [
        bounds[i, 0] + (torch.arange(int(count[i]), device=device) + 0.5) * size[i]
        for i in range(3)
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)


class Adapter(nn.Module):
    # 官方ResNet启用重入checkpoint；固定Prob-256路径使用全部可训练参数，禁用DDP未使用参数遍历。
    ddp_find_unused_parameters = False
    capabilities = {
        "depth": False,
        "confidence": False,
        "points": False,
        "occupancy": True,
        "semantics": True,
        "native_gaussians": True,
        "unknown": False,
    }
    implementation_limitations = [
        "官方SparseConv3D把多个高斯映射到重复整数坐标；当前spconv默认Native后端在固定RGB、标定、权重和RNG下仍有显著非确定性。",
        "完整Prob-256合成诊断的重复推理语义不一致率约16.77%；不能把前后向可运行称为默认数值一致性验收通过。见gaussianformer2_prob256_repeatability.json。",
        "官方仓库未发布完整初始化器第一阶段训练配置；补充训练入口不等于第一阶段论文超参数复现。",
    ]

    def __init__(self, config):
        super().__init__()
        from mmengine import Config

        self.config = config
        options = config.get("model", {})
        native_path = options.get("native_config", "config/prob/nuscenes_gs25600.py")
        path = Path(native_path)
        if not path.is_absolute():
            path = UPSTREAM / path
        cfg = Config.fromfile(str(path))
        if cfg.model.lifter.type != "GaussianLifterV2" or not cfg.model.head.get(
            "use_localaggprob", False
        ):
            raise ValueError("必须使用第二代 GaussianLifterV2 + localagg_prob；不接受第一代配置")
        self.stage = options.get("training_stage", "occupancy")
        if self.stage not in ("initializer", "occupancy"):
            raise ValueError("training_stage 必须为 initializer 或 occupancy")
        if self.stage == "initializer":
            self.capabilities = {
                "pixel_distribution": True,
                "occupancy": False,
                "semantics": False,
                "unknown": False,
            }
            self.implementation_limitations = self.implementation_limitations[-1:]
        else:
            warnings.warn(
                self.implementation_limitations[0]
                + " 详见 official_methods/docs/gaussianformer2.md。",
                RuntimeWarning,
            )
        self.roi = config.get("evaluation", {}).get("roi", [[0, 64], [-24, 24], [-2, 6]])
        self.voxel_size = float(options.get("voxel_size", 0.5))
        self.shape = tuple(_grid(self.roi, self.voxel_size, "cpu").shape[:3])
        pc_range = [r[0] for r in self.roi] + [r[1] for r in self.roi]
        model_cfg = cfg.model
        model_cfg.encoder.deformable_model.num_cams = int(options.get("num_cameras", 4))
        model_cfg.lifter.update(
            pc_range=pc_range,
            voxel_size=self.voxel_size,
            occ_resolution=list(self.shape),
            empty_label=17,
        )
        for block in [
            model_cfg.encoder.deformable_model.kps_generator,
            model_cfg.encoder.refine_layer,
            model_cfg.encoder.spconv_layer,
        ]:
            block.pc_range = pc_range
        model_cfg.head.cuda_kwargs.update(
            H=self.shape[0],
            W=self.shape[1],
            D=self.shape[2],
            pc_min=pc_range[:3],
            grid_size=self.voxel_size,
        )
        initializer = options.get("initializer_checkpoint", "")
        full_checkpoint = options.get("checkpoint", "")
        synthetic_random_init = bool(options.get("synthetic_random_init", False))
        loading_shared = bool(config.get("_loading_shared_checkpoint", False))
        if (
            not loading_shared
            and not synthetic_random_init
            and not options.get("backbone_checkpoint")
            and not full_checkpoint
        ):
            raise FileNotFoundError(
                "GF2 原生训练使用 FCOS3D 骨干预训练；请填写 backbone_checkpoint 或完整 checkpoint"
            )
        if (
            not loading_shared
            and not initializer
            and not full_checkpoint
            and not synthetic_random_init
            and self.stage != "initializer"
        ):
            raise FileNotFoundError(
                "GF2 的分布初始化器在官方占据训练阶段被冻结。请设置 initializer_checkpoint、"
                "完整 checkpoint，或先训练 training_stage=initializer；随机权重只用于显式合成测试。"
            )
        # 构建后检查每个必需权重，再加载；不接受上游 strict=False 的零匹配成功信息。
        model_cfg.lifter.pretrained_path = None
        if self.stage == "initializer":
            model_cfg.freeze_lifter = False
        if options.get("smoke_num_anchors") is not None:
            if not synthetic_random_init:
                raise ValueError("减少高斯数量只允许显式 synthetic_random_init 冒烟测试")
            total = int(options["smoke_num_anchors"])
            model_cfg.lifter.num_anchor = total * 3 // 4
            model_cfg.lifter.random_samples = total - total * 3 // 4
        # 保持官方默认随机分布采样；可复现测试在调用前固定 RNG 状态。
        if "deterministic_sampling" in options:
            model_cfg.lifter.deterministic = bool(options["deterministic_sampling"])
        self.native_config = cfg
        loss_module = _native_modules()
        from mmseg.models import build_segmentor

        self.native_model = build_segmentor(model_cfg)
        self.native_model.init_weights()
        if "depth_range" in options:
            low, high = options["depth_range"]
            self.native_model.lifter.depth_bins.copy_(
                torch.linspace(low, high, self.native_model.lifter.num_samples)
            )
        self.native_losses = loss_module.OPENOCC_LOSS.build(cfg.loss)
        self.weight_load_report = {}
        if initializer:
            state = torch.load(initializer, map_location="cpu", weights_only=False)
            state = state.get("state_dict", state)
            lifter = self.native_model.lifter
            required = {
                k
                for k in lifter.state_dict()
                if k.startswith(("initialize_backbone.", "projection."))
            }
            missing = required - state.keys()
            if missing:
                raise ValueError(
                    f"初始化器权重不完整，缺少 {len(missing)} 项，例如 {sorted(missing)[:3]}"
                )
            result = lifter.load_state_dict(
                {k: v for k, v in state.items() if k in required}, strict=False
            )
            self.weight_load_report["initializer"] = {
                "matched": len(required),
                "missing_non_initializer": result.missing_keys,
            }
        for name, checkpoint in [
            ("backbone", options.get("backbone_checkpoint", "")),
            ("full", full_checkpoint),
        ]:
            if checkpoint:
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                state = state.get("state_dict", state.get("model", state))
                if name == "backbone":
                    required = {
                        k for k in self.native_model.state_dict() if k.startswith("img_backbone.")
                    }
                    if required - state.keys():
                        raise ValueError(
                            "FCOS3D骨干权重缺少当前原生 img_backbone 参数，不能冻结/训练未正确加载的骨干"
                        )
                    state = {k: v for k, v in state.items() if k in required}
                # ROI是协议常量；加载原nuScenes权重不能把CARLA配置范围悄悄改回去。
                geometry = {
                    k: v.clone()
                    for k, v in self.native_model.named_buffers()
                    if k.endswith(("pc_range", "grid_size"))
                }
                result = self.native_model.load_state_dict(state, strict=name == "full")
                for key, value in self.native_model.named_buffers():
                    if key in geometry:
                        value.copy_(geometry[key])
                self.weight_load_report[name] = {
                    "missing": result.missing_keys,
                    "unexpected": result.unexpected_keys,
                }
        if self.stage == "initializer":
            trainable = [
                p
                for name, p in self.native_model.named_parameters()
                if name.startswith(("lifter.initialize_backbone.", "lifter.projection."))
                and p.requires_grad
            ]
            self.native_model.requires_grad_(False)
            for parameter in trainable:
                parameter.requires_grad_(True)

    @property
    def native(self):
        return self.native_model

    def prepare_training_inputs(self, inputs, targets):
        """复用官方颜色增强；翻转与项目整数像素中心的K同步，3D标签不变。"""
        if not self.training or not self.config.get("training", {}).get(
            "native_augmentation", True
        ):
            return inputs
        if "occupancy" not in targets:
            from official_methods.occupancy_labels import observed_occupancy

            targets["occupancy"] = [observed_occupancy(inputs, targets, self.roi, self.voxel_size)]
        from dataset.transform_3d import PhotoMetricDistortionMultiViewImage

        rgb = inputs["rgb"]
        b, n, _, h, w = rgb.shape
        # 上游先读BGR，扰动后再在NormalizeMultiviewImage中转回RGB。
        bgr = (rgb.detach().cpu().permute(0, 1, 3, 4, 2).numpy()[..., ::-1] * 255).astype(
            np.float32
        )
        output, calibration = [], inputs["K"].clone()
        for index in range(b):
            flip = bool(np.random.randint(2))
            images = [image[:, ::-1].copy() if flip else image.copy() for image in bgr[index]]
            images = PhotoMetricDistortionMultiViewImage()({"img": images})["img"]
            output.append(np.stack(images)[..., ::-1].copy())
            if flip:
                transform = calibration.new_tensor([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]])
                calibration[index] = transform @ calibration[index]
        augmented = torch.from_numpy(np.stack(output)).to(device=rgb.device, dtype=rgb.dtype)
        return dict(inputs, rgb=augmented.permute(0, 1, 4, 2, 3) / 255, K=calibration)

    def native_inputs(self, inputs):
        rgb = inputs["rgb"]
        b, n, _, h, w = rgb.shape
        if h % 32 or w % 32:
            raise ValueError(
                "原生 SECONDFPN 要求高宽为32的倍数；请在数据层同步缩放图像与K，例如704×1280"
            )
        if b != 1:
            raise ValueError(
                "官方 localagg_prob 每 GPU 仅支持 B=1；使用 DDP 或梯度累积扩大有效批量"
            )
        if n != self.native_config.model.encoder.deformable_model.num_cams:
            raise ValueError("实际相机数与配置不符；不得复制相机补足视图")
        if not inputs.get("view_mask", torch.ones(b, n, device=rgb.device, dtype=torch.bool)).all():
            raise ValueError("当前官方 GF2 适配要求四台相机都有效，不复制或填充缺失图像")
        if "image_mask" in inputs and not inputs["image_mask"].all():
            raise ValueError("GF2 官方采样不支持任意图内缺损掩码；请先几何一致裁剪到共同有效图像")
        mean = rgb.new_tensor([123.675, 116.28, 103.53]).view(1, 1, 3, 1, 1)
        std = rgb.new_tensor([58.395, 57.12, 57.375]).view(1, 1, 3, 1, 1)
        images = (rgb.float() * 255 - mean) / std
        # 官方 raster 坐标以图像边界为0；项目像素中心为整数，平移半像素。
        k = inputs["K"].float().clone()
        k[..., :2, 2] += 0.5
        intrinsic = torch.eye(4, device=rgb.device).expand(b, n, 4, 4).clone()
        intrinsic[..., :3, :3] = k
        metas = {
            "projection_mat": intrinsic @ inputs["T_cam_from_rig"].float(),
            "image_wh": rgb.new_tensor([w, h]).expand(b, n, 2),
            "occ_xyz": _grid(self.roi, self.voxel_size, rgb.device)[None].expand(b, -1, -1, -1, -1),
        }
        return {"imgs": images, "metas": metas}

    def forward(self, inputs):
        args = self.native_inputs(inputs)
        if not args["imgs"].is_cuda:
            raise RuntimeError(
                "GF2 官方概率叠加、变形采样、DCNv2与spconv需要CUDA；没有伪造CPU替代模型"
            )
        # 官方扩展接收float32；避免外层AMP将几何概率和协方差降精度。
        with torch.autocast(device_type="cuda", enabled=False):
            if self.stage == "initializer":
                native = self.native_model.lifter(**args)
                native["metas"] = args["metas"]
                return {"native": native}
            native = self.native_model(**args)
        semantics = native["final_occ"].reshape(1, *self.shape)
        volume = dict(
            state=(semantics[0] != 17).to(torch.uint8),
            semantics=semantics[0],
            grid_origin=args["imgs"].new_tensor([r[0] for r in self.roi]),
            voxel_size=args["imgs"].new_full((3,), self.voxel_size),
        )
        return {
            "native": native,
            "occupancy": [volume],
            "semantics": semantics,
            "occupancy_probability": native["bin_logits"][-1].reshape(1, *self.shape),
            "grid_origin": args["imgs"].new_tensor([r[0] for r in self.roi]),
            "voxel_size": args["imgs"].new_full((3,), self.voxel_size),
        }

    def _labels(self, inputs, targets):
        if "occupancy" in targets:
            label = targets["occupancy"][0]
        else:
            from official_methods.occupancy_labels import observed_occupancy

            label = observed_occupancy(inputs, targets, self.roi, self.voxel_size)
        device = inputs["rgb"].device
        label = {key: torch.as_tensor(value, device=device) for key, value in label.items()}
        if tuple(label["state"].shape) != self.shape:
            raise ValueError("占据标签网格与配置不一致；禁止把GT范围用于确定推理网格")
        torch.testing.assert_close(
            label["grid_origin"].float(),
            torch.tensor([r[0] for r in self.roi], device=device).float(),
        )
        torch.testing.assert_close(
            label["voxel_size"].float().expand(3), torch.full((3,), self.voxel_size, device=device)
        )
        state = label["state"].long()
        semantic = label.get("semantics", torch.full_like(state, -1)).long()
        if (
            self.stage == "occupancy"
            and (state == 1).any()
            and not ((state == 1) & (semantic >= 0) & (semantic < 17)).any()
        ):
            raise ValueError(
                "存在 occupied 标签但缺少有效的 0–16 语义标签；不能静默仅训练 free 类别"
            )
        native = torch.full_like(state, 255)
        native[state == 0] = 17
        known_sem = (state == 1) & (semantic >= 0) & (semantic < 17)
        native[known_sem] = semantic[known_sem]
        return state, native

    def _pixel_target(self, native, state):
        logits = native["pixel_logits"]
        b, n, h, w, channels = logits.shape
        metas = native["metas"]
        us = (torch.arange(w, device=logits.device) + 0.5) * metas["image_wh"][0, 0, 0] / w
        vs = (torch.arange(h, device=logits.device) + 0.5) * metas["image_wh"][0, 0, 1] / h
        vv, uu = torch.meshgrid(vs, us, indexing="ij")
        uv1 = torch.stack([uu, vv, torch.ones_like(uu)], -1)
        rays = uv1[..., None, :] * self.native_model.lifter.depth_bins[:, None]
        homogeneous = torch.cat([rays, torch.ones_like(rays[..., :1])], -1)
        points = torch.einsum("bnij,hwdj->bnhwdi", metas["projection_mat"].inverse(), homogeneous)[
            ..., :3
        ]
        origin = logits.new_tensor([r[0] for r in self.roi])
        indices = ((points - origin) / self.voxel_size).floor().long()
        sizes = indices.new_tensor(self.shape)
        inside = ((indices >= 0) & (indices < sizes)).all(-1)
        safe = indices.clamp_min(0).minimum(sizes - 1)
        sampled = state[safe[..., 0], safe[..., 1], safe[..., 2]]
        valid = inside & (sampled != 2)
        occupied = (sampled == 1) & inside
        # 无命中桶只有整条有限范围射线已知时才监督；unknown不等于empty。
        empty_valid = valid.all(-1, keepdim=True)
        pixel_gt = torch.cat([occupied, ~occupied.any(-1, keepdim=True)], -1).float()
        mask = torch.cat([valid, empty_valid], -1)
        return pixel_gt, mask

    def loss(self, prediction, inputs, targets):
        native = prediction["native"]
        state, labels = self._labels(inputs, targets)
        logits = native["pixel_logits"]
        losses = {}
        pixel_gt, pixel_mask = self._pixel_target(native, state)
        if self.stage == "initializer":
            if not pixel_mask.any():
                raise ValueError("本批无已观测空间标签，不能训练分布初始化器")
            losses["pixel_distribution"] = F.binary_cross_entropy(
                logits.softmax(-1)[pixel_mask], pixel_gt[pixel_mask]
            )
        else:
            valid = labels != 255
            if valid.any():
                args = dict(
                    pred_occ=native["pred_occ"],
                    sampled_xyz=native["sampled_xyz"],
                    sampled_label=labels.reshape(1, -1),
                    occ_mask=valid[None],
                )
                losses["occupancy_native"] = self.native_losses.losses[0](args)
            # 原Prob-256冻结initializer；这个原生项数值可记录，但不产生initializer梯度。
            if pixel_mask.any():
                losses["pixel_distribution"] = F.binary_cross_entropy(
                    logits.softmax(-1)[pixel_mask], pixel_gt[pixel_mask]
                )
            if not valid.any():
                raise ValueError("GF2原生语义占据损失缺少有效free/语义标签；不能伪造occupied类别")
        losses["total"] = sum(losses.values())
        return losses

    def export_initializer(self, path):
        """导出官方 lifter.load_state_dict 可读取的初始化器参数及 BN 缓冲区。"""
        if self.stage != "initializer":
            raise ValueError("只能从 training_stage=initializer 导出已训练的初始化器")
        state = {
            key: value.detach().cpu()
            for key, value in self.native_model.lifter.state_dict().items()
            if key.startswith(("initialize_backbone.", "projection."))
        }
        torch.save(
            {
                "state_dict": state,
                "source_commit": COMMIT,
                "stage": self.stage,
                "config": self.config,
            },
            path,
        )

    def configure_optimizer(self):
        settings = self.config.get("training", {})
        lr = settings.get("lr", self.native_config.optimizer.optimizer.lr)
        groups = [{"params": [], "lr": lr * 0.1}, {"params": [], "lr": lr}]
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                groups[0 if "img_backbone" in name else 1]["params"].append(parameter)
        return torch.optim.AdamW(
            [group for group in groups if group["params"]], lr=lr, weight_decay=0.01
        )

    def configure_scheduler(self, optimizer, total_steps):
        # 与官方 timm CosineLRScheduler(t_in_epochs=False, warmup_prefix=False) 一致。
        settings = self.config.get("training", {})
        warmup = int(settings.get("warmup_iters", 500))
        minimum = settings.get("lr", self.native_config.optimizer.optimizer.lr) * 0.1

        def schedule(base):
            def multiplier(step):
                # 官方训练在 global_iter += 1 之前调用 step_update(global_iter)。
                step = max(step - 1, 0)
                if warmup and step < warmup:
                    return (1e-6 + (base - 1e-6) * step / warmup) / base
                progress = min(step / max(total_steps, 1), 1.0)
                return (
                    minimum + 0.5 * (base - minimum) * (1 + math.cos(math.pi * progress))
                ) / base

            return multiplier

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, [schedule(group["lr"]) for group in optimizer.param_groups]
        )
