"""Niantic 官方网络的几何/监督薄适配；绝不向推理字典加入标签。"""

from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM = Path(__file__).resolve().parents[1] / "upstream"


def native_intrinsics(K, scale=1.0):
    """本项目整数像素中心 -> 官方 +0.5 像素中心，再缩放。"""
    out = torch.eye(4, device=K.device, dtype=K.dtype).expand(*K.shape[:-2], 4, 4).clone()
    out[..., :3, :3] = K
    out[..., :2, 2] += 0.5
    out[..., :2, :] *= scale
    return out


def depth_intrinsics(K, scale):
    out = native_intrinsics(K, scale)[..., :3, :3].clone()
    out[..., :2, 2] -= 0.5
    return out


def sample_views(tensor, indices):
    batch = torch.arange(tensor.shape[0], device=tensor.device)[:, None]
    return tensor[batch, indices]


class NianticAdapter(nn.Module):
    capabilities = {
        "depth": True,
        "confidence": False,
        "points": False,
        "occupancy": False,
        "semantics": False,
        "native": True,
    }
    prediction_scale = 0.5
    matching_scale = 0.25

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.settings = config.get("model", {})
        self.reference_indices = list(self.settings.get("reference_indices", [0]))
        self.native_model = None

    def prepare_native(self, inputs, reference):
        rgb, K, T = inputs["rgb"], inputs["K"].float(), inputs["T_cam_from_rig"].float()
        B, N, _, H, W = rgb.shape
        expected = self.native_model.run_opts.model_num_views
        if N != expected:
            raise ValueError(f"配置需要 {expected} 台真实相机，实际 {N}；禁止复制图像补视角。")
        if (H, W) != (
            self.native_model.run_opts.image_height,
            self.native_model.run_opts.image_width,
        ):
            raise ValueError("图像尺寸必须与原生投影网格一致；请先同步缩放 RGB 和内参。")
        if not inputs.get("view_mask", torch.ones(B, N, device=rgb.device, dtype=torch.bool)).all():
            raise ValueError("本集成要求全部四台相机有效；缺失视角不能伪装成有效观测。")
        if self.training and inputs.get("_geometric_flip", False):
            rgb = rgb.flip(-1)
            K = K.clone()
            K[..., 0, 2] = W - 1 - K[..., 0, 2]
            T = T.clone()
            T[..., 0, :] *= -1  # 原生数据增强的相机 x 反射；相对投影仍严格一致。
        if "image_mask" in inputs and not inputs["image_mask"].all():
            raise ValueError(
                "官方元数据成本体未接收任意像素缺失掩码；请先裁剪到四目均有效的矩形，禁止把无效源像素用于匹配。"
            )
        poses = torch.linalg.inv(T)
        sources = torch.tensor([i for i in range(N) if i != reference], device=rgb.device)[
            None
        ].expand(B, -1)
        relative = T[:, reference, None] @ sample_views(poses, sources)
        trace = relative[..., :3, :3].diagonal(dim1=-2, dim2=-1).sum(-1)
        rdist = torch.sqrt((2 * (1 - trace / 3)).clamp_min(0))
        tdist = relative[..., :3, 3].norm(dim=-1)
        order = torch.sqrt(tdist.square() + rdist.square()).argsort(dim=1, stable=True)
        sources = sources.gather(1, order)
        mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
        std = rgb.new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
        images = (rgb - mean) / std
        data = {"image_b3hw": images, "cam_T_world_b44": T, "world_T_cam_b44": poses}
        for i in range(5):
            ki = native_intrinsics(K, self.prediction_scale / 2**i)
            data[f"K_s{i}_b44"], data[f"invK_s{i}_b44"] = ki, torch.linalg.inv(ki)
        km = native_intrinsics(K, self.matching_scale)
        data["K_matching_b44"], data["invK_matching_b44"] = km, torch.linalg.inv(km)
        cur = {key: value[:, reference] for key, value in data.items()}
        src = {key: sample_views(value, sources) for key, value in data.items()}
        return cur, src, sources

    def call_native(self, cur, src, phase):
        return self.native_model(
            phase, cur, src, unbatched_matching_encoder_forward=phase != "train", return_mask=True
        )

    def forward(self, inputs):
        native, source_indices = [], []
        phase = "train" if self.training else "test"
        for ref in self.reference_indices:
            cur, src, order = self.prepare_native(inputs, ref)
            if self.training and "_training_depth_range" in inputs:
                cur["min_depth"] = inputs["_training_depth_range"][:, ref, 0]
                cur["max_depth"] = inputs["_training_depth_range"][:, ref, 1]
            if self.training and hasattr(self, "color_transform"):
                mean = cur["image_b3hw"].new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
                std = cur["image_b3hw"].new_tensor([0.229, 0.224, 0.225])[None, :, None, None]

                def jitter(image):
                    return (self.color_transform((image * std + mean).clamp(0, 1)) - mean) / std

                cur["image_b3hw"] = jitter(cur["image_b3hw"])
                src["image_b3hw"] = torch.stack(
                    [jitter(image) for image in src["image_b3hw"].unbind(1)], 1
                )
            if self.training and hasattr(self.native_model, "color_aug"):
                cur["image_b3hw"] = self.native_model.color_aug(
                    cur["image_b3hw"], denormalize_first=True
                )
                src["image_b3hw"] = torch.stack(
                    [
                        self.native_model.color_aug(image, denormalize_first=True)
                        for image in src["image_b3hw"].unbind(1)
                    ],
                    1,
                )
            native.append(self.call_native(cur, src, phase))
            source_indices.append(order)
        depth = torch.cat([out["depth_pred_s0_b1hw"] for out in native], dim=1)
        if self.training and inputs.get("_geometric_flip", False):
            depth = depth.flip(-1)
        valid = torch.isfinite(depth) & (depth > 0)
        if "image_mask" in inputs:
            masks = inputs["image_mask"][:, self.reference_indices].float()
            valid &= F.interpolate(masks, depth.shape[-2:], mode="nearest-exact").bool()
        return {
            "native": native,
            "depth": depth,
            "depth_valid": valid,
            "depth_K": depth_intrinsics(
                inputs["K"][:, self.reference_indices], self.prediction_scale
            ),
            "reference_indices": self.reference_indices,
            "source_indices": source_indices,
        }

    def prepare_training_inputs(self, inputs, targets):
        if not self.training:
            raise RuntimeError("几何增强只能在 train() 模式调用。")
        enabled = self.config.get("training", {}).get("geometric_flip", True)
        return dict(inputs, _geometric_flip=bool(enabled and torch.rand(()).item() < 0.5))

    def loss(self, prediction, inputs, targets):
        if "depth" not in targets:
            raise ValueError("此方法的原生监督训练要求米制 Z-depth 标签。")
        losses = []
        shape = prediction["depth"].shape[-2:]
        depth = targets["depth"]
        if depth.ndim == 5:
            depth = depth.squeeze(2)
        mask = targets.get("depth_valid", torch.isfinite(depth) & (depth > 0)).bool()
        if mask.ndim == 5:
            mask = mask.squeeze(2)
        if "image_mask" in inputs:
            mask = mask & inputs["image_mask"].bool()
        mask = mask & torch.isfinite(depth) & (depth > 0)
        depth = F.interpolate(depth.nan_to_num(), shape, mode="nearest-exact")
        mask = F.interpolate(mask.float(), shape, mode="nearest-exact").bool()
        depth = depth.masked_fill(~mask, torch.nan)
        if self.training and inputs.get("_geometric_flip", False):
            depth, mask = depth.flip(-1), mask.flip(-1)
        for i, ref in enumerate(self.reference_indices):
            cur, src, order = self.prepare_native(inputs, ref)
            cur["depth_b1hw"] = depth[:, ref : ref + 1]
            cur["mask_b_b1hw"] = mask[:, ref : ref + 1]
            cur["mask_b1hw"] = cur["mask_b_b1hw"].float()
            src["depth_b1hw"] = sample_views(depth, order).unsqueeze(2)
            if not cur["mask_b_b1hw"].any():
                zero = sum(
                    (
                        value.sum() * 0
                        for value in prediction["native"][i].values()
                        if torch.is_tensor(value) and value.requires_grad
                    ),
                    prediction["depth"].sum() * 0,
                )
                losses.append({"loss": zero})
                continue
            cur["normals_b3hw"] = self.native_model.compute_normals(
                cur["depth_b1hw"], cur["invK_s0_b44"]
            )
            outputs = dict(prediction["native"][i])
            outputs["normals_pred_b3hw"] = self.native_model.compute_normals(
                outputs["depth_pred_s0_b1hw"], cur["invK_s0_b44"]
            )
            if "sky" in targets:
                sky = targets["sky"]
                if sky.ndim == 5:
                    sky = sky.squeeze(2)
                cur["skymask_b1hw"] = F.interpolate(
                    sky[:, ref : ref + 1].float(), shape, mode="nearest-exact"
                )
                if self.training and inputs.get("_geometric_flip", False):
                    cur["skymask_b1hw"] = cur["skymask_b1hw"].flip(-1)
            else:
                cur["skymask_b1hw"] = torch.full_like(cur["depth_b1hw"], torch.nan)
            losses.append(self.native_model.compute_losses(cur, src, outputs))
        result = {
            key: torch.stack([item[key] for item in losses if key in item]).mean()
            for key in set().union(*(item.keys() for item in losses))
        }
        result["total"] = result["loss"]
        return result

    def load_native_weights(self, path, strict=True):
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            **(
                {"pickle_module": self.checkpoint_pickle_module}
                if hasattr(self, "checkpoint_pickle_module")
                else {}
            ),
        )
        state = payload.get("state_dict", payload)
        current = self.native_model.state_dict()
        regenerated = []
        for key in list(state):
            # 图像大小变化时重建解析像素网格，绝不跳过可训练参数。
            if (
                key.endswith("pix_coords_13N")
                and key in current
                and state[key].shape != current[key].shape
            ):
                state[key] = current[key]
                regenerated.append(key)
        result = self.native_model.load_state_dict(state, strict=strict)
        self.weight_loading_report = {
            "path": str(path),
            "regenerated_pixel_grids": regenerated,
            "missing_keys": result.missing_keys,
            "unexpected_keys": result.unexpected_keys,
        }
        return result

    def configure_optimizer(self):
        configured = self.native_model.configure_optimizers()
        return configured["optimizer"], configured["lr_scheduler"]["scheduler"]


def configure_options(options, config, defaults):
    for key, value in defaults.items():
        setattr(options, key, value)
    model, train = config.get("model", {}), config.get("training", {})
    mapping = {
        "image_height": "image_height",
        "image_width": "image_width",
        "num_views": "model_num_views",
        "depth_bins": "matching_num_depth_bins",
        "min_depth": "min_matching_depth",
        "max_depth": "max_matching_depth",
    }
    for key, field in mapping.items():
        if key in model:
            setattr(options, field, model[key])
    options.pretrained = bool(model.get("pretrained", False))
    options.lr = float(train.get("lr", options.lr))
    options.wd = float(train.get("weight_decay", options.wd))
    options.lr_steps = list(train.get("lr_steps", options.lr_steps))
    return options
