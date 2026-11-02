"""单次官方编码、共享匹配描述子的冻结深度初值器。

仅使用实际存在的相机。任意像素洞不属于官方代价体支持范围，因此明确报错；
整幅相机缺失由 view_mask 处理。这里不修改独立 baseline 的网络或推理入口。
"""

import copy
import json
from pathlib import Path

import torch
from torch import nn


class PrimaryDepthHead(nn.Module):
    """复用原 DPT 的全部模块和权重名，推理仅执行主深度输出头。"""

    def __init__(self, original, auxiliary_outputs=False):
        super().__init__()
        self.use_clstoken = original.use_clstoken
        self.auxiliary_outputs = auxiliary_outputs
        self._full_forward = type(original).forward
        for name, module in original.named_children():
            self.add_module(name, module)

    def forward(self, features, patch_h, patch_w):
        if self.auxiliary_outputs:
            return self._full_forward(self, features, patch_h, patch_w)

        maps = []
        for index, feature in enumerate(features):
            patches, token = feature
            if self.use_clstoken:
                readout = token[:, None].expand_as(patches)
                patches = self.readout_projects[index](torch.cat((patches, readout), dim=-1))
            image = patches.transpose(1, 2).reshape(patches.shape[0], -1, patch_h, patch_w)
            maps.append(self.resize_layers[index](self.projects[index](image)))

        layer_1 = self.scratch.layer1_rn(maps[0])
        layer_2 = self.scratch.layer2_rn(maps[1])
        layer_3 = self.scratch.layer3_rn(maps[2])
        layer_4 = self.scratch.layer4_rn(maps[3])
        path_4 = self.scratch.refinenet4(layer_4, size=layer_3.shape[-2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3, size=layer_2.shape[-2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2, size=layer_1.shape[-2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1)
        return {"log_depth_pred_s0_b1hw": self.output_convs[0](path_1)}


class SelectiveFusionEncoder(nn.Module):
    """保留原融合器及权重名，仅允许显式跳过非注入、非输出的 block。

    这是待适配的结构消融，不保证直接跳层仍保留原模型精度。
    默认空列表走原 forward，不改变原计算或四级 DPT 接口。
    """

    def __init__(self, original, skip_indices=()):
        super().__init__()
        protected = set(original.feat_fuser_layers_idx) | set(original.intermediate_layers_idx)
        if (
            any(type(index) is not int or not 0 <= index < len(original.model.blocks)
                for index in skip_indices)
            or len(set(skip_indices)) != len(skip_indices)
            or protected.intersection(skip_indices)
        ):
            raise ValueError("fusion_skip_indices 只能包含不重复的非注入、非输出 block 的零基编号")
        self.skip_indices = tuple(skip_indices)
        self.num_channels = original.num_channels
        self.num_ch_cv = original.num_ch_cv
        self.feat_fuser_layers_idx = original.feat_fuser_layers_idx
        self.intermediate_layers_idx = original.intermediate_layers_idx
        self._full_forward = type(original).forward
        self._prepare = type(original).prepare_tokens_with_masks
        for name, module in original.named_children():
            self.add_module(name, module)

    def prepare_tokens_with_masks(self, volume, image_features, masks=None):
        return self._prepare(self, volume, image_features, masks)

    def forward(self, volume, image_features):
        if not self.skip_indices:
            return self._full_forward(self, volume, image_features)
        features = []
        tokens = self.prepare_tokens_with_masks(volume, image_features[:2])
        for index, block in enumerate(self.model.blocks):
            if index in self.skip_indices:
                continue
            if index in self.feat_fuser_layers_idx:
                level = self.feat_fuser_layers_idx.index(index)
                patches, token = image_features[2 + level]
                context = torch.cat((token[:, None], patches), dim=1)
                tokens = tokens + self.cv_feat_fusers[level](context)
            tokens = block(tokens)
            if index in self.intermediate_layers_idx:
                features.append((tokens[:, 1:], tokens[:, 0]))
        return features


def _build_native(settings):
    """先构造固定官方架构，权重由调用方严格加载，不触发联网。"""
    from official_methods.methods.mvsanywhere.adapter import Adapter

    path = Path(settings["model_config"])
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("method") != "mvsanywhere":
        raise ValueError("shared_initializer.model_config 必须是 MVSAnywhere 配置")
    config = copy.deepcopy(config)
    config["_loading_shared_checkpoint"] = True
    config["model"].update(pretrained=False, weights="", da_weights="")
    return Adapter(config).native_model


def _load_projective_state(module, state):
    """分辨率变化仅重建解析像素网格，其余参数和缓冲仍严格加载。"""
    current = module.state_dict()
    buffers = dict(module.named_buffers())
    compatible = dict(state)
    regenerated = []
    for key in compatible:
        if (
            key.endswith("pix_coords_13N")
            and key in buffers
            and compatible[key].shape != current[key].shape
        ):
            compatible[key] = current[key]
            regenerated.append(key)
    module.load_state_dict(compatible, strict=True)
    return regenerated


class SharedPretrainedInitializer(nn.Module):
    """四目/缺相机共用一次原生前向，结果不持久缓存，也不读取监督标签。

    state 为本模块完整 state_dict；提供它时不会再读取任何外部初始化权重。
    fusion_weights / decoder_weights / readout_weights 可严格加载已适配的
    原生模块，不改变网络层；完整 state 恢复时不再读取外部权重。
    返回 native 是按 batch 顺序排列的原生输出字典列表，无可用参考时为 None。
    当前按样本执行，允许同一 batch 的有效相机集合不同。
    """

    def __init__(self, settings, state=None):
        super().__init__()
        self.reference_index = int(settings.get("reference_index", 2))
        if self.reference_index < 0:
            raise ValueError("reference_index 必须非负")
        self.native_model = _build_native(settings)
        self.native_model.depth_decoder = PrimaryDepthHead(
            self.native_model.depth_decoder, bool(settings.get("auxiliary_outputs", False))
        )
        if hasattr(self.native_model, "cost_volume_net"):
            self.native_model.cost_volume_net = SelectiveFusionEncoder(
                self.native_model.cost_volume_net, settings.get("fusion_skip_indices", [])
            )
        if state is not None:
            self.regenerated_pixel_grids = _load_projective_state(self, state)
        else:
            weights = settings.get("weights", "")
            if not weights:
                raise ValueError("冻结初值器必须提供已有 weights 或完整 state，禁止随机初始化")
            payload = torch.load(weights, map_location="cpu", weights_only=False, mmap=True)
            saved = payload.get("model", payload.get("state_dict", payload))
            if any(key.startswith("native_model.") for key in saved):
                self.regenerated_pixel_grids = _load_projective_state(self, saved)
            else:
                self.regenerated_pixel_grids = _load_projective_state(self.native_model, saved)
            if settings.get("decoder_weights") and settings.get("readout_weights"):
                raise ValueError("完整 decoder 已包含主输出头，请勿同时指定 readout_weights")
            for key, module_path in (
                ("fusion_weights", "cost_volume_net"),
                ("decoder_weights", "depth_decoder"),
                ("readout_weights", "depth_decoder.output_convs.0"),
            ):
                if settings.get(key):
                    payload = torch.load(
                        settings[key], map_location="cpu", weights_only=False, mmap=True,
                    )
                    module = self.native_model.get_submodule(module_path)
                    module.load_state_dict(payload.get("model", payload), strict=True)
        if settings.get("fast_cost_volume", False):
            self.native_model.cost_volume = self.native_model.cost_volume.to_fast()
        self.decoder_channels = self.native_model.depth_decoder.scratch.layer1_rn.out_channels
        self.requires_grad_(False)
        self.eval()
        if settings.get("cache_position_encoding", False):
            from .position_cache import install_position_cache

            install_position_cache(self.native_model.encoder.model)
            install_position_cache(self.native_model.cost_volume_net.model)

    def train(self, mode=True):
        super().train(False)
        return self

    def _prepare(self, inputs, batch_index, active):
        """保持官方的像素中心、源图排序、输入标准化和变换方向。"""
        from official_methods.methods.niantic_common import native_intrinsics

        rgb = inputs["rgb"][batch_index : batch_index + 1, active].float()
        intrinsics = inputs["K"][batch_index : batch_index + 1, active].float()
        transforms = inputs["T_cam_from_rig"][batch_index : batch_index + 1, active].float()
        reference = active.index(self.reference_index)
        poses = torch.linalg.inv(transforms)
        sources = torch.tensor(
            [index for index in range(len(active)) if index != reference], device=rgb.device
        )
        relative = transforms[:, reference, None] @ poses[:, sources]
        trace = relative[..., :3, :3].diagonal(dim1=-2, dim2=-1).sum(-1)
        rotation_distance = (2 * (1 - trace / 3)).clamp_min(0).sqrt()
        translation_distance = relative[..., :3, 3].norm(dim=-1)
        distance = (translation_distance.square() + rotation_distance.square()).sqrt()
        order = distance.argsort(dim=1, stable=True)[0]
        sources = sources[order]
        mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
        std = rgb.new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
        matching_k = native_intrinsics(intrinsics, 0.25)
        data = {
            "image_b3hw": (rgb - mean) / std,
            "cam_T_world_b44": transforms,
            "world_T_cam_b44": poses,
            "K_matching_b44": matching_k,
            "invK_matching_b44": torch.linalg.inv(matching_k),
        }
        current = {key: value[:, reference] for key, value in data.items()}
        source = {key: value[:, sources] for key, value in data.items()}
        physical_sources = [active[index] for index in sources.tolist()]
        return current, source, physical_sources

    @torch.no_grad()
    def forward(self, inputs):
        rgb = inputs["rgb"]
        batch, views, _, height, width = rgb.shape
        if self.reference_index >= views:
            raise ValueError("reference_index 超出输入相机数")
        options = self.native_model.run_opts
        if (height, width) != (options.image_height, options.image_width):
            raise ValueError("初值器 RGB 尺寸必须匹配 model_config；缩放时须同步修改内参")
        view_mask = inputs.get(
            "view_mask", torch.ones(batch, views, device=rgb.device, dtype=torch.bool)
        ).bool()
        image_mask = inputs.get("image_mask")

        depth = rgb.new_zeros((batch, 1, height, width), dtype=torch.float32)
        valid = torch.zeros_like(depth, dtype=torch.bool)
        matching = rgb.new_zeros((batch, views, 16, height // 4, width // 4), dtype=torch.float32)
        decoder = rgb.new_zeros(
            (batch, self.decoder_channels, height // 4, width // 4), dtype=torch.float32
        )
        native = []
        effective_mask = torch.zeros_like(view_mask)
        for index in range(batch):
            active = view_mask[index].nonzero(as_tuple=True)[0].tolist()
            if self.reference_index not in active or len(active) < 2:
                native.append(None)
                continue
            # 同中心纯旋转不提供米制三角化约束，也会使官方范围启发式退化。
            # 仅从本扩展的源集合中排除，不修改独立 baseline 的处理方式。
            poses = torch.linalg.inv(inputs["T_cam_from_rig"][index, active].float())
            centers = poses[:, :3, 3]
            baseline = (centers - centers[active.index(self.reference_index)]).norm(dim=-1)
            active = [camera for camera, distance in zip(active, baseline.tolist())
                      if camera == self.reference_index or distance > 1e-7]
            if len(active) < 2:
                native.append(None)
                continue
            if image_mask is not None and not bool(image_mask[index, active].all()):
                raise ValueError("官方候选体不支持有效相机内的像素缺失；请裁剪有效矩形或将整台相机标为缺失")
            current, source, physical_sources = self._prepare(inputs, index, active)
            captured = []
            hook = self.native_model.depth_decoder.scratch.refinenet2.register_forward_hook(
                lambda module, arguments, output: captured.append(output)
            )
            try:
                prediction = self.native_model(
                    "test", current, source,
                    unbatched_matching_encoder_forward=True,
                    return_mask=True,
                    estimate_depth_range=True,
                    num_refinement_steps=0,
                )
            finally:
                hook.remove()
            if len(captured) != 1:
                raise RuntimeError("单遍初值器必须且只能执行一次 DPT refinenet2")
            value = prediction["depth_pred_s0_b1hw"].float()
            if value.shape[-2:] != (height, width):
                raise RuntimeError("初值器要求官方全分辨率 prediction_scale=1")
            depth[index : index + 1] = value
            valid[index : index + 1] = torch.isfinite(value) & (value > 0)
            matching[index, self.reference_index] = current["matching_cur_feats"][0].float()
            matching[index, physical_sources] = source["matching_src_feats"][0].float()
            decoder[index : index + 1] = captured[0].float()
            effective_mask[index, active] = True
            native.append(prediction)

        return {
            "depth": depth,
            "valid": valid,
            "matching_features": matching,
            "decoder_features": decoder,
            "effective_view_mask": effective_mask,
            "native": native,
            "reference_index": self.reference_index,
        }
