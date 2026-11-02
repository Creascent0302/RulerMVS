"""独立稠密深度解码：候选证据压缩、分层图像融合和标定尺度细化。

这不是官方 DPT 的复刻。输入保留完整粗候选响应；输出为米制光轴深度。
valid 只表示参考图像中的有效预测，绝不表示该位置已经被多目观测确认。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from jar.encoder import norm
from jar.geometry import (
    camera_centers,
    image_texture,
    pixel_grid,
    project,
    safe_masked_softmax,
    sample_maps,
    unproject,
)
from jar.pretrained_matching import sample_pretrained_matching


def aligned_resample(features, source_stride, target_stride, size):
    """按原图 stride*i 格点采样；不使用图像 resize 的半像素平移。"""
    if source_stride == target_stride and tuple(features.shape[-2:]) == tuple(size):
        return features
    coordinates = pixel_grid(*size, device=features.device, stride=target_stride)
    coordinates = coordinates.reshape(1, -1, 2).expand(features.shape[0], -1, -1)
    sampled = sample_maps(features, coordinates, stride=source_stride, padding_mode="border")
    return (
        sampled.transpose(1, 2)
        .reshape(features.shape[0], features.shape[1], *size)
        .to(features.dtype)
    )


class ResidualImageBlock(nn.Module):
    def __init__(self, channels, lightweight=False):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, padding=1, groups=channels if lightweight else 1, bias=False
            ),
            norm(channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                1 if lightweight else 3,
                padding=0 if lightweight else 1,
                bias=False,
            ),
            norm(channels),
        )

    def forward(self, features):
        return F.gelu(features + self.body(features))


class CandidateContext(nn.Module):
    """每张图的低分辨率候选上下文；位置来自像素、rig射线和相机中心。"""

    def __init__(self, channels, layers, heads, far):
        super().__init__()
        self.far = far
        self.position = nn.Sequential(
            nn.Linear(8, channels), nn.GELU(), nn.Linear(channels, channels)
        )
        block = nn.TransformerEncoderLayer(
            channels,
            heads,
            dim_feedforward=channels * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, features, inputs, valid):
        batch, views = inputs["rgb"].shape[:2]
        height, width = features.shape[-2:]
        uv = pixel_grid(height, width, device=features.device, stride=16).reshape(-1, 2)
        image_h, image_w = inputs["rgb"].shape[-2:]
        normalized_uv = uv / uv.new_tensor([max(image_w - 1, 1), max(image_h - 1, 1)]) * 2 - 1
        positions = []
        for sample in range(batch):
            for view in range(views):
                intrinsic = inputs["K"][sample, view].detach().float()
                transform = inputs["T_cam_from_rig"][sample, view].detach().float()
                _, rays = unproject(uv, torch.ones_like(uv[:, 0]), intrinsic, transform)
                center = camera_centers(transform).expand(len(uv), -1) / self.far
                positions.append(torch.cat((normalized_uv, F.normalize(rays, dim=-1), center), -1))
        mask = valid.flatten(1).bool()
        # Attention不能接收整行全遮挡；仅开放一个零哨兵，输出仍由真实mask归零。
        safe_mask = mask.clone()
        safe_mask[:, 0] |= ~safe_mask.any(-1)
        tokens = features.flatten(2).transpose(1, 2) + self.position(torch.stack(positions))
        tokens = torch.where(mask[..., None], tokens, 0)
        result = self.encoder(tokens, src_key_padding_mask=~safe_mask)
        result = torch.where(mask[..., None], self.norm(result), 0)
        return result.transpose(1, 2).reshape_as(features)


class DenseCalibratedRefinement(nn.Module):
    """stride4上的局部候选细化，投影导数控制候选间距而非代替精确投影。"""

    def __init__(self, feature_channels, decoder_channels, near, far, settings):
        super().__init__()
        hidden = int(settings.get("hidden_dim", 32))
        self.chunk_size = int(settings.get("chunk_size", 1024))
        self.minimum_step = float(settings.get("min_step", 0.02))
        self.maximum_step = float(settings.get("max_step", 4.0))
        self.near, self.far = near, far
        self.stride = 4
        if hidden < 1 or self.chunk_size < 1 or not 0 < self.minimum_step <= self.maximum_step:
            raise ValueError("稠密几何细化的维度、chunk与步长范围不合法")
        self.descriptor = nn.Linear(feature_channels * 2 + 7, hidden)
        self.query = nn.Linear(decoder_channels, hidden)
        self.score = nn.Sequential(nn.GELU(), nn.Linear(hidden, 2))
        self.gate = nn.Linear(decoder_channels, 1)
        initial_gate = float(settings.get("initial_gate", 0.05))
        if not 0 < initial_gate < 1:
            raise ValueError("initial_gate 必须位于 (0,1)")
        nn.init.normal_(self.gate.weight, std=1e-3)
        nn.init.constant_(self.gate.bias, math.log(initial_gate / (1 - initial_gate)))
        self.register_buffer("offsets", torch.linspace(-1.0, 1.0, 9))

    def _chunk(
        self,
        uv,
        depth,
        query,
        reference,
        features,
        intrinsic,
        transform,
        view_mask,
        image_mask,
        image_size,
        texture=None,
        maximum_step=None,
        image_scale=None,
    ):
        count, views = len(uv), len(features)
        points, rays = unproject(uv, depth, intrinsic[reference], transform[reference])
        center_uv, center_z, _, jacobian = project(points, intrinsic, transform, image_size)
        # 不反传线性化步长；候选本身仍依赖当前深度并执行精确可微重投影。
        derivative = (jacobian * rays[None, :, None]).sum(-1).norm(dim=-1).detach()
        centers = camera_centers(transform)
        baseline = (centers - centers[reference]).norm(dim=-1)
        geometric = (baseline[:, None] > 1e-4) & (derivative > 1e-4) & (center_z > 0)
        geometric = geometric & view_mask[:, None] & view_mask[reference]
        geometric[reference] = False
        average_derivative = (derivative * geometric).sum(0) / geometric.sum(0).clamp_min(1)
        step = (
            (self.stride / average_derivative.clamp_min(1e-6))
            .clamp(self.minimum_step, self.maximum_step)
            .detach()
        )
        candidates = (depth[:, None] + step[:, None] * self.offsets).clamp(self.near, self.far)
        candidate_points = (
            points[:, None] + rays[:, None] * (candidates - depth[:, None])[..., None]
        )
        projected, source_z, valid, _ = project(
            candidate_points.reshape(-1, 3), intrinsic, transform, image_size, view_mask, image_mask
        )
        valid = valid.reshape(views, count, 9) & geometric[:, :, None]
        reference_feature = sample_maps(features[reference : reference + 1], uv[None], self.stride)[
            0
        ]
        reference_feature = F.normalize(reference_feature, dim=-1, eps=1e-6)
        sampled = sample_maps(features, projected, self.stride).reshape(views, count, 9, -1)
        sampled = F.normalize(sampled, dim=-1, eps=1e-6)
        product = sampled * reference_feature[None, :, None]
        difference = (sampled - reference_feature[None, :, None]).abs()

        reference_direction = F.normalize(rays, dim=-1)
        source_direction = F.normalize(candidate_points[None] - centers[:, None, None], dim=-1)
        ray_cosine = (source_direction * reference_direction[None, :, None]).sum(-1)
        displacement = (projected.reshape(views, count, 9, 2) - center_uv[:, :, None]) / self.stride
        metadata = torch.stack(
            [
                ((candidates - depth[:, None]) / step[:, None]).expand(views, -1, -1),
                torch.log1p(derivative / self.stride)[:, :, None].expand(-1, -1, 9),
                (baseline[:, None] / depth[None].clamp_min(self.near))[:, :, None].expand(
                    -1, -1, 9
                ),
                (source_z.reshape(views, count, 9).clamp_min(1e-4) / candidates[None]).log(),
                ray_cosine,
                displacement[..., 0],
                displacement[..., 1],
            ],
            dim=-1,
        ).clamp(-20, 20)
        descriptor = torch.cat((difference, product, metadata), dim=-1)
        descriptor = torch.where(valid[..., None], descriptor, 0)
        encoded = self.descriptor(descriptor) + self.query(query)[None, :, None]
        prediction = self.score(encoded)
        reliability = safe_masked_softmax(prediction[..., 1].float(), valid, dim=0)
        score = torch.where(valid, 4 * product.sum(-1) + prediction[..., 0].float(), 0)
        score = (score * reliability).sum(0)
        candidate_valid = valid.any(0)
        probability = safe_masked_softmax(score, candidate_valid, dim=-1)
        correction = (probability * (candidates - depth[:, None])).sum(-1)
        gate = self.gate(query).squeeze(-1).sigmoid() * candidate_valid.any(-1)
        refined = (depth + gate * correction).clamp(self.near, self.far)
        support = valid.any(-1).sum(0)
        return refined, gate, step, support

    def forward(self, depth, decoder_features, inputs, pyramid_features, reference_valid):
        batch, reference_count, height, width = depth.shape
        views = inputs["rgb"].shape[1]
        references = inputs.get("_reference_indices", range(reference_count))
        uv = pixel_grid(height, width, device=depth.device, stride=self.stride).reshape(-1, 2)
        view_mask = inputs.get(
            "view_mask", torch.ones(batch, views, device=depth.device, dtype=torch.bool)
        )
        all_depth, all_gate, all_step, all_support = [], [], [], []
        for sample in range(batch):
            sample_outputs = [[], [], [], []]
            intrinsic = inputs["K"][sample].detach().float()
            transform = inputs["T_cam_from_rig"][sample].detach().float()
            image_mask = inputs.get("image_mask")
            image_mask = None if image_mask is None else image_mask[sample]
            texture = inputs.get("_dense_refinement_texture")
            texture = None if texture is None else texture[sample]
            matching_scale = inputs.get("_pretrained_matching_scale")
            matching_scale = None if matching_scale is None else matching_scale[sample]
            for output_index, reference in enumerate(references):
                original = depth[sample, output_index].reshape(-1)
                tokens = decoder_features[sample, output_index].flatten(1).transpose(0, 1)
                field_options = {}
                if "_candidate_fields" in inputs:
                    field_options["candidate_field"] = inputs["_candidate_fields"][sample][reference]
                pieces = [[], [], [], []]
                for start in range(0, len(uv), self.chunk_size):
                    stop = start + self.chunk_size
                    values = self._chunk(
                        uv[start:stop],
                        original[start:stop],
                        tokens[start:stop],
                        reference,
                        pyramid_features[sample].float(),
                        intrinsic,
                        transform,
                        view_mask[sample],
                        image_mask,
                        inputs["rgb"].shape[-2:],
                        texture,
                        inputs.get("_dense_refinement_max_step"),
                        matching_scale,
                        **field_options,
                    )
                    for parts, value in zip(pieces, values):
                        parts.append(value)
                for destination, parts in zip(sample_outputs, pieces):
                    destination.append(torch.cat(parts).reshape(height, width))
            for destination, values in zip(
                (all_depth, all_gate, all_step, all_support), sample_outputs
            ):
                destination.append(torch.stack(values))
        return {
            "depth": torch.where(reference_valid, torch.stack(all_depth), depth),
            "gate": torch.stack(all_gate) * reference_valid,
            "step_m": torch.stack(all_step),
            "source_support": torch.stack(all_support) * reference_valid,
        }


class DensePretrainedRefinement(DenseCalibratedRefinement):
    """重新读取冻结Hero描述子；只在可观测的局部峰附近更新当前深度。

    原生描述子的格点中心为1.5+4*j，不能按本地CNN的4*j采样。
    标定导数只确定搜索跨度，九个候选始终精确投影。纹理、视差和
    候选分数变化共同决定是否允许更新，不能仅靠网络产生置信度。
    """

    def __init__(self, decoder_channels, near, far, settings):
        super().__init__(16, decoder_channels, near, far, settings)
        hidden = self.query.out_features
        self.texture_threshold = float(settings.get("texture_threshold", 0.002))
        self.score_temperature = float(settings.get("score_temperature", 12.0))
        self.contrast_scale = float(settings.get("contrast_scale", 0.05))
        self.minimum_contrast = float(settings.get("minimum_contrast", 0.001))
        self.step_pixels = float(settings.get("step_pixels", 1.0))
        self.gate_multiplier = float(settings.get("gate_multiplier", 1.0))
        if not math.isfinite(self.gate_multiplier) or self.gate_multiplier <= 0:
            raise ValueError("gate_multiplier须为有限正数")
        self.gate_source = settings.get("gate_source", "learned")
        if self.gate_source not in ("learned", "evidence"):
            raise ValueError("gate_source须为learned或evidence")
        if self.gate_source == "evidence":
            self.gate.requires_grad_(False)
        self.pass_max_steps = settings.get("pass_max_steps")
        if self.pass_max_steps is not None:
            if not isinstance(self.pass_max_steps, list) or not self.pass_max_steps:
                raise ValueError("pass_max_steps须为空缺或非空数值列表")
            self.pass_max_steps = [float(value) for value in self.pass_max_steps]
            if any(
                not math.isfinite(value) or value < self.minimum_step
                for value in self.pass_max_steps
            ):
                raise ValueError("每轮最大步长须有限且不小于min_step")
        if min(self.score_temperature, self.contrast_scale, self.step_pixels) <= 0:
            raise ValueError("预训练细化的温度、对比尺度、像素步长须为正")
        if min(self.texture_threshold, self.minimum_contrast) < 0:
            raise ValueError("纹理与候选对比阈值须非负")
        self.depth_condition = nn.Linear(3, hidden)
        # 学习残差从小扰动开始，避免随机评分覆盖预训练匹配度量。
        nn.init.normal_(self.score[-1].weight, std=1e-3)
        nn.init.zeros_(self.score[-1].bias)
        self.observation_aggregation = settings.get("observation_aggregation", "reference")
        if self.observation_aggregation == "camera_consensus":
            from .camera_consensus import CameraConsensus

            self.camera_consensus = CameraConsensus(**settings.get("camera_consensus", {}))
        elif self.observation_aggregation != "reference":
            raise ValueError("observation_aggregation须为reference或camera_consensus")

    def _candidate_score(self, score, candidates, uv, reference):
        """研究扩展点：可继承全局候选证据；默认保持原评分逐值不变。"""
        return score

    def _observation_scores(self, source, ref, prediction, valid, reference, reference_valid):
        reliability = safe_masked_softmax(prediction[..., 1].float(), valid, dim=0)
        cosine = (source * ref[None, :, None]).sum(-1)
        raw_score = (torch.where(valid, cosine, 0) * reliability).sum(0)
        learned_score = (
            torch.where(valid, prediction[..., 0].float().tanh(), 0) * reliability
        ).sum(0)
        if self.observation_aggregation == "reference":
            return raw_score, learned_score, valid.any(0), valid
        views, count, candidates, channels = source.shape
        values = source.clone()
        values[reference] = ref[:, None].expand(-1, candidates, -1)
        camera_valid = valid.clone()
        camera_valid[reference] = reference_valid[:, None].expand(-1, candidates)
        consensus = self.camera_consensus(
            values.reshape(1, views, -1, channels), camera_valid.reshape(1, views, -1)
        )
        # 两个等权视图时1-2*variance恰好等于原余弦相似度。
        score = 1 - 2 * consensus["identity_energy"][0].reshape(count, candidates)
        supported = consensus["valid"][0].reshape(count, candidates)
        accepted = consensus["weights"][0].reshape(views, count, candidates) > 1e-6
        accepted[reference] = False
        return torch.where(supported, score, 0.0), learned_score, supported, accepted

    def _chunk(
        self,
        uv,
        depth,
        query,
        reference,
        features,
        intrinsic,
        transform,
        view_mask,
        image_mask,
        image_size,
        texture=None,
        maximum_step=None,
        image_scale=None,
        candidate_field=None,
    ):
        count, views = len(uv), len(features)
        points, rays = unproject(uv, depth, intrinsic[reference], transform[reference])
        center_uv, center_z, _, jacobian = project(points, intrinsic, transform, image_size)
        derivative = (jacobian * rays[None, :, None]).sum(-1).norm(dim=-1).detach()
        centers = camera_centers(transform)
        baseline = (centers - centers[reference]).norm(dim=-1)
        geometric = (baseline[:, None] > 1e-4) & (derivative > 1e-4) & (center_z > 0)
        geometric &= view_mask[:, None] & view_mask[reference]
        geometric[reference] = False
        average_derivative = (derivative * geometric).sum(0) / geometric.sum(0).clamp_min(1)
        step = self.step_pixels * self.stride / average_derivative.clamp_min(1e-6)
        maximum = self.maximum_step if maximum_step is None else maximum_step
        step = step.clamp(self.minimum_step, maximum).detach()
        candidates = (depth[:, None] + step[:, None] * self.offsets).clamp(self.near, self.far)
        candidate_points = (
            points[:, None] + rays[:, None] * (candidates - depth[:, None])[..., None]
        )
        projected, source_z, valid, _ = project(
            candidate_points.reshape(-1, 3), intrinsic, transform, image_size, view_mask, image_mask
        )
        valid = valid.reshape(views, count, 9) & geometric[:, :, None]
        # texture由forward从输入RGB计算，既不是标签，也不是可学习置信度。
        ref_texture = sample_maps(texture[reference : reference + 1], uv[None])[0, :, 0]
        source_texture = sample_maps(texture, projected).reshape(views, count, 9)
        if self.observation_aggregation == "reference":
            valid &= ref_texture[None, :, None] > self.texture_threshold
        valid &= source_texture > self.texture_threshold

        ref = sample_pretrained_matching(
            features[reference : reference + 1], uv[None], image_scale
        )[0]
        ref = F.normalize(ref, dim=-1, eps=1e-6)
        source = sample_pretrained_matching(features, projected, image_scale).reshape(
            views, count, 9, 16
        )
        source = F.normalize(source, dim=-1, eps=1e-6)
        product = source * ref[None, :, None]
        difference = (source - ref[None, :, None]).abs()
        reference_direction = F.normalize(rays, dim=-1)
        source_direction = F.normalize(candidate_points[None] - centers[:, None, None], dim=-1)
        displacement = (projected.reshape(views, count, 9, 2) - center_uv[:, :, None]) / self.stride
        metadata = torch.stack(
            (
                ((candidates - depth[:, None]) / step[:, None]).expand(views, -1, -1),
                torch.log1p(derivative / self.stride)[:, :, None].expand(-1, -1, 9),
                (baseline[:, None] / depth[None].clamp_min(self.near))[:, :, None].expand(
                    -1, -1, 9
                ),
                (source_z.reshape(views, count, 9).clamp_min(1e-4) / candidates[None]).log(),
                (source_direction * reference_direction[None, :, None]).sum(-1),
                displacement[..., 0],
                displacement[..., 1],
            ),
            -1,
        ).clamp(-20, 20)
        description = torch.cat((difference, product, metadata), -1)
        description = torch.where(valid[..., None], description, 0)
        depth_condition = torch.stack(
            (
                (depth.log() - math.log(self.near)) / math.log(self.far / self.near),
                torch.log1p(step / depth.clamp_min(self.near)),
                geometric.float().mean(0),
            ),
            -1,
        )
        query_features = self.query(query) + self.depth_condition(depth_condition)
        prediction = self.score(self.descriptor(description) + query_features[None, :, None])
        raw_score, learned_score, candidate_valid, accepted_sources = self._observation_scores(
            source,
            ref,
            prediction,
            valid,
            reference,
            (ref_texture > self.texture_threshold) & view_mask[reference],
        )
        score = self.score_temperature * raw_score + learned_score
        score = self._candidate_score(score, candidates, uv, reference)
        if candidate_field is not None:
            from .candidate_field import sample_candidate_field

            score = score + sample_candidate_field(candidate_field, candidates, uv)
        # 不跨多个局部峰求均值：先选获胜峰，再在相邻三个候选间连续定位。
        winner = score.masked_fill(~candidate_valid, -1e4).argmax(-1)
        local_peak = (torch.arange(9, device=depth.device)[None] - winner[:, None]).abs() <= 1
        probability = safe_masked_softmax(score, candidate_valid & local_peak, dim=-1)
        correction = (probability * (candidates - depth[:, None])).sum(-1)
        maximum = raw_score.masked_fill(~candidate_valid, -1e4).amax(-1)
        minimum = raw_score.masked_fill(~candidate_valid, 1e4).amin(-1)
        contrast = (maximum - minimum).clamp_min(0)
        observable = candidate_valid.sum(-1) >= 2
        observable &= contrast > self.minimum_contrast
        evidence = contrast / (contrast + self.contrast_scale)
        if self.gate_source == "evidence":
            gate = evidence * observable
        else:
            gate = self.gate(query).squeeze(-1).sigmoid() * evidence * observable
        if self.gate_multiplier != 1.0:
            gate = (gate * self.gate_multiplier).clamp(0, 1)
        refined = (depth + gate * correction).clamp(self.near, self.far)
        support = accepted_sources.any(-1).sum(0)
        return refined, gate, step, support

    def forward(self, depth, decoder_features, inputs, matching_features, reference_valid):
        if matching_features.shape[:3] != (*inputs["rgb"].shape[:2], 16):
            raise ValueError("预训练细化须提供[B,N,16,H4,W4]原生Hero描述子")
        batch, views, _, height, width = inputs["rgb"].shape
        texture = inputs.get("_texture")
        if texture is None:
            texture = image_texture(inputs["rgb"].flatten(0, 1)).reshape(
                batch, views, 1, height, width
            )
        # 复用父类的逐样本/视图/chunk遍历；不把逐帧张量保存在模型属性中。
        work = {**inputs, "_dense_refinement_texture": texture}
        if self.pass_max_steps is None:
            # 原默认单轮路径不增加任何张量运算，保留已有权重的逐值结果。
            result = super().forward(
                depth, decoder_features, work, matching_features, reference_valid
            )
        else:
            current = depth
            passes = []
            for maximum_step in self.pass_max_steps:
                pass_inputs = {**work, "_dense_refinement_max_step": maximum_step}
                result = super().forward(
                    current, decoder_features, pass_inputs, matching_features, reference_valid
                )
                passes.append(
                    {
                        "max_step_m": maximum_step,
                        "mean_gate": result["gate"].detach().mean(),
                        "mean_absolute_update_m": (result["depth"] - current).detach().abs().mean(),
                        "mean_source_support": result["source_support"].detach().float().mean(),
                    }
                )
                current = result["depth"]
            result["passes"] = passes
        result["depth_before"] = depth
        result["residual_m"] = result["depth"] - depth
        return result


class DenseDepthDecoder(nn.Module):
    """完整粗候选→32/16/8/4层级解码→标定细化→原图连续深度。"""

    strides = (32, 16, 8, 4)

    def __init__(self, cfg, checkpoint_state=None):
        super().__init__()
        model = cfg["model"]
        settings = model.get("dense_decoder", {})
        self.depth_initialization = settings.get("depth_initialization", "candidate_mean")
        if self.depth_initialization not in ("candidate_mean", "learned"):
            raise ValueError("depth_initialization必须为candidate_mean或learned")
        self.channels = int(model.get("feature_dim", 64))
        self.bins = int(model.get("depth_bins", 48))
        hidden = int(settings.get("hidden_dim", 64))
        blocks = int(settings.get("blocks_per_stage", 2))
        full_hidden = int(settings.get("full_resolution_dim", 16))
        self.near, self.far = float(model.get("near", 2)), float(model.get("far", 80))
        if min(hidden, full_hidden, blocks) < 1 or self.bins < 2 or not 0 < self.near < self.far:
            raise ValueError("稠密解码器维度、候选数或深度范围不合法")
        self.log_near = math.log(self.near)
        self.log_range = math.log(self.far / self.near)
        self.candidate_encoder = nn.Sequential(
            nn.Conv2d(self.bins * 4 + 5, hidden, 1, bias=False),
            norm(hidden),
            nn.GELU(),
            ResidualImageBlock(hidden),
        )
        context_layers = int(settings.get("context_layers", 2))
        context_heads = int(settings.get("context_heads", 4))
        if context_layers < 0 or context_heads < 1 or (context_layers and hidden % context_heads):
            raise ValueError("context_layers需非负，hidden_dim需能被context_heads整除")
        if context_layers:
            self.candidate_context = CandidateContext(
                hidden, context_layers, context_heads, self.far
            )
        self.image_adapters = nn.ModuleDict()
        self.context_adapters = nn.ModuleDict()
        self.fusions = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        for stride in self.strides:
            self.image_adapters[str(stride)] = nn.Sequential(
                nn.Conv2d(self.channels, hidden, 1, bias=False),
                norm(hidden),
                nn.GELU(),
            )
            self.context_adapters[str(stride)] = nn.Sequential(
                nn.Conv2d(self.channels, hidden, 1, bias=False),
                norm(hidden),
                nn.GELU(),
            )
            self.fusions[str(stride)] = nn.Sequential(
                nn.Conv2d(hidden * 4, hidden, 1, bias=False),
                norm(hidden),
                nn.GELU(),
                *[ResidualImageBlock(hidden) for _ in range(blocks)],
            )
            self.heads[str(stride)] = nn.Conv2d(hidden, 2, 1)
        self.full_projection = nn.Conv2d(hidden, full_hidden, 1)
        self.use_rgb = bool(settings.get("rgb_skip", True))
        if self.use_rgb:
            self.rgb_encoder = nn.Sequential(
                nn.Conv2d(3, full_hidden, 3, padding=1, bias=False),
                norm(full_hidden),
                nn.GELU(),
            )
        self.full_fusion = nn.Sequential(
            nn.Conv2d(full_hidden * (2 if self.use_rgb else 1), full_hidden, 1, bias=False),
            norm(full_hidden),
            nn.GELU(),
            *[ResidualImageBlock(full_hidden, lightweight=True) for _ in range(blocks)],
        )
        self.heads["1"] = nn.Conv2d(full_hidden, 2, 1)
        for head in self.heads.values():
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)
        refinement = settings.get("calibrated_refinement", {})
        self.refinement_source = refinement.get("feature_source", "local")
        if self.refinement_source not in ("local", "pretrained"):
            raise ValueError("calibrated_refinement.feature_source须为local或pretrained")
        if refinement.get("enabled", True):
            if self.refinement_source == "pretrained":
                self.refinement = DensePretrainedRefinement(hidden, self.near, self.far, refinement)
            else:
                self.refinement = DenseCalibratedRefinement(
                    self.channels, hidden, self.near, self.far, refinement
                )
        recovery_settings = settings.get("surface_recovery", {})
        if recovery_settings.get("enabled", False):
            if not hasattr(self, "refinement") or self.refinement_source != "pretrained":
                raise ValueError("surface_recovery需要预训练描述子细化器")
            from .spatial_surface_recovery import SpatialRiskHead

            self.surface_proposal_mode = recovery_settings.get("proposal_mode", "legacy")
            self.surface_proposal_count = int(recovery_settings.get("proposal_count", 3))
            self.surface_peak_separation_m = float(
                recovery_settings.get("peak_separation_m", 1.0)
            )
            if self.surface_peak_separation_m <= 0:
                raise ValueError("surface_recovery.peak_separation_m必须为正数")
            self.surface_consensus_weight = float(
                recovery_settings.get("view_consensus_weight", 0.0)
            )
            self.surface_recovery = SpatialRiskHead()
            prefix = "dense_decoder.surface_recovery."
            saved = {
                key[len(prefix):]: value
                for key, value in (checkpoint_state or {}).items()
                if key.startswith(prefix)
            }
            if saved:
                self.surface_recovery.load_state_dict(saved, strict=True)
            elif recovery_settings.get("weights"):
                record = torch.load(recovery_settings["weights"], map_location="cpu", weights_only=False)
                self.surface_recovery.load_state_dict(record["model"], strict=True)
            elif checkpoint_state is not None:
                raise ValueError("旧检查点缺少surface_recovery参数；需给出训练好的独立权重")
        reader_settings = settings.get("raw_feature_reader", {})
        if reader_settings.get("enabled", False):
            if not hasattr(self, "surface_recovery"):
                raise ValueError("原始特征读取首版需要 surface_recovery 候选接口")
            from .raw_feature_reader import RawFeatureSurfaceReader

            with torch.random.fork_rng(devices=[]):
                self.raw_feature_reader = RawFeatureSurfaceReader(reader_settings)
            prefix = "dense_decoder.raw_feature_reader."
            saved = {
                key[len(prefix):]: value
                for key, value in (checkpoint_state or {}).items()
                if key.startswith(prefix)
            }
            if saved:
                self.raw_feature_reader.load_state_dict(saved, strict=True)
            elif reader_settings.get("weights"):
                record = torch.load(
                    reader_settings["weights"], map_location="cpu", weights_only=False
                )
                self.raw_feature_reader.load_state_dict(record["model"], strict=True)
        posterior_settings = settings.get("posterior_readout", {})
        if posterior_settings.get("enabled", False):
            from .posterior_readout import PosteriorDepthReadout

            with torch.random.fork_rng(devices=[]):
                self.posterior_readout = PosteriorDepthReadout(
                    hidden, self.bins, self.near, self.far, posterior_settings
                )

    def _candidate_packet(self, coarse):
        logits = coarse["logits"].float()
        inverse = coarse["inverse_depths"].float()
        if inverse.ndim != 1 or len(inverse) != self.bins or logits.shape[2] != self.bins + 1:
            raise ValueError("粗候选必须是depth_bins个逆深度以及额外一个no-match通道")
        if not bool(torch.isfinite(inverse).all() & (inverse > 0).all()):
            raise ValueError("物理逆深度必须为有限正数")
        inverse, order = inverse.sort()
        scores = logits[:, :, :-1].index_select(2, order)
        valid = torch.isfinite(scores) & (scores > -9999)
        clean = torch.nan_to_num(scores, nan=0.0, posinf=80.0, neginf=-80.0).clamp(-80, 80)
        no_match = torch.nan_to_num(logits[:, :, -1:], nan=0.0, posinf=80.0, neginf=-80.0).clamp(
            -80, 80
        )
        probability = safe_masked_softmax(
            torch.cat((clean, no_match), dim=2),
            torch.cat((valid, torch.ones_like(no_match, dtype=torch.bool)), dim=2),
            dim=2,
        )
        conditional = safe_masked_softmax(clean, valid, dim=2)
        maximum = clean.masked_fill(~valid, -80).amax(2, keepdim=True)
        relative = (clean - maximum).clamp(-20, 0) / 20
        physical = ((inverse - 1 / self.far) / (1 / self.near - 1 / self.far))[
            None, None, :, None, None
        ]
        physical = physical.expand_as(scores)
        mean = (conditional * physical).sum(2, keepdim=True)
        deviation = ((conditional * (physical - mean).square()).sum(2, keepdim=True) + 1e-8).sqrt()
        entropy = -(conditional * conditional.clamp_min(1e-8).log()).sum(
            2, keepdim=True
        ) / math.log(self.bins)
        summary = torch.cat(
            (probability[:, :, -1:], mean, deviation, entropy, valid.float().mean(2, keepdim=True)),
            2,
        )
        packet = torch.cat(
            (relative * valid, probability[:, :, :-1], valid.float(), physical * valid, summary), 2
        )
        expected = (conditional * (-inverse.log())[None, None, :, None, None]).sum(2, keepdim=True)
        expected = torch.where(
            valid.any(2, keepdim=True), expected, self.log_near + self.log_range / 2
        )
        code = torch.logit(((expected - self.log_near) / self.log_range).clamp(0.02, 0.98))
        return packet.flatten(0, 1), code.flatten(0, 1)

    @staticmethod
    def _valid_image(inputs):
        rgb = inputs["rgb"]
        batch, views, _, height, width = rgb.shape
        view_mask = inputs.get(
            "view_mask", torch.ones(batch, views, device=rgb.device, dtype=torch.bool)
        )
        mask = inputs.get(
            "image_mask",
            torch.ones(batch, views, height, width, device=rgb.device, dtype=torch.bool),
        )
        mask = mask.bool() & view_mask[:, :, None, None].bool()
        if "image_sizes" in inputs:
            uv = pixel_grid(height, width, device=rgb.device)
            sizes = inputs["image_sizes"]
            mask = (
                mask
                & (uv[..., 0] < sizes[..., 1, None, None])
                & (uv[..., 1] < sizes[..., 0, None, None])
            )
        return mask

    def _depth_from_code(self, code, batch, views, references=None, source_views=None):
        if references is None:
            return (self.log_near + self.log_range * code.sigmoid()).exp()
        # 非线性仍使用原相机槽位，避免不同SIMD分组的微小舍入触发边缘投影mask变化。
        original = code.reshape(batch, views, *code.shape[1:])
        full = code.new_zeros((batch, source_views, *code.shape[1:]))
        full = full.index_copy(1, references, original).flatten(0, 1)
        depth = (self.log_near + self.log_range * full.sigmoid()).exp()
        return (
            depth.reshape(batch, source_views, *code.shape[1:])
            .index_select(1, references)
            .flatten(0, 1)
        )

    def _prediction(
        self, features, code, stride, mask, batch, views, references=None, source_views=None
    ):
        head = self.heads[str(stride)](features).float()
        code = code.float() + head[:, :1]
        depth = self._depth_from_code(code, batch, views, references, source_views)
        valid = mask & torch.isfinite(depth)
        stage = {
            "depth": torch.where(valid, depth, 0).reshape(batch, views, *depth.shape[-2:]),
            "stride": stride,
            "confidence_logits": head[:, 1].reshape(batch, views, *depth.shape[-2:]),
            "valid": valid.reshape(batch, views, *depth.shape[-2:]),
        }
        return stage, code

    def _recover_initial(self, depth, valid, source_inputs):
        from .spatial_surface_recovery import build_inference_features

        references = source_inputs.get("_reference_indices", tuple(range(depth.shape[1])))
        selected, switches, candidates, logits, candidate_valid = [], [], [], [], []
        for sample in range(depth.shape[0]):
            sample_depth, sample_switch = [], []
            sample_candidates, sample_logits, sample_valid = [], [], []
            for slot, reference in enumerate(references):
                field = source_inputs["_candidate_fields"][sample][reference]
                entry = {
                    "candidate_valid": field["valid"],
                    "inverse_depths": field["inverse_depths"],
                    "hero": field["recovery"]["hero"],
                    "source_cosine": field["recovery"]["source_cosine"],
                    "source_valid": field["recovery"]["source_valid"],
                }
                if "proposal_prior" in field["recovery"]:
                    entry["proposal_prior"] = field["recovery"]["proposal_prior"]
                if self.surface_proposal_mode == "coarse_prior" and not hasattr(self, "proposal_observer"):
                    raise ValueError("coarse_prior 必须安装精确 proposal_observer，不能将先验当观测")
                proposal = build_inference_features(
                    entry, field["recovery"]["scores"], depth[sample, slot],
                    proposal_mode=self.surface_proposal_mode,
                    proposal_count=self.surface_proposal_count,
                    view_consensus=self.surface_consensus_weight != 0.0,
                    near=self.near,
                    far=self.far,
                    peak_separation_m=self.surface_peak_separation_m,
                )
                if hasattr(self, "proposal_observer"):
                    proposal = self.proposal_observer(
                        proposal, entry, source_inputs, sample, reference
                    )
                # 独立小头按FP16缓存训练；推理也采用同样的特征量化。
                features = proposal["features"].to(torch.float16).float()
                scores = self.surface_recovery(features, proposal["candidate_valid"])
                if self.surface_consensus_weight != 0.0:
                    scores = scores + self.surface_consensus_weight * proposal["view_consensus"]
                if hasattr(self, "raw_feature_reader"):
                    raw_maps = source_inputs.get("_foundation_maps")
                    if raw_maps is None:
                        raise ValueError("原始特征读取需要本次编码提供 _foundation_maps")
                    memory = torch.stack([item[sample, reference] for item in raw_maps])
                    scores = self.raw_feature_reader(
                        features, proposal["candidate_valid"], memory, scores
                    )
                choice = scores.argmax(0)
                switched = valid[sample, slot] & (choice != 0)
                candidate = proposal["candidate_depth"].gather(0, choice[None])[0]
                sample_depth.append(torch.where(switched, candidate, depth[sample, slot]))
                sample_switch.append(switched)
                sample_candidates.append(proposal["candidate_depth"])
                sample_logits.append(scores)
                sample_valid.append(proposal["candidate_valid"])
            selected.append(torch.stack(sample_depth))
            switches.append(torch.stack(sample_switch))
            candidates.append(torch.stack(sample_candidates))
            logits.append(torch.stack(sample_logits))
            candidate_valid.append(torch.stack(sample_valid))
        return {
            "depth": torch.stack(selected),
            "switch": torch.stack(switches),
            "candidate_depth": torch.stack(candidates),
            "logits": torch.stack(logits),
            "candidate_valid": torch.stack(candidate_valid),
        }

    def forward(self, inputs, pyramid, coarse, context=None):
        source_inputs, source_pyramid = inputs, pyramid
        references = inputs.get("_reference_indices")
        source_batch, source_views = inputs["rgb"].shape[:2]
        packet, code = self._candidate_packet(coarse)
        indices = None
        if references is not None:
            indices = torch.as_tensor(references, device=inputs["rgb"].device, dtype=torch.long)
            # 图像解码只读取参考图；匹配/细化始终保留所有源相机。
            inputs = dict(inputs)
            for key in ("rgb", "K", "T_cam_from_rig", "view_mask", "image_mask", "image_sizes"):
                if key in inputs:
                    inputs[key] = inputs[key].index_select(1, indices)
            pyramid = {
                stride: feature.index_select(1, indices) for stride, feature in pyramid.items()
            }
            context = {
                stride: feature.index_select(1, indices)
                for stride, feature in (context or {}).items()
            }
            packet = (
                packet.reshape(source_batch, source_views, *packet.shape[1:])
                .index_select(1, indices)
                .flatten(0, 1)
            )
            code = (
                code.reshape(source_batch, source_views, *code.shape[1:])
                .index_select(1, indices)
                .flatten(0, 1)
            )
            coarse = {
                key: (
                    value.index_select(1, indices)
                    if torch.is_tensor(value)
                    and value.ndim >= 2
                    and value.shape[:2] == (source_batch, source_views)
                    else value
                )
                for key, value in coarse.items()
            }
        rgb = inputs["rgb"]
        batch, views, _, height, width = rgb.shape
        valid_image = self._valid_image(inputs)
        flat_mask = valid_image.flatten(0, 1)[:, None].float()
        context = context or {}
        if self.depth_initialization == "learned":
            # 第一层从融合特征预测完整log-depth；候选证据仍通过packet进入各级。
            # 零对应配置深度范围的对数中点；后续级在已学习的深度编码上预测残差。
            code = torch.zeros_like(code)
        coarse_stride = int(coarse.get("stride", 16))
        cost = self.candidate_encoder(packet)
        if hasattr(self, "candidate_context"):
            context_size = pyramid[16].shape[-2:]
            low_cost = aligned_resample(cost, coarse_stride, 16, context_size)
            cost_mask = aligned_resample(flat_mask, 1, 16, context_size) > 0.999
            global_context = self.candidate_context(low_cost, inputs, cost_mask)
            if coarse_stride == 16:
                # 默认分支直接保留旧计算，避免减加造成数值差异。
                cost = global_context
            else:
                # 只回传低分辨率上下文的改变量，保留原格点的细匹配证据。
                cost = cost + aligned_resample(
                    global_context - low_cost, 16, coarse_stride, cost.shape[-2:]
                )
        previous, previous_stride = None, coarse_stride
        stages = []
        refinement = None
        posterior = None
        recovery = None
        for stride in self.strides:
            feature = pyramid[stride]
            size = feature.shape[-2:]
            mask = aligned_resample(flat_mask, 1, stride, size) > 0.999
            local = feature.flatten(0, 1)
            extra = context.get(stride, torch.zeros_like(feature)).flatten(0, 1)
            image_features = self.image_adapters[str(stride)](torch.where(mask, local, 0))
            semantic_features = self.context_adapters[str(stride)](torch.where(mask, extra, 0))
            inherited = (
                torch.zeros_like(image_features)
                if previous is None
                else aligned_resample(previous, previous_stride, stride, size)
            )
            evidence = aligned_resample(cost, coarse_stride, stride, size)
            fused = self.fusions[str(stride)](
                torch.cat((image_features, semantic_features, inherited, evidence), 1)
            )
            code = aligned_resample(code, previous_stride, stride, size)
            stage, code = self._prediction(
                fused, code, stride, mask, batch, views, indices, source_views
            )
            if stride == 4 and hasattr(self, "refinement"):
                internal_depth = self._depth_from_code(
                    code, batch, views, indices, source_views
                ).reshape(batch, views, *size)
                if hasattr(self, "surface_recovery"):
                    recovery = self._recover_initial(internal_depth, stage["valid"], source_inputs)
                    internal_depth = recovery["depth"]
                matching = source_pyramid[4]
                if self.refinement_source == "pretrained":
                    matching = source_inputs.get("_pretrained_matching_features")
                    if matching is None:
                        raise ValueError("预训练稠密细化需要_pretrained_matching_features")
                refinement = self.refinement(
                    internal_depth,
                    fused.reshape(batch, views, *fused.shape[1:]),
                    {
                        **source_inputs,
                        "image_mask": (
                            valid_image if references is None else self._valid_image(source_inputs)
                        ),
                    },
                    matching,
                    stage["valid"],
                )
                refined = refinement["depth"]
                stage["depth"] = torch.where(stage["valid"], refined, 0)
                code_depth = refined
                if references is not None:
                    code_depth = refined.new_ones((batch, source_views, *size)).index_copy(
                        1, indices, refined
                    )
                code = torch.logit(
                    ((code_depth.log() - self.log_near) / self.log_range).clamp(1e-5, 1 - 1e-5)
                )
                if references is not None:
                    code = code.index_select(1, indices)
                code = code.flatten(0, 1)[:, None]
            if stride == 4 and hasattr(self, "posterior_readout"):
                posterior = self.posterior_readout(
                    fused, stage["depth"].flatten(0, 1)[:, None], coarse, stride
                )
                stage["depth"] = torch.where(stage["valid"], posterior["depth"], 0)
                # code继续保留原连续预测，仅用于最终单元内偏移；不能覆盖后验选峰。
            stages.append(stage)
            previous, previous_stride = fused, stride
        full_features = aligned_resample(self.full_projection(previous), 4, 1, (height, width))
        if self.use_rgb:
            image = torch.where(valid_image[:, :, None], rgb, 0).flatten(0, 1)
            full_features = torch.cat((full_features, self.rgb_encoder(image)), dim=1)
        full_features = self.full_fusion(full_features)
        code = aligned_resample(code, 4, 1, (height, width))
        stage, _ = self._prediction(
            full_features, code, 1, flat_mask.bool(), batch, views, indices, source_views
        )
        posterior_full = None
        if posterior is not None:
            posterior_full = self.posterior_readout.full_resolution(posterior, stage["depth"])
            stage["depth"] = torch.where(stage["valid"], posterior_full["depth"], 0)
        stages.append(stage)
        result = {
            "depth": stage["depth"],
            "valid": stage["valid"],
            "confidence": stage["confidence_logits"].sigmoid() * stage["valid"],
            "confidence_logits": stage["confidence_logits"],
            "stages": stages,
        }
        if refinement is not None:
            result["refinement"] = refinement
        if recovery is not None:
            result["surface_recovery"] = recovery
        if posterior is not None:
            result["posterior"] = posterior
            result["posterior_full"] = posterior_full
        if references is not None:

            def restore(value):
                if isinstance(value, dict):
                    return {key: restore(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [restore(item) for item in value]
                if torch.is_tensor(value) and value.ndim >= 2 and value.shape[:2] == (batch, views):
                    full = value.new_zeros((batch, source_views, *value.shape[2:]))
                    return full.index_copy(1, indices, value)
                return value

            result = restore(result)
            result["reference_indices"] = tuple(references)
        return result
