"""独立公制候选解码器：全局选面 → 精确细格取证 → 连续深度。

只接收冻结匹配描述子与四层单图特征，不调用旧CNN、初值器、候选场、K9
或GN。虚拟节点是计算区域；输出深度仍是图像假设，不能冒充已验证表面。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from jar.depth_sampling import inverse_depth_grid
from jar.geometry import pixel_grid, project, safe_masked_softmax, sample_maps, unproject
from .initial_surface_memory import CandidateChannelNorm
from .metric_ruler_network import depth_cell_edges
from .ruler_attention import RulerSourceAttention


def _partition(value, grid):
    """[B,C,D,H,W] → [B,node,element,C]；边缘只补零，不环绕。"""
    batch, channels, depth, height, width = value.shape
    blocks = tuple(math.ceil(size / cells) for size, cells in zip((depth, height, width), grid))
    padded = tuple(cells * block for cells, block in zip(grid, blocks))
    value = F.pad(value, (0, padded[2] - width, 0, padded[1] - height, 0, padded[0] - depth))
    value = value.reshape(batch, channels, grid[0], blocks[0], grid[1], blocks[1], grid[2], blocks[2])
    return value.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(batch, math.prod(grid), math.prod(blocks), channels), blocks


def _broadcast_nodes(nodes, grid, blocks, shape):
    value = nodes.transpose(1, 2).reshape(nodes.shape[0], nodes.shape[-1], *grid)
    for axis, repeat in zip((2, 3, 4), blocks):
        value = value.repeat_interleave(repeat, dim=axis)
    return value[..., :shape[0], :shape[1], :shape[2]]


class MetricSceneFusion(nn.Module):
    """候选内容汇聚为少量视锥节点，再通过公制关系进行全局协商。

    正常尺寸为112节点。全部深度候选保留在残差通路中；仅全局消息经过
    节点瓶颈。默认只读取所属节点；depth_bank允许同一图像区域内的
    不同深度节点竞争，候选内容决定读取对象，另有零值null允许拒绝。
    """

    def __init__(self, channels, raw_channels, settings):
        super().__init__()
        hidden = int(settings.get("scene_channels", 128))
        heads = int(settings.get("scene_heads", 4))
        if hidden % heads:
            raise ValueError("scene_channels须能被scene_heads整除")
        self.grid = tuple(int(number) for number in settings.get("scene_grid", (4, 4, 7)))
        if len(self.grid) != 3 or min(self.grid) < 1:
            raise ValueError("scene_grid须为三个正整数")
        self.heads = heads
        self.scene_return = settings.get("scene_return", "owner")
        if self.scene_return not in {"owner", "depth_bank"}:
            raise ValueError("scene_return须为owner或depth_bank")
        self.scene_null_prior = settings.get("scene_null_prior", "single")
        if self.scene_null_prior not in {"single", "balanced"}:
            raise ValueError("scene_null_prior须为single或balanced")
        self.scene_return_geometry = bool(settings.get("scene_return_geometry", False))
        if self.scene_return_geometry and (self.scene_return != "depth_bank" or self.scene_null_prior != "balanced"):
            raise ValueError("scene_return_geometry需要depth_bank与balanced null")
        self.context = nn.Conv2d(4 * raw_channels, hidden, 1)
        self.pool_query = nn.Linear(hidden, channels)
        self.pool_key = nn.Linear(channels, channels, bias=False)
        self.pool_value = nn.Linear(channels, hidden)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(hidden), "qkv": nn.Linear(hidden, hidden * 3),
                "out": nn.Linear(hidden, hidden), "norm2": nn.LayerNorm(hidden),
                "ffn": nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)),
            }) for _ in range(int(settings.get("scene_blocks", 1)))
        ])
        self.geometry_strength = nn.Parameter(torch.zeros(heads))
        self.return_key = nn.Linear(hidden, channels)
        self.return_value = nn.Linear(hidden, channels)
        self.return_query = nn.Conv3d(channels, channels, 1)
        self.norm = CandidateChannelNorm(channels)

    def _region_log_prior(self, points, center, covariance, available, grid):
        """计算区域的公制相容性，不是表面证据或观测置信度。"""
        batch = points.shape[0]
        depth_nodes, height_nodes, width_nodes = grid
        spatial_nodes = height_nodes * width_nodes
        positions, _ = _partition(points.float(), (1, height_nodes, width_nodes))
        center = center.float().reshape(batch, depth_nodes, spatial_nodes, 3).transpose(1, 2)
        # 每个节点只求一次3×3逆，避免按每个候选反复求解。所有几何保持FP32。
        with torch.autocast(device_type=points.device.type, enabled=False):
            precision = torch.linalg.inv(covariance.float() + 0.01 * torch.eye(3, device=points.device))
            precision = precision.reshape(batch, depth_nodes, spatial_nodes, 3, 3).transpose(1, 2)
            displacement = positions[:, :, :, None] - center[:, :, None]
            transformed = torch.einsum("bgkij,bglkj->bglki", precision, displacement)
            distance = (displacement * transformed).sum(-1).clamp(0, 16)
            prior = safe_masked_softmax(-0.5 * distance, available[:, :, None], dim=-1)
            return prior.clamp_min(1e-12).log()

    def _depth_bank_return(self, features, nodes, known, grid, points=None, center=None, covariance=None):
        """每个候选读取同xy区域的深度节点；不构建全候选两两关系。"""
        batch, channels, depth, height, width = features.shape
        depth_nodes, height_nodes, width_nodes = grid
        spatial_nodes = height_nodes * width_nodes
        queries, blocks = _partition(self.return_query(features), (1, height_nodes, width_nodes))
        keys = self.return_key(nodes).reshape(batch, depth_nodes, spatial_nodes, channels).transpose(1, 2)
        values = self.return_value(nodes).reshape(batch, depth_nodes, spatial_nodes, channels).transpose(1, 2)
        available = known.reshape(batch, depth_nodes, spatial_nodes).transpose(1, 2)
        if self.scene_return_geometry:
            with torch.autocast(device_type=features.device.type, enabled=False):
                scores = queries.float() @ keys.float().transpose(-1, -2) / math.sqrt(channels)
        else:
            scores = queries.float() @ keys.float().transpose(-1, -2) / math.sqrt(channels)
        # null始终输出零消息。balanced按有效节点数修正其先验，
        # 使所有节点等零logit时总接收质量仍为1/2，而非随节点数增加。
        if self.scene_return_geometry:
            scores = scores + self._region_log_prior(points, center, covariance, available, grid)
            scores = F.pad(scores, (0, 1), value=0)
        elif self.scene_null_prior == "balanced":
            null_score = available.sum(-1, keepdim=True).clamp_min(1).float().log()
            null_score = null_score[:, :, None].expand(-1, -1, scores.shape[2], -1)
            scores = torch.cat((scores, null_score), -1)
        else:
            scores = F.pad(scores, (0, 1), value=0)
        available = F.pad(available, (0, 1), value=True)
        weights = safe_masked_softmax(scores, available[:, :, None], dim=-1)
        message = weights[..., :-1].to(values.dtype) @ values
        message = message.reshape(batch, height_nodes, width_nodes, depth, blocks[1], blocks[2], channels)
        message = message.permute(0, 6, 3, 1, 4, 2, 5)
        return message.reshape(batch, channels, depth, height_nodes * blocks[1], width_nodes * blocks[2])[..., :height, :width]

    def forward(self, features, valid, points, raw, use_geometry=True):
        shape = features.shape[-3:]
        grid = tuple(min(size, requested) for size, requested in zip(shape, self.grid))
        cells, blocks = _partition(features, grid)
        masks = _partition(valid[:, None].float(), grid)[0][..., 0] > 0.5
        positions = _partition(points, grid)[0].float()
        # 原图上下文与候选必须使用同一分区；adaptive_avg_pool在非整除
        # 尺寸会产生重叠窗口，和这里的ceil分区不一致。
        raw_context = self.context(torch.cat(raw, 1))
        raw_cells, _ = _partition(raw_context[:, :, None], (1, *grid[1:]))
        raw_valid, _ = _partition(valid.any(1)[:, None, None].float(), (1, *grid[1:]))
        context = (raw_cells * raw_valid).sum(2) / raw_valid.sum(2).clamp_min(1)
        context = context[:, None].expand(-1, grid[0], -1, -1).flatten(1, 2)
        query = self.pool_query(context)
        logits = (self.pool_key(cells) * query[:, :, None]).sum(-1) / math.sqrt(features.shape[1])
        weights = safe_masked_softmax(logits.float(), masks, dim=-1)
        nodes = self.pool_value((cells * weights[..., None]).sum(2)) + context
        known = masks.any(-1)
        count = masks.sum(-1, keepdim=True).clamp_min(1)
        center = (positions * masks[..., None]).sum(2) / count
        delta = positions - center[:, :, None]
        if self.scene_return_geometry:
            with torch.autocast(device_type=features.device.type, enabled=False):
                covariance = torch.einsum("bgli,bglj,bgl->bgij", delta, delta, masks.float()) / count[..., None]
        else:
            covariance = torch.einsum("bgli,bglj,bgl->bgij", delta, delta, masks.float()) / count[..., None]
        # 计算区域的空间足迹，不是声称已测得的表面方差。各向同性小项防退化。
        displacement = center[:, :, None] - center[:, None]
        combined = covariance[:, :, None] + covariance[:, None] + 0.01 * torch.eye(3, device=features.device)
        distance = (displacement * torch.linalg.solve(combined, displacement[..., None])[..., 0]).sum(-1)
        metric_bias = -0.5 * distance.clamp(0, 16)[:, None]
        strength = self.geometry_strength.sigmoid()[None, :, None, None]
        metric_bias = metric_bias * strength * float(use_geometry)
        for layer in self.layers:
            normalized = layer["norm1"](nodes)
            q, k, value = layer["qkv"](normalized).chunk(3, -1)
            q, k, value = [item.unflatten(-1, (self.heads, item.shape[-1] // self.heads)).transpose(1, 2)
                           for item in (q, k, value)]
            scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1]) + metric_bias
            attention = safe_masked_softmax(scores, known[:, None, None], dim=-1)
            message = (attention.to(value.dtype) @ value).transpose(1, 2).flatten(2)
            nodes = nodes + layer["out"](message) * known[..., None]
            nodes = nodes + layer["ffn"](layer["norm2"](nodes)) * known[..., None]
        if self.scene_return == "owner":
            key = _broadcast_nodes(self.return_key(nodes), grid, blocks, shape)
            value = _broadcast_nodes(self.return_value(nodes), grid, blocks, shape)
            gate = ((self.return_query(features) * key).sum(1, keepdim=True) / math.sqrt(features.shape[1])).sigmoid()
            message = gate * value
        else:
            message = self._depth_bank_return(features, nodes, known, grid, points, center, covariance)
        result = self.norm(features + message) * valid[:, None]
        return result, {"node_count": math.prod(grid), "valid_nodes": known, "geometry_distance": distance}


class RulerDirectDecoder(nn.Module):
    """输入约定见 forward；只输出参考图，所有长度/深度为米。"""

    def __init__(self, settings=None):
        super().__init__()
        self.settings = dict(settings or {})
        settings = self.settings
        self.near = float(settings.get("near", 2.0))
        self.far = float(settings.get("far", 80.0))
        self.count = int(settings.get("depth_bins", 96))
        self.peak_count = int(settings.get("peak_count", 4))
        if not 1 <= self.peak_count <= self.count:
            raise ValueError("peak_count必须在1和depth_bins之间")
        rho = inverse_depth_grid(self.near, self.far, self.count, "hybrid",
                                 float(settings.get("transition", 16.0)),
                                 float(settings.get("near_fraction", 0.5)))
        self.register_buffer("depth_lattice", rho.flip(0).reciprocal())
        channels = int(settings.get("channels", 32))
        raw_channels = int(settings.get("raw_channels", 64))
        self.use_geometry = bool(settings.get("use_geometry", True))
        self.use_scene = bool(settings.get("use_scene", True))
        self.absolute_information = bool(settings.get("absolute_information", True))
        self.use_fine_offset = bool(settings.get("use_fine_offset", True))
        self.proposal_transport = settings.get("proposal_transport", "nearest")
        if self.proposal_transport not in {"nearest", "bilinear", "appearance"}:
            raise ValueError("proposal_transport须为nearest/bilinear/appearance")
        self.fine_readout = settings.get("fine_readout", "winner")
        if self.fine_readout not in {"winner", "mode_mean"}:
            raise ValueError("fine_readout须为winner或mode_mean")
        self.source_statistic = settings.get("source_statistic", "weighted")
        if self.source_statistic not in {"weighted", "best_two", "second"}:
            raise ValueError("source_statistic须为weighted/best_two/second")
        self.source = RulerSourceAttention(feature_channels=16, channels=16)
        self.evidence = nn.Sequential(nn.Conv3d(18, channels, 1), nn.GELU(), nn.Conv3d(channels, channels, 1))
        self.raw_adapters = nn.ModuleList([nn.Conv2d(768, raw_channels, 1) for _ in range(4)])
        self.raw_query = nn.Conv3d(channels, raw_channels, 1)
        self.raw_output = nn.Conv3d(raw_channels, channels, 1)
        self.position = nn.Sequential(nn.Linear(3, channels), nn.GELU(), nn.Linear(channels, channels))
        self.entry_norm = CandidateChannelNorm(channels)
        self.scene = MetricSceneFusion(channels, raw_channels, settings)
        self.scene.requires_grad_(self.use_scene)
        self.local = nn.Sequential(
            nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), groups=channels),
            nn.GELU(), nn.Conv3d(channels, channels, 1),
        )
        self.fine_inherit = nn.Conv3d(channels, channels, 1)
        self.score = nn.Conv3d(channels, 1, 1)
        self.offset = nn.Conv3d(channels, 1, 1)
        self.upsample_mask = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
                                          nn.Conv2d(channels, 16 * 9, 1))
        nn.init.normal_(self.score.weight, std=1e-3)
        nn.init.zeros_(self.score.bias)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.local_kind = settings.get("local_kind", "ray")
        if self.local_kind in {"volume", "scale_space"}:
            from .ruler_spatial import CandidateSpatialConv, CandidateScaleSpace

            # 在原模块和所有头初始化之后替换，保持共同参数和初始函数一致。
            block = CandidateSpatialConv if self.local_kind == "volume" else CandidateScaleSpace
            self.local = block(channels).initialize_from_depth(self.local)
        elif self.local_kind in {"plane_flat", "plane_metric"}:
            from .ruler_spatial import CandidatePlaneMessage

            self.local = CandidatePlaneMessage(
                channels, use_metric=self.local_kind == "plane_metric"
            ).initialize_from_depth(self.local)
        elif self.local_kind != "ray":
            raise ValueError("local_kind须为ray/volume/scale_space/plane_flat/plane_metric")

    @staticmethod
    def _aligned(features, shape, stride):
        grid = pixel_grid(*shape, device=features.device, stride=stride).reshape(1, -1, 2)
        sampled = sample_maps(features, (grid - 7.5).expand(len(features), -1, -1), stride=16, padding_mode="border")
        return sampled.transpose(1, 2).reshape(len(features), features.shape[1], *shape)

    def _prepare(self, inputs):
        matching = inputs["matching_features"]
        if matching.ndim == 4:
            matching = matching[None]
        batch, views = matching.shape[:2]
        if matching.ndim != 5 or matching.shape[2] != 16:
            raise ValueError("matching_features必须是[B,V,16,Hm,Wm]或[V,16,Hm,Wm]")
        if len(inputs["raw_maps"]) != 4 or any(value.shape[:2] != (batch, 768) for value in inputs["raw_maps"]):
            raise ValueError("raw_maps需要四层[B,768,H16,W16]")
        height, width = (int(value) for value in inputs["image_size"])
        reference = int(inputs["reference_index"])
        if height % 8 or width % 8 or not 0 <= reference < views:
            raise ValueError("图像尺寸必须整除8，reference_index必须有效")
        intrinsic, transform = inputs["K"].float(), inputs["T_cam_from_rig"].float()
        view_mask = inputs.get("view_mask", torch.ones(batch, views, device=matching.device, dtype=torch.bool)).bool()
        calibration_valid = torch.isfinite(intrinsic).all((-2, -1)) & torch.isfinite(transform).all((-2, -1))
        calibration_valid &= (intrinsic[..., 0, 0] > 0) & (intrinsic[..., 1, 1] > 0)
        view_mask = view_mask & calibration_valid
        intrinsic = torch.where(view_mask[..., None, None], intrinsic, torch.eye(3, device=matching.device))
        transform = torch.where(view_mask[..., None, None], transform, torch.eye(4, device=matching.device))
        image_mask = inputs.get("image_mask", torch.ones(batch, views, height, width, device=matching.device, dtype=torch.bool))
        image_mask = image_mask.bool() & view_mask[..., None, None]
        matching_valid = torch.isfinite(matching).all(2) & view_mask[..., None, None]
        matching = torch.where(matching_valid[:, :, None], matching.float(), 0)
        matching = torch.nan_to_num(matching)
        scale = torch.as_tensor(inputs.get("matching_scale", (1., 1.)), device=matching.device, dtype=torch.float32)
        scale = scale.reshape(-1, 2).expand(batch, -1)
        raw = [adapter(torch.nan_to_num(value.float())) for adapter, value in zip(self.raw_adapters, inputs["raw_maps"])]
        return {"matching": matching, "matching_valid": matching_valid,
                "K": intrinsic, "T": transform, "view_mask": view_mask,
                "image_mask": image_mask, "scale": scale, "reference": reference,
                "image_size": (height, width), "raw": raw}

    def _observe(self, prepared, candidates, stride, half_width):
        batch, count, height, width = candidates.shape
        views = prepared["matching"].shape[1]
        reference = prepared["reference"]
        grid = pixel_grid(height, width, device=candidates.device, stride=stride).reshape(-1, 2)
        results = {name: [] for name in ("reference", "source", "source_valid", "ray_derivative", "ref_valid", "points")}
        for sample in range(batch):
            features = prepared["matching"][sample]
            intrinsic, transform = prepared["K"][sample], prepared["T"][sample]
            points, rays = unproject(grid[None].expand(count, -1, -1), candidates[sample].reshape(count, -1),
                                     intrinsic[reference], transform[reference])
            uv, _, valid, jacobian = project(points.reshape(-1, 3), intrinsic, transform, prepared["image_size"],
                                             prepared["view_mask"][sample], prepared["image_mask"][sample])
            scale = prepared["scale"][sample]
            feature_uv = (uv + 0.5) * scale / 4 - 0.5
            upper = feature_uv.new_tensor((features.shape[-1] - 0.5, features.shape[-2] - 0.5))
            valid = valid & ((feature_uv >= -0.5) & (feature_uv <= upper)).all(-1)
            finite_samples = sample_maps(prepared["matching_valid"][sample, :, None].float(),
                                          feature_uv, padding_mode="border")[..., 0] > 0.999
            valid &= finite_samples
            source = sample_maps(features, feature_uv).transpose(1, 2).reshape(views, 16, count, height, width)
            anchor_uv = (grid[None] + 0.5) * scale / 4 - 0.5
            anchor = sample_maps(features[reference:reference + 1], anchor_uv).transpose(1, 2).reshape(16, height, width)
            anchor_finite = sample_maps(prepared["matching_valid"][sample, reference:reference + 1, None].float(),
                                       anchor_uv, padding_mode="border")[0, :, 0] > 0.999
            ref_valid = sample_maps(prepared["image_mask"][sample, reference:reference + 1, None].float(),
                                    grid[None])[0, :, 0] > 0.999
            ref_valid &= prepared["view_mask"][sample, reference]
            derivative = (jacobian * rays.reshape(1, -1, 1, 3)).sum(-1) * scale / 4
            # 共心纯旋转不提供米制匹配支持，但单图深度假设仍可以输出。
            valid &= derivative.square().sum(-1) > 1e-10
            valid[reference] = False
            valid = valid.reshape(views, count, height, width)
            valid &= (ref_valid & anchor_finite).reshape(1, 1, height, width)
            source = F.normalize(torch.where(valid[:, None], source, 0), dim=1, eps=1e-6)
            results["reference"].append(F.normalize(anchor, dim=0, eps=1e-6))
            results["source"].append(source)
            results["source_valid"].append(valid)
            results["ray_derivative"].append(derivative.transpose(1, 2).reshape(views, 2, count, height, width))
            results["ref_valid"].append(ref_valid.reshape(1, height, width))
            results["points"].append(points.reshape(count, height, width, 3).permute(3, 0, 1, 2))
        result = {name: torch.stack(values) for name, values in results.items()}
        result["cell_half_width"] = half_width
        return result

    def _tokens(self, prepared, observation, candidates, stride):
        source = self.source(**{name: observation[name] for name in
                                ("reference", "source", "source_valid", "ray_derivative", "cell_half_width")},
                             use_geometry=self.use_geometry)
        metric = source["information"].clamp_min(0).log1p() * float(self.use_geometry and self.absolute_information)
        features = self.evidence(torch.cat((source["pooled"], source["null_mass"], metric), 1))
        normalized = (candidates.log() - math.log(self.near)) / math.log(self.far / self.near)
        coordinates = torch.stack((normalized, (math.pi * normalized).sin(), (math.pi * normalized).cos()), -1)
        features = features + self.position(coordinates).movedim(-1, 1)
        raw = [self._aligned(value, candidates.shape[-2:], stride) for value in prepared["raw"]]
        bank = torch.stack(raw, 2)
        query = self.raw_query(features)
        layer_logits = torch.einsum("bcdhw,bclhw->bldhw", query, bank) / math.sqrt(bank.shape[1])
        visual = torch.einsum("bldhw,bclhw->bcdhw", layer_logits.softmax(1), bank)
        valid = observation["ref_valid"].expand_as(candidates)
        features = self.entry_norm(features + self.raw_output(visual)) * valid[:, None]
        cosine = (observation["source"] * observation["reference"][:, None, :, None]).sum(2)
        support = observation["source_valid"].sum(1)
        # 拒绝的坏源不能经固定均值旁路再次进入评分；null质量保留，
        # 不能把极小的剩余观测权重重新归一化为100%证据。
        similarity = (cosine * source["weights"]).sum(1)
        if self.source_statistic != "weighted":
            ranked = cosine.masked_fill(~observation["source_valid"], -torch.inf).sort(1, descending=True).values
            count = observation["source_valid"].sum(1)
            first = torch.where(count > 0, ranked[:, 0], 0)
            second = torch.where(count > 1, ranked[:, 1], first) if ranked.shape[1] > 1 else first
            agreement = (first + second) * 0.5 if self.source_statistic == "best_two" else second
            similarity = agreement * source["weights"].sum(1)
        return features, raw, valid, similarity, support, source

    def _prediction_heads(self, observation):
        return self.score, self.offset

    def _prediction(self, features, candidates, valid, similarity, support, lower, upper, observation, use_offset=True):
        score_head, offset_head = self._prediction_heads(observation)
        scores = score_head(features)[:, 0] + 4 * similarity
        scores = scores.masked_fill(~valid, -1e4)
        probability = safe_masked_softmax(scores.float(), valid, dim=1)
        fraction = offset_head(features)[:, 0].tanh() if use_offset else torch.zeros_like(candidates)
        length = torch.where(fraction >= 0, upper - candidates, candidates - lower)
        continuous = (candidates + length * fraction).clamp(self.near, self.far)
        winner = scores.argmax(1, keepdim=True)
        depth = continuous.gather(1, winner)
        depth = torch.where(valid.any(1, keepdim=True), depth, 0)
        return {"scores": scores, "probability": probability, "candidate_depth": continuous,
                "sampling_depth": candidates, "valid": valid, "depth": depth,
                "support": support, "ref_valid": valid.any(1, keepdim=True),
                "selected_index": winner, "confidence": probability.gather(1, winner),
                "source_valid": observation["source_valid"]}

    def _fine_proposals(self, coarse, features, observation):
        scores, candidates = coarse["scores"], coarse["candidate_depth"]
        available = coarse["valid"].clone()
        chosen = []
        separation = float(self.settings.get("peak_separation_m", 1.0))
        exclusion = int(self.settings.get("peak_exclusion_bins", 0))
        bins = torch.arange(scores.shape[1], device=scores.device)[None, :, None, None]
        for _ in range(self.peak_count):
            active = available.any(1, keepdim=True)
            index = scores.masked_fill(~available, -1e4).argmax(1, keepdim=True)
            index = torch.where(active, index, coarse["selected_index"])
            chosen.append(index)
            center = candidates.gather(1, index)
            available &= (candidates - center).abs() >= separation
            if exclusion > 0:
                available &= (bins - index).abs() > exclusion
            available.scatter_(1, index, False)
        indices = torch.cat(chosen, 1)
        peaks = candidates.gather(1, indices)
        edges = depth_cell_edges(self.depth_lattice)
        widths = (edges[1:] - edges[:-1])[None, :, None, None].expand_as(candidates)
        widths = widths.gather(1, indices) * 0.5
        g = observation["ray_derivative"].square().sum(2).sqrt()
        g = torch.where(observation["source_valid"], g, 0).amax(1).gather(1, indices)
        pixel_step = float(self.settings.get("fine_step_pixels", 0.5)) / g.clamp_min(1e-6)
        steps = torch.maximum(widths, pixel_step).clamp(float(self.settings.get("fine_min_step_m", 0.05)),
                                                       float(self.settings.get("fine_max_step_m", 8.0)))
        peaks = F.interpolate(peaks, scale_factor=2, mode="nearest")
        steps = F.interpolate(steps, scale_factor=2, mode="nearest")
        offsets = peaks.new_tensor((-1., 0., 1.))[None, None, :, None, None]
        candidates = (peaks[:, :, None] + steps[:, :, None] * offsets).flatten(1, 2).clamp(self.near, self.far)
        half_width = steps[:, :, None].expand(-1, -1, 3, -1, -1).flatten(1, 2) * 0.5
        channel_indices = indices[:, None].expand(-1, features.shape[1], -1, -1, -1)
        inherited = features.gather(2, channel_indices)
        inherited = inherited.repeat_interleave(2, -2).repeat_interleave(2, -1).repeat_interleave(3, 2)
        return candidates, half_width, inherited

    @staticmethod
    def _mode_depth(fine):
        """先选表面峰，再在该峰的三个局部采样内连续定位。

        不把不同粗表面的深度直接平均。保留离散选峰，但最终深度损失
        可以训练选中峰内部的相对分数；不能恢复已经删除的其他表面。
        """
        group_start = fine["selected_index"] // 3 * 3
        offsets = torch.arange(3, device=group_start.device)[None, :, None, None]
        indices = group_start + offsets
        valid = fine["valid"].gather(1, indices)
        scores = fine["scores"].gather(1, indices)
        weights = safe_masked_softmax(scores.float(), valid, dim=1)
        depths = fine["candidate_depth"].gather(1, indices)
        return (weights * depths).sum(1, keepdim=True)

    def _upsample(self, fine, features, image_mask, selected_features=None):
        if selected_features is None:
            winner = fine["selected_index"][:, None].expand(-1, features.shape[1], -1, -1, -1)
            selected_features = features.gather(2, winner)[:, :, 0]
        batch, _, height, width = selected_features.shape
        mask = self.upsample_mask(selected_features).reshape(batch, 1, 9, 4, 4, height, width)
        neighbor_valid = F.unfold(fine["ref_valid"].float(), 3, padding=1).reshape(batch, 1, 9, 1, 1, height, width) > 0.5
        weights = safe_masked_softmax(mask.float(), neighbor_valid, dim=2)

        def mix(value):
            neighbors = F.unfold(value.float(), 3, padding=1).reshape(batch, 1, 9, 1, 1, height, width)
            return (weights * neighbors).sum(2).permute(0, 1, 4, 2, 5, 3).reshape(batch, 1, height * 4, width * 4)

        depth = mix(fine["depth"])
        confidence = mix(fine["confidence"])
        # 掩码可能比stride4更细：整块没有有效采样点时保持无证据先验，
        # 不能把这些实际有RGB的像素静默移出评测，也不能输出非法0米。
        fallback = math.sqrt(self.near * self.far)
        depth = torch.where(depth > 0, depth, fallback)
        # 支持数只取最近细格点的真实可采样源数，不把凸组合权重当新证据。
        support = fine["support"].gather(1, fine["selected_index"])
        support = support.repeat_interleave(4, -2).repeat_interleave(4, -1)
        valid = image_mask.bool()
        return {"depth": torch.where(valid, depth, 0), "valid": valid,
                "confidence": torch.where(valid, confidence, 0),
                "support": torch.where(valid, support, 0), "unknown": torch.ones_like(valid),
                "observed": torch.zeros_like(valid), "state": torch.full_like(support, 2)}

    def _render(self, fine, features, image_mask, prepared):
        return self._upsample(fine, features, image_mask)

    def forward(self, inputs):
        """matching_features、raw_maps、K/T、掩码、matching_scale与image_size；不读取GT。"""
        prepared = self._prepare(inputs)
        batch = prepared["matching"].shape[0]
        height, width = prepared["image_size"]
        candidates = self.depth_lattice[None, :, None, None].expand(batch, -1, height // 8, width // 8)
        edges = depth_cell_edges(self.depth_lattice)
        half_width = (edges[1:] - edges[:-1]) * 0.5
        observation = self._observe(prepared, candidates, 8, half_width)
        features, raw, valid, similarity, support, source = self._tokens(prepared, observation, candidates, 8)
        scene = None
        if self.use_scene:
            features, scene = self.scene(features, valid, observation["points"], raw, self.use_geometry)
        if self.local_kind == "ray":
            local = self.local(features)
        elif self.local_kind in {"volume", "scale_space"}:
            local = self.local(features, valid)
        else:
            reference = prepared["reference"]
            local = self.local(features, valid, self.depth_lattice, observation["points"],
                               prepared["K"][:, reference], prepared["T"][:, reference], 8)
        features = features + local * valid[:, None]
        coarse = self._prediction(features, candidates, valid, similarity, support,
                                  edges[:-1][None, :, None, None], edges[1:][None, :, None, None], observation)
        if self.proposal_transport == "nearest":
            fine_candidates, fine_width, inherited = self._fine_proposals(coarse, features, observation)
        else:
            from .ruler_proposals import transport_proposals

            fine_candidates, fine_width, inherited = transport_proposals(
                self, coarse, features, observation, prepared
            )
        fine_observation = self._observe(prepared, fine_candidates, 4, fine_width)
        fine_features, _, fine_valid, fine_similarity, fine_support, _ = self._tokens(prepared, fine_observation, fine_candidates, 4)
        fine_features = self.entry_norm(fine_features + self.fine_inherit(inherited)) * fine_valid[:, None]
        fine = self._prediction(fine_features, fine_candidates, fine_valid, fine_similarity, fine_support,
                                fine_candidates - fine_width, fine_candidates + fine_width, fine_observation,
                                use_offset=self.use_fine_offset)
        if self.fine_readout == "mode_mean":
            fine["depth"] = self._mode_depth(fine)
        image_mask = prepared["image_mask"][:, prepared["reference"]:prepared["reference"] + 1]
        output = self._render(fine, fine_features, image_mask, prepared)
        output.update(coarse=coarse, fine=fine, scene=scene,
                      reference_index=prepared["reference"], continuous_offset_used=True,
                      support_semantics="几何可采样的源视图数；非可见性认证")
        return output
