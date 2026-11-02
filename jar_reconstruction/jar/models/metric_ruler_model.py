"""公制候选网络的完整重建运行时，不改原自研模型的默认入口。

保留原图像编码、学习式初值、K9、原始特征读取、标定细化和全图输出头。
候选场可在 stride4/8/16 取证与推理，再按需插值整条评分曲线。
它不是原模型的等价加速：细物体、遮挡边界和粗格点外的证据可能损失。
候选网络的连续偏移头暂未接入原 K9 的固定逆深度格点，不宣称已使用该能力。
"""

from collections import OrderedDict
import copy

import torch
from torch import nn

from jar.model import ReconstructionModel
from .dense_decoder import aligned_resample
from .initial_surface_memory import plane_sampling_plan
from .metric_ruler_evidence import extract_ruler_features
from .metric_ruler_geometry_evidence import RulerProjectionCache, extract_geometry_ruler_features
from .metric_ruler_network import MetricRulerCandidateNetwork, metric_depth_plan
from .ruler_attention_network import build_ruler_network


class _ConditionalProposalObserver(nn.Module):
    """模式开关不改设计模块公式；原模型旁路直接返回原 proposal。"""

    def __init__(self, observer, enabled=True):
        super().__init__()
        self.observer = observer
        self.enabled = enabled

    def forward(self, proposal, *arguments):
        return self.observer(proposal, *arguments) if self.enabled else proposal


def upsample_candidate_curves(values, valid, size, source_stride=16, target_stride=4):
    """逐候选插值；不混合深度峰，不跨无效邻点，不外推最后粗格点之外。"""
    valid = valid.expand_as(values)
    if source_stride == target_stride and tuple(values.shape[-2:]) == tuple(size):
        return torch.where(valid, values, 0), valid
    height, width = values.shape[-2:]
    clean = torch.where(valid, values, 0).reshape(1, -1, height, width)
    coverage = valid.float().reshape_as(clean)
    result = aligned_resample(clean, source_stride, target_stride, size)
    known = aligned_resample(coverage, source_stride, target_stride, size) >= 1 - 1e-5
    y = torch.arange(size[0], device=values.device) * target_stride
    x = torch.arange(size[1], device=values.device) * target_stride
    inside = (y[:, None] <= (height - 1) * source_stride)
    inside = inside & (x[None, :] <= (width - 1) * source_stride)
    known = known & inside
    shape = (*values.shape[:-2], *size)
    return torch.where(known, result, 0).reshape(shape), known.reshape(shape)


def add_metric_score_residual(field, correction, coarse_valid, alpha, coarse_stride=16):
    """保留真实细格证据，仅把粗核相对其起点的评分改变量作为上下文。

    评分修正不新增观测、可见性或候选。无可靠粗上下文时仍使用原细场。
    """
    if alpha == 0:
        return field
    value, known = upsample_candidate_curves(
        correction, coarse_valid, field["valid"].shape[-2:],
        coarse_stride, field["stride"],
    )
    update = torch.where(known & field["valid"], value * alpha, 0)
    result = dict(field)
    result["residual"] = field["residual"] + update
    if "recovery" in field:
        recovery = dict(field["recovery"])
        recovery["scores"] = field["recovery"]["scores"] + update
        result["recovery"] = recovery
    return result


class _MetricRulerField(nn.Module):
    """托管原候选场并只替换它的取证/评分；不缓存图像相关数据。"""

    def __init__(self, baseline, settings, network, extractor):
        super().__init__()
        self.baseline = baseline
        self.settings = settings
        self.network = network
        evidence_kind = settings.get("evidence_kind", "jet")
        if evidence_kind not in ("jet", "geometry"):
            raise ValueError("evidence_kind 只能为 jet 或 geometry")
        if extractor is None:
            extractor = extract_geometry_ruler_features if evidence_kind == "geometry" else extract_ruler_features
        self.extractor = extractor
        self.projection_cache = RulerProjectionCache(capacity=4)
        self.mode = settings.get("mode", "ruler")
        self.coarse_memory_only = bool(settings.get("coarse_memory_only", False))
        self.fine_evidence_mode = settings.get("fine_evidence_mode", "replace")
        if self.fine_evidence_mode not in ("replace", "residual"):
            raise ValueError("fine_evidence_mode 只允许 replace 或 residual")
        self.residual_weight = float(settings.get("residual_weight", 1.0))
        if not 0 <= self.residual_weight <= 1:
            raise ValueError("residual_weight 须位于 [0,1]")
        self.stride = int(settings.get("evidence_stride", 16))
        if self.stride not in (4, 8, 16):
            raise ValueError("候选取证步长只允许4/8/16，图像尺寸保持不变")
        if baseline.neighbor_pixels % self.stride:
            raise ValueError("原候选场的邻域像素跨度必须整除取证步长")
        self.latest = []
        self.latest_availability = []
        self._plan_cache = OrderedDict()

    def _apply(self, function, recurse=True):
        # 普通 Python 缓存不受 Module.to 管理，迁移时主动释放旧设备张量。
        self._plan_cache.clear()
        self.projection_cache.clear()
        self.latest = []
        self.latest_availability = []
        return super()._apply(function, recurse=recurse)

    def _plans(self, entry):
        # 仅对少量标定值和候选格点作键；相机/候选范围变动后不会误用旧计划。
        def signature(value):
            return tuple(value.detach().float().cpu().reshape(-1).tolist())

        metric = getattr(self.network, "mode", "metric") == "metric"
        key = (
            str(entry["cost"].device), tuple(entry["cost"].shape[-2:]), self.stride, metric,
            signature(entry["inverse_depths"]), signature(entry["intrinsic"]),
            signature(entry["transform"]),
        )
        if key not in self._plan_cache:
            # eval 生成的常量计划之后也允许训练反传；不保留 inference tensor。
            with torch.inference_mode(False), torch.no_grad():
                planes = plane_sampling_plan(
                    entry["inverse_depths"], entry["intrinsic"], entry["transform"],
                    entry["cost"].shape[-2:], self.stride,
                    neighbor_step=self.baseline.neighbor_pixels // self.stride,
                )
                depths = metric_depth_plan(entry["inverse_depths"], metric=metric)
            self._plan_cache[key] = (depths, planes)
            if len(self._plan_cache) > 16:
                self._plan_cache.popitem(last=False)
        return self._plan_cache[key]

    def _network_inputs(self, work, entry, sample, reference):
        height, width = work["rgb"].shape[-2:]
        raw = work.get("_foundation_maps")
        if raw is None and isinstance(self.network, MetricRulerCandidateNetwork):
            raise ValueError("公制候选网络需要原前端同次提供四层 _foundation_maps")
        matching = work["_pretrained_matching_features"][sample]
        if self.settings.get("match_cache_quantization", True):
            matching = matching.half().float()
        scale = work.get("_pretrained_matching_scale")
        scale = matching.new_ones(1, 2) if scale is None else scale[sample:sample + 1]
        image_mask = work.get("image_mask")
        if image_mask is None:
            image_mask = torch.ones(
                *work["rgb"].shape[:2], height, width,
                dtype=torch.bool, device=matching.device,
            )
        view_mask = work.get("view_mask")
        if view_mask is None:
            view_mask = torch.ones(*work["rgb"].shape[:2], dtype=torch.bool, device=matching.device)
        return {
            **{name: entry[name] for name in (
                "cost", "hero", "source_cosine", "source_valid", "candidate_valid", "inverse_depths"
            )},
            "K": work["K"][sample:sample + 1],
            "T_cam_from_rig": work["T_cam_from_rig"][sample:sample + 1],
            "view_mask": view_mask[sample:sample + 1],
            "image_mask": image_mask[sample:sample + 1],
            "image_size": [height, width],
            "matching_features": matching.float(),
            "matching_scale": scale,
            "reference_index": reference,
            "stride": self.stride,
            "raw_maps": [] if raw is None else [value[sample, reference][None] for value in raw],
        }

    def _field(self, entry, scores, size):
        valid = entry["candidate_valid"]
        coarse_scores = scores
        scores, fine_valid = upsample_candidate_curves(scores, valid, size, self.stride)
        hero, _ = upsample_candidate_curves(entry["hero"], valid[:, None], size, self.stride)
        cosine, source_valid = upsample_candidate_curves(
            entry["source_cosine"], entry["source_valid"], size, self.stride
        )
        # 保守有效性不因曲线插值自动变成可见性，也不增加原视图支持。
        source_valid = source_valid & fine_valid[:, None]
        cosine = torch.where(source_valid, cosine, 0)
        support = source_valid.sum(1)
        scores = scores.masked_fill(~fine_valid, -1e4)
        field = {
            "residual": torch.where(fine_valid, scores - 4 * hero[:, 0], 0),
            "valid": fine_valid,
            "inverse_depths": entry["inverse_depths"],
            "stride": 4,
            "weight": self.baseline.weight,
        }
        if self.baseline.capture_recovery:
            field["recovery"] = {
                "scores": scores, "hero": hero,
                "source_cosine": cosine, "source_valid": source_valid,
                "source_mean": cosine.sum(1) / support.clamp_min(1),
                "source_support": support,
            }
            if self.settings.get("proposal_prior", False):
                from .metric_ruler_fine import coarse_proposal_prior

                field["recovery"]["proposal_prior"] = coarse_proposal_prior(
                    scores=coarse_scores,
                    valid=valid, size=size, source_stride=self.stride,
                )
        return field

    def forward(self, work, pyramid, sweep):
        if self.mode == "original":
            return self.baseline(work, pyramid, sweep)
        if self.mode != "ruler":
            raise ValueError("完整模型 mode 只能为 original 或 ruler")
        fine_fields = None
        if self.fine_evidence_mode == "residual":
            fine_fields = self.baseline(work, pyramid, sweep)
            if self.residual_weight == 0:
                self.latest = [{} for _ in fine_fields]
                self.latest_availability = [
                    {reference: self._availability(field) for reference, field in sample.items()}
                    for sample in fine_fields
                ]
                return fine_fields
        evidence_scope = work.get("_first_sweep_evidence")
        if evidence_scope is not None:
            entries = evidence_scope["entries"]
            if entries is None:
                raise RuntimeError("已启用首遍复用，但原粗匹配没有提供同格点证据")
        else:
            source = pyramid[self.baseline.cnn_stride]
            batch, views = source.shape[:2]
            size = pyramid[self.stride].shape[-2:]
            resampled = aligned_resample(source.flatten(0, 1), self.baseline.cnn_stride, self.stride, size)
            sampling = dict(pyramid)
            sampling[self.stride] = resampled.reshape(batch, views, *resampled.shape[1:])
            entries = sweep(work, sampling, stride=self.stride, return_evidence=True)
        fields, self.latest, self.latest_availability = [], [], []
        for sample, references in enumerate(entries):
            sample_fields, sample_outputs, availability = {}, {}, {}
            for reference, entry in references.items():
                depth_plans, plane_plans = self._plans(entry)
                scores = self.baseline.memory(
                    entry["cost"], entry["hero"], entry["candidate_valid"],
                    entry["inverse_depths"], plane_plans,
                    entry["source_cosine"], entry["source_valid"],
                )
                base_scores = scores
                if not self.coarse_memory_only:
                    network_inputs = self._network_inputs(work, entry, sample, reference)
                    options = {"ruler_scale": float(self.settings.get("ruler_scale_m", 1.0))}
                    if self.extractor is extract_geometry_ruler_features:
                        options["cache"] = self.projection_cache
                    features, stats = self.extractor(network_inputs, **options)
                    output = self.network(
                        network_inputs, {"features": features, "stats": stats},
                        depth_plans, plane_plans, base_scores=scores,
                    )
                    scores = output["scores"]
                    sample_outputs[reference] = output
                if fine_fields is None:
                    sample_fields[reference] = self._field(entry, scores, pyramid[4].shape[-2:])
                else:
                    sample_fields[reference] = add_metric_score_residual(
                        fine_fields[sample][reference], scores - base_scores,
                        entry["candidate_valid"], self.residual_weight, self.stride,
                    )
                field = sample_fields[reference]
                availability[reference] = self._availability(field)
            fields.append(sample_fields)
            self.latest.append(sample_outputs)
            self.latest_availability.append(availability)
        return fields

    @staticmethod
    def _availability(field):
        valid = field["valid"]
        support = field.get("recovery", {}).get("source_support")
        return {
            "candidate": valid.any(1),
            "two_sources": (valid & (support >= 2)).any(1) if support is not None else valid.any(1),
        }


class MetricRulerReconstructionModel(nn.Module):
    """传入已严格恢复的原自研模型，独立封装候选改造后的完整推理/训练。

    settings.model 对应 MetricRulerCandidateNetwork 的训练配置；mode='original'
    精确旁路，mode='ruler' 执行新候选场。coarse_memory_only 只供结构控制实验。
    默认冻结原权重，与候选缓存训练一致；关闭 freeze_original 可联合训练。
    """

    def __init__(self, original, settings, network=None, extractor=None):
        super().__init__()
        if not isinstance(original, ReconstructionModel) or not hasattr(original, "candidate_field"):
            raise TypeError("需要带候选场的原 ReconstructionModel，不接受完整 Hero 代替")
        report = getattr(original, "_checkpoint_load_report", {})
        if report.get("missing_keys") or report.get("unexpected_keys"):
            raise ValueError("原自研必须严格恢复全部权重后再集成")
        self.settings = copy.deepcopy(settings)
        if settings.get("proposal_prior", False):
            if not settings.get("exact_observer", False):
                raise ValueError("邻域候选先验必须由 exact_observer 在真实细格重新验证")
            if settings.get("fine_evidence_mode", "replace") != "replace":
                raise ValueError("邻域提议与原细场残差是两个独立实验，不混合开关")
        self.original = original
        self._base_proposal_mode = original.dense_decoder.surface_proposal_mode
        if settings.get("freeze_original", True):
            original.requires_grad_(False)
        network = build_ruler_network(settings.get("model", {})) if network is None else network
        original.candidate_field = _MetricRulerField(
            original.candidate_field, self.settings, network, extractor
        )
        if settings.get("reuse_first_sweep", False):
            field = original.candidate_field
            if field.stride != 16 or field.baseline.cnn_stride != 16:
                raise ValueError("首遍证据复用要求 evidence_stride 和 cnn_stride 均为16")
            if getattr(original, "multifocal_active", False):
                raise ValueError("首遍证据复用目前仅适用于普通广角前端")
        if settings.get("exact_observer", False):
            from .metric_ruler_fine import ExactRulerProposalObserver

            observer = ExactRulerProposalObserver({
                "texture_threshold": original.cfg["model"].get("texture_threshold", 0.002),
                "min_parallax": 1e-5,
                "stride": 4,
            })
            original.dense_decoder.proposal_observer = _ConditionalProposalObserver(
                observer, enabled=self.mode == "ruler"
            )

    @property
    def cfg(self):
        return self.original.cfg

    @property
    def candidate_network(self):
        return self.original.candidate_field.network

    @property
    def mode(self):
        return self.original.candidate_field.mode

    @mode.setter
    def mode(self, value):
        if value not in ("original", "ruler"):
            raise ValueError("mode 只能为 original 或 ruler")
        self.original.candidate_field.mode = value

    def load_candidate_checkpoint(self, path):
        record = torch.load(path, map_location="cpu", weights_only=False)
        if record["plan"]["model"] != self.settings.get("model", {}):
            raise ValueError("候选网络配置必须与训练 checkpoint 完全一致")
        self.candidate_network.load_state_dict(record["model"], strict=True)
        return record.get("step")

    def checkpoint_record(self):
        """完整自包含权重；恢复时不需要旧外部预训练文件或训练缓存。"""
        settings = copy.deepcopy(self.settings)
        settings["mode"] = self.mode
        return {
            "architecture": "metric_ruler_reconstruction",
            "config": copy.deepcopy(self.cfg),
            "settings": settings,
            "model": self.state_dict(),
        }

    @classmethod
    def from_checkpoint(cls, path, network=None, extractor=None):
        record = torch.load(path, map_location="cpu", weights_only=False)
        if record.get("architecture") != "metric_ruler_reconstruction":
            raise ValueError("需要完整公制候选重建 checkpoint，不能混用候选核权重")
        original_state = {}
        for name, value in record["model"].items():
            if not name.startswith("original."):
                continue
            name = name[len("original."):]
            if name.startswith("candidate_field.baseline."):
                name = "candidate_field." + name[len("candidate_field.baseline."):]
            elif name.startswith("candidate_field."):
                continue
            elif name.startswith("dense_decoder.proposal_observer."):
                continue
            original_state[name] = value
        original = ReconstructionModel(record["config"], checkpoint_state=original_state)
        original.load_state_dict(original_state, strict=True)
        model = cls(original, record["settings"], network=network, extractor=extractor)
        model.load_state_dict(record["model"], strict=True)
        return model

    def forward(self, inputs):
        field = self.original.candidate_field
        field.latest = []
        field.latest_availability = []
        observer = getattr(self.original.dense_decoder, "proposal_observer", None)
        if isinstance(observer, _ConditionalProposalObserver):
            observer.enabled = self.mode == "ruler"
        self.original.dense_decoder.surface_proposal_mode = (
            "coarse_prior" if self.mode == "ruler" and self.settings.get("proposal_prior", False)
            else self._base_proposal_mode
        )
        reuse = self.mode == "ruler" and self.settings.get("reuse_first_sweep", False)
        scope = {"stride": 16, "entries": None} if reuse else None
        work = {**inputs, "_first_sweep_evidence": scope} if reuse else inputs
        try:
            result = self.original(work)
        finally:
            if scope is not None:
                # 即使后端失败也释放临时引用；不会跨帧保留图像证据或计算图。
                scope.clear()
        if self.mode == "ruler":
            result["metric_ruler"] = {
                "candidates": field.latest,
                "field_availability": field.latest_availability,
                "evidence_stride": field.stride,
                "field_stride": 4,
                "coarse_memory_only": field.coarse_memory_only,
                "continuous_offset_used_by_backend": False,
                "candidate_is_verified_surface": False,
            }
        return result
