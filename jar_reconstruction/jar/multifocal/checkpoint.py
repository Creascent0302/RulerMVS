"""严格核心权重迁移；新增分支缺失与真正错误明确区分。"""

from copy import deepcopy
from .options import CORE_SWITCHES

# 旧权重配置中的字段；当前模型没有这些独立旧解码器。
LEGACY_FIELDS = (
    "tree",
    "hierarchy",
    "refiner",
    "refine_steps",
    "local_candidates",
    "refine_stride",
    "stop_linearization_gradient",
    "keep_two_peaks",
    "neighbor_cell_size",
)

CONTEXT_BRANCHES = {
    "candidate_field": "candidate_field.",
    "image_context": "image_context.",
    "hypothesis_context": "decoder.hypothesis_context.",
    "deep_encoder": "encoder.extra_stages.",
    "foundation": (
        "encoder.foundation.",
        "encoder.foundation_adapters.",
        "encoder.foundation_gates.",
    ),
    "candidate_reasoning": tuple(f"decoder.blocks.{stage}.reasoning." for stage in range(4)),
    "pair_appearance": tuple(f"decoder.blocks.{stage}.pair_appearance." for stage in range(4)),
    "cost_context": "coarse.context.",
    "dense_decoder": "dense_decoder.",
    "feature_enrichment": "encoder.feature_enrichment.",
    "matching_evidence": "coarse.matching_evidence.",
    "feature_exchange": "feature_exchange.",
    "candidate_transport": "coarse.candidate_transport.",
}

# 允许向已存在的父模块增加独立子模块；不允许借此跳过父模块旧权重。
NESTED_BRANCHES = {
    "matching_evidence.per_view": "coarse.matching_evidence.per_view.",
    "dense_decoder.calibrated_refinement": "dense_decoder.refinement.",
    "dense_decoder.posterior_readout": "dense_decoder.posterior_readout.",
    "dense_decoder.surface_recovery": "dense_decoder.surface_recovery.",
    "dense_decoder.raw_feature_reader": "dense_decoder.raw_feature_reader.",
}


def _nested_settings(model, branch):
    parent, child = branch.split(".")
    settings = deepcopy(model.get(parent, {}).get(child, {}))
    default = child == "calibrated_refinement"
    settings["enabled"] = model.get(parent, {}).get("enabled", False) and settings.get(
        "enabled", default
    )
    return settings


def _uses_pretrained_matching(model):
    refinement = _nested_settings(model, "dense_decoder.calibrated_refinement")
    return (
        (refinement["enabled"] and refinement.get("feature_source", "local") == "pretrained")
        or model.get("pretrained_matching", {}).get("weight", 0) > 0
        or any(
            model.get(branch, {}).get("enabled", False)
            for branch in ("matching_evidence", "feature_exchange")
        )
    )


def _branch_settings(model, branch):
    """嵌套的冻结编码器仍作为独立权重分支，主开关优先。"""
    if branch == "foundation":
        parent = model.get("deep_encoder", {})
        settings = deepcopy(parent.get("foundation", {}))
        settings["enabled"] = parent.get("enabled", False) and settings.get("enabled", False)
        return settings
    settings = deepcopy(model.get(branch, {}))
    if branch == "deep_encoder":
        settings.pop("foundation", None)
    return settings


def _branch_structure(settings, branch):
    settings = deepcopy(settings)
    if branch == "candidate_field":
        settings.pop("weights", None)
    for nested in NESTED_BRANCHES:
        parent, child = nested.split(".")
        if parent == branch:
            settings.pop(child, None)
    if branch == "foundation":
        # 已保存全部参数，初始化路径不影响结构；融合位置允许显式推理消融。
        for key in ("weights", "prefix", "fuse_matching"):
            settings.pop(key, None)
    if branch == "matching_evidence":
        # 只切换现有描述子的原始分数来源，没有新增/替换学习参数。
        settings.pop("primary_descriptor", None)
    return settings


def _nested_structure(settings, branch):
    settings = deepcopy(settings)
    if branch in ("dense_decoder.surface_recovery", "dense_decoder.raw_feature_reader"):
        settings.pop("weights", None)
    if branch == "dense_decoder.raw_feature_reader":
        # 相同权重允许显式旁路残差；分块只改变执行，不改变读取层结构。
        settings.pop("residual_scale", None)
        settings.pop("chunk_size", None)
    if branch == "dense_decoder.calibrated_refinement":
        settings.pop("chunk_size", None)  # 仅改变独立query的运行分块，不改变层或候选。
    if (
        branch == "dense_decoder.calibrated_refinement"
        and settings.get("feature_source") == "pretrained"
    ):
        # 这些具名开关改变既有搜索或门控，显式记录而不更换学习参数。
        settings.pop("gate_multiplier", None)
        settings.pop("max_step", None)
        settings.pop("pass_max_steps", None)
        settings.pop("gate_source", None)
        settings.pop("observation_aggregation", None)
        settings.pop("camera_consensus", None)
    return settings


def _core(config):
    model = deepcopy(config["model"])
    model.pop("sweep_chunk", None)  # 完整代价体不变，仅分批投影候选。
    model.pop("execution", None)  # 只选择现有参考输出的执行调度，参数形状不变。
    model.pop("multifocal", None)
    model.pop("sparse_decoder", None)  # 只控制独立头是否运行，保留已有参数以供消融。
    model.pop("dense_evidence", None)  # 无学习参数的预测后验证。
    model.pop("matching_distillation", None)  # 无参数的训练时匹配教师监督。
    model.pop("cost_calibration", None)  # 无参数的显式候选校准。
    model.pop("pretrained_matching", None)  # 可选外部匹配证据不改变原分层解码器参数。
    model.pop("et_photo_score_scale", None)  # 无新增参数的匹配先验权重消融。
    model.pop("et_curve_context", None)  # 无新增权重的同层曲线传递消融。
    for branch in CONTEXT_BRANCHES:
        model.pop(branch, None)
    for key in LEGACY_FIELDS:
        model.pop(key, None)
    # 这些布尔开关不改变权重形状，允许使用同一权重进行推理消融。
    for key in CORE_SWITCHES:
        model.pop("et_" + key, None)
    model.pop("roi", None)  # ROI没有学习参数；迁移时必须在报告中列出范围变化。
    if model.get("name") == "mf_et_jar":
        model["name"] = "et_jar"
    return model


def configs_compatible(saved_cfg, cfg):
    if _core(saved_cfg) != _core(cfg):
        return False
    for branch in CONTEXT_BRANCHES:
        before = _branch_settings(saved_cfg["model"], branch)
        after = _branch_settings(cfg["model"], branch)
        if before.get("enabled", False) and after.get("enabled", False):
            # 同时启用时不允许悄悄换结构；关闭整个分支是显式推理消融。
            if _branch_structure(before, branch) != _branch_structure(after, branch):
                return False
    for branch in NESTED_BRANCHES:
        before = _nested_settings(saved_cfg["model"], branch)
        after = _nested_settings(cfg["model"], branch)
        if (
            before["enabled"]
            and after["enabled"]
            and (_nested_structure(before, branch) != _nested_structure(after, branch))
        ):
            return False
    return True


def checkpoint_model_state(record):
    """在构建模型、读取外部初始化文件之前，拒绝声明与实际状态矛盾的文件。"""
    state = record.get("model", record)
    saved_model = record.get("config", {}).get("model", {})
    if record.get("architecture") == "ruler_direct":
        # 直接解码器保存自己的完整foundation/matching/decoder，不使用旧ET分支键。
        return state
    matching_enabled = _uses_pretrained_matching(saved_model)
    if matching_enabled and not any(key.startswith("pretrained_matching.") for key in state):
        raise ValueError("checkpoint声明启用pretrained_matching，但整个分支权重缺失；禁止外部补齐")
    foundation = _branch_settings(saved_model, "foundation")
    if foundation.get("enabled", False) and not any(
        key.startswith("encoder.foundation.model.") for key in state
    ):
        raise ValueError("checkpoint声明启用foundation，但预训练编码状态缺失；禁止外部补齐")
    for branch, prefix in CONTEXT_BRANCHES.items():
        present = any(key.startswith(prefix) for key in state)
        enabled = _branch_settings(saved_model, branch).get("enabled", False)
        if enabled:
            settings = _branch_settings(saved_model, branch)
            parameter_free = branch == "deep_encoder" and settings.get(
                "stage_blocks", [2, 3, 4, 3, 2]
            ) == [1, 1, 1, 1, 1]
            if not present and not parameter_free:
                raise ValueError(f"checkpoint声明启用{branch}，但整个分支权重缺失；禁止随机补齐")
        elif record.get("config") is not None and present:
            raise ValueError(f"checkpoint声明未启用{branch}却包含该分支权重，配置与状态矛盾")
    for branch, prefix in NESTED_BRANCHES.items():
        if record.get("config") is None:
            break
        present = any(key.startswith(prefix) for key in state)
        enabled = _nested_settings(saved_model, branch)["enabled"]
        if enabled and not present:
            raise ValueError(f"checkpoint声明启用{branch}但子分支权重缺失；禁止随机补齐")
        if present and not enabled:
            raise ValueError(f"checkpoint声明未启用{branch}却包含该子分支权重")
    return state


def load_model_weights(model, record, *, allow_base_transfer=False):
    if getattr(model, "architecture", None) == "ruler_direct":
        return model.load_checkpoint_weights(record)
    source = checkpoint_model_state(record)
    saved = record.get("config")
    if saved is not None and not configs_compatible(saved, model.cfg):
        raise ValueError("旧 checkpoint 与目标 ET 核心结构/搜索配置不同，禁止静默部分加载")
    target = model.state_dict()
    added = (
        "multifocal_frontend.",
        "decoder.sampler.",
        "decoder.birth_scorer.",
        "pretrained_matching.",
        *(
            prefix
            for item in CONTEXT_BRANCHES.values()
            for prefix in ((item,) if isinstance(item, str) else item)
        ),
    )
    missing = sorted(set(target) - set(source))
    extra = sorted(set(source) - set(target))
    matching_missing = any(key.startswith("pretrained_matching.") for key in missing)
    saved_matching = saved["model"].get("pretrained_matching", {}) if saved else None
    if matching_missing and (saved_matching is None or _uses_pretrained_matching(saved["model"])):
        raise ValueError(
            "仅声明未启用pretrained_matching的旧checkpoint允许新增分支；当前分支状态缺失"
        )
    nested_added = []
    if saved is not None:
        for branch, prefix in NESTED_BRANCHES.items():
            before = _nested_settings(saved["model"], branch)
            after = _nested_settings(model.cfg["model"], branch)
            if not before["enabled"] and after["enabled"]:
                if any(key.startswith(prefix) for key in source):
                    raise ValueError(f"旧配置未启用{branch}却携带部分权重，禁止混合初始化")
                nested_added.append(prefix)
    for branch, prefix in CONTEXT_BRANCHES.items():
        branch_missing = any(key.startswith(prefix) for key in missing)
        branch_extra = any(key.startswith(prefix) for key in extra)
        before = _branch_settings(saved["model"], branch) if saved else None
        after = _branch_settings(model.cfg["model"], branch)
        if branch_missing:
            branch_missing_keys = [key for key in missing if key.startswith(prefix)]
            only_new_children = bool(nested_added) and all(
                key.startswith(tuple(nested_added)) for key in branch_missing_keys
            )
            if before is None or (before.get("enabled", False) and not only_new_children):
                raise ValueError(f"checkpoint的{branch}分支权重缺失；仅旧配置未启用时允许新增")
            if not before.get("enabled", False) and any(key.startswith(prefix) for key in source):
                raise ValueError(f"checkpoint的{branch}配置未启用却携带部分权重，禁止混合初始化")
        if branch_extra and after.get("enabled", False):
            raise ValueError(f"checkpoint的{branch}包含目标分支没有的权重，禁止静默裁切")
    allowed = allow_base_transfer and all(k.startswith(added) for k in missing + extra)
    if (missing or extra) and not allowed:
        raise ValueError(f"checkpoint 键不匹配：missing={missing}, unexpected={extra}")
    # 固定候选是标定式搜索配置，不是学习参数；不能被同形状旧buffer静默覆盖。
    for key, value in target.items():
        if key.endswith(".inverse_depths") and key in source:
            if not value.detach().cpu().equal(source[key].detach().cpu()):
                raise ValueError(f"checkpoint 固定深度候选与当前配置不一致：{key}")
    # load_state_dict 始终保留相同键的形状核查；不会裁切训练好的层。
    # 仅已列出的可选分支允许显式增删；其余键完整匹配后仍用 strict=True。
    merged = dict(target)
    merged.update({k: v for k, v in source.items() if k in target})
    model.load_state_dict(merged, strict=True)
    changes = {}
    if saved is not None:
        for key, default in CORE_SWITCHES.items():
            field = "et_" + key
            before = saved["model"].get(field, default)
            after = model.cfg["model"].get(field, default)
            if before != after:
                changes[field] = {"checkpoint": before, "effective": after}
    if saved is not None and saved["model"].get("roi") != model.cfg["model"].get("roi"):
        changes["roi"] = {
            "checkpoint": saved["model"].get("roi"),
            "effective": model.cfg["model"].get("roi"),
        }
    if saved is not None:
        before_chunk = saved["model"].get("sweep_chunk", 8)
        after_chunk = model.cfg["model"].get("sweep_chunk", 8)
        if before_chunk != after_chunk:
            changes["sweep_chunk"] = {"checkpoint": before_chunk, "effective": after_chunk}
        before_scale = saved["model"].get("et_photo_score_scale", 1.0)
        after_scale = model.cfg["model"].get("et_photo_score_scale", 1.0)
        if before_scale != after_scale:
            changes["et_photo_score_scale"] = {"checkpoint": before_scale, "effective": after_scale}
        before_distillation = saved["model"].get("matching_distillation", {})
        after_distillation = model.cfg["model"].get("matching_distillation", {})
        if before_distillation != after_distillation:
            changes["matching_distillation"] = {
                "checkpoint": before_distillation,
                "effective": after_distillation,
            }
        before_calibration = saved["model"].get("cost_calibration", {})
        after_calibration = model.cfg["model"].get("cost_calibration", {})
        if before_calibration != after_calibration:
            changes["cost_calibration"] = {
                "checkpoint": before_calibration,
                "effective": after_calibration,
            }
        before_primary = (
            saved["model"].get("matching_evidence", {}).get("primary_descriptor", False)
        )
        after_primary = (
            model.cfg["model"].get("matching_evidence", {}).get("primary_descriptor", False)
        )
        if before_primary != after_primary:
            changes["matching_evidence.primary_descriptor"] = {
                "checkpoint": before_primary,
                "effective": after_primary,
            }
        before_refiner = _nested_settings(saved["model"], "dense_decoder.calibrated_refinement")
        after_refiner = _nested_settings(model.cfg["model"], "dense_decoder.calibrated_refinement")
        if before_refiner["enabled"] and after_refiner["enabled"]:
            before_chunk = before_refiner.get("chunk_size", 1024)
            after_chunk = after_refiner.get("chunk_size", 1024)
            if before_chunk != after_chunk:
                changes["dense_decoder.calibrated_refinement.chunk_size"] = {
                    "checkpoint": before_chunk,
                    "effective": after_chunk,
                }
        if (
            before_refiner["enabled"]
            and after_refiner["enabled"]
            and (
                before_refiner.get("feature_source")
                == after_refiner.get("feature_source")
                == "pretrained"
            )
        ):
            for field, default in (
                ("gate_multiplier", 1.0),
                ("max_step", 4.0),
                ("pass_max_steps", None),
                ("gate_source", "learned"),
                ("observation_aggregation", "reference"),
                ("camera_consensus", {}),
            ):
                before_value, after_value = before_refiner.get(field, default), after_refiner.get(
                    field, default
                )
                if before_value != after_value:
                    changes[f"dense_decoder.calibrated_refinement.{field}"] = {
                        "checkpoint": before_value,
                        "effective": after_value,
                    }
    before_matching = saved["model"].get("pretrained_matching", {}) if saved else {}
    after_matching = model.cfg["model"].get("pretrained_matching", {})
    if before_matching != after_matching:
        changes["pretrained_matching"] = {
            "checkpoint": before_matching,
            "effective": after_matching,
        }
    before_context = saved["model"].get("et_curve_context", {}) if saved else {}
    after_context = model.cfg["model"].get("et_curve_context", {})
    if before_context != after_context:
        changes["et_curve_context"] = {"checkpoint": before_context, "effective": after_context}
    context_initialization = {}
    for branch, prefix in CONTEXT_BRANCHES.items():
        before = _branch_settings(saved["model"], branch) if saved else {}
        after = _branch_settings(model.cfg["model"], branch)
        if before != after:
            changes[branch] = {"checkpoint": before, "effective": after}
        initialized = [key for key in missing if key.startswith(prefix)]
        active = after.get("enabled", False)
        context_initialization[branch] = {
            "source": (
                "explicit_initialization"
                if initialized
                else ("checkpoint" if active else "disabled")
            ),
            "initialized_keys": initialized,
        }
    matching_added = [key for key in missing if key.startswith("pretrained_matching.")]
    return {
        "missing_keys": missing,
        "unexpected_keys": extra,
        "configuration_changes": changes,
        "base_transfer": bool(missing or extra),
        "pretrained_matching_initialization": {
            "source": (
                after_matching.get("weights")
                if matching_added
                else ("checkpoint" if hasattr(model, "pretrained_matching") else "disabled")
            ),
            "prefix": after_matching.get("prefix", "matching_model.") if matching_added else None,
            "initialized_keys": matching_added,
            "frozen": hasattr(model, "pretrained_matching"),
        },
        "context_initialization": context_initialization,
        "note": "缺少的新分支保留显式初始化；没有复制/裁切旧权重",
    }
