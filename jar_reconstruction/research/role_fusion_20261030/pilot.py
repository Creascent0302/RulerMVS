"""职责分离融合的短预算入口，复用已验证训练流程，不改生产模型。"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.models.position_cache import install_position_cache
from jar.pretraining.model import PretrainingModel
from research.early_geometry_20261030 import pilot as engine
from research.near_evidence_20261030.pilot import parameter_hash, path


EXPERIMENT = "role_fusion_20261030"


def install(model, settings):
    options = settings.get("variant", {})
    if options:
        from research.role_fusion_20261030.variants import install as install_variant

        install_variant(model, options)
    return model


def load_model(filename, device="cpu"):
    saved = torch.load(path(filename), map_location="cpu", weights_only=False, mmap=True)
    marker = saved.get("experiment")
    if marker is not None and marker != EXPERIMENT:
        raise ValueError(f"不能用本轮入口静默加载其他研究架构：{marker}")
    model = PretrainingModel(saved["config"], initialize=False)
    if marker == EXPERIMENT:
        install(model, saved["pilot_settings"])
    model.load_state_dict(saved["model"], strict=True)
    model.requires_grad_(False).eval().to(device)
    install_position_cache(model.foundation.model)
    return model, saved


def sources():
    names = [
        "jar/pretraining/model.py", "jar/pretraining/encoders.py",
        "jar/pretraining/data.py", "jar/pretraining/losses.py",
        "jar/models/ruler_direct.py", "jar/models/ruler_proposals.py",
        "research/early_geometry_20261030/pilot.py",
        "research/near_evidence_20261030/pilot.py",
        "research/v2_training_20261028/objectives.py",
    ]
    names.extend(str(file.relative_to(ROOT)) for file in
                 (ROOT / "research" / EXPERIMENT).glob("*.py"))
    return sorted(set(names))


def run(settings):
    settings = dict(settings)
    # 非恒等改造不伪装成同一初始函数；仍保存相同起始权重的实际初始评测。
    settings["verify_initial"] = bool(settings.get("initial_identity", True))
    native_objective = engine.depth_objective
    native_save = engine.atomic_save
    extra_terms = []
    gradient_maxima = {}
    gradient_handles = []
    original_state_check = {}

    def install_tracked(model, options):
        original_names = set(dict(model.named_parameters()))
        if settings.get("verify_old_model_frozen", False):
            state = {name: value for name, value in model.state_dict().items()
                     if name != "decoder.depth_lattice"}
            original_state_check.update(keys=set(state), sha256=parameter_hash(state),
                                        parameter_tensors=len(original_names))
        install(model, options)
        for name, parameter in model.named_parameters():
            if name in original_names:
                continue
            gradient_maxima[name] = None

            def record(gradient, key=name):
                maximum = gradient.detach().abs().amax()
                previous = gradient_maxima[key]
                gradient_maxima[key] = maximum if previous is None else torch.maximum(previous, maximum)

            gradient_handles.append(parameter.register_hook(record))
        return model

    def objective(prediction, inputs, targets, domain, loss_config, correspondence_weight):
        terms = native_objective(prediction, inputs, targets, domain, loss_config, correspondence_weight)
        options = settings.get("auxiliary", {})
        if options:
            from research.role_fusion_20261030.variants import training_loss

            extra = training_loss(prediction, inputs, targets, domain, options)
            if "total" not in extra or not torch.is_tensor(extra["total"]):
                raise ValueError("附加损失必须返回含可反传total张量的字典")
            terms["total"] = terms["total"] + extra["total"]
            extra_terms.append({
                "step": len(extra_terms) + 1, "dataset": domain["name"],
                **{name: float(value.detach()) if torch.is_tensor(value) else value
                   for name, value in extra.items()},
            })
        return terms

    def save_checkpoint(payload, *args, **kwargs):
        payload["experiment"] = EXPERIMENT
        payload["initial_identity_expected"] = settings["verify_initial"]
        return native_save(payload, *args, **kwargs)

    # 临时替换只在单个实验进程有效，退出即恢复；生产模型源码保持不变。
    with ExitStack() as stack:
        for name, replacement in {
            "install": install_tracked, "sources": sources,
            "depth_objective": objective, "atomic_save": save_checkpoint,
        }.items():
            stack.enter_context(patch.object(engine, name, replacement))
        try:
            engine.run(settings)
        finally:
            for handle in gradient_handles:
                handle.remove()
            if extra_terms:
                filename = path(settings["output"]) / "auxiliary.jsonl"
                filename.write_text("".join(json.dumps(row) + "\n" for row in extra_terms))
    if gradient_maxima:
        output = path(settings["output"])
        completed = json.loads((output / "completed.json").read_text())
        gradients = {name: float(value) if value is not None else 0.
                     for name, value in gradient_maxima.items()}
        modules = {}
        for module in ("context_projection", "context_null", "handoff_gate"):
            names = [name for name in gradients if module in name.split(".")]
            if names:
                modules[module] = {
                    "maximum_absolute_gradient": max(gradients[name] for name in names),
                    "maximum_parameter_change": max(completed["new_parameter_changes"][name] for name in names),
                }
        passed = all(value["maximum_absolute_gradient"] > 0 and value["maximum_parameter_change"] > 0
                     for value in modules.values())
        (output / "new_module_checks.json").write_text(json.dumps({
            "passed": passed, "parameters": gradients, "modules": modules,
        }, indent=2) + "\n")
        if not passed:
            raise RuntimeError("新增context或handoff模块没有获得非零梯度/参数更新")
    if original_state_check:
        saved = torch.load(path(settings["output"]) / "final.pt", map_location="cpu", mmap=True, weights_only=False)
        state = {name: value for name, value in saved["model"].items()
                 if name in original_state_check["keys"]}
        final_hash = parameter_hash(state)
        passed = final_hash == original_state_check["sha256"]
        (path(settings["output"]) / "old_model_frozen.json").write_text(json.dumps({
            "passed": passed, "initial_sha256": original_state_check["sha256"],
            "final_sha256": final_hash, "parameter_tensors": original_state_check["parameter_tensors"],
            "state_tensors_checked": len(state), "excluded_runtime_buffers": ["decoder.depth_lattice"],
            "scope": "全部旧参数及持久统计保持；仅排除按domain可重建的运行时深度格点。",
        }, ensure_ascii=False, indent=2) + "\n")
        if not passed:
            raise RuntimeError("只训练新增模块时旧参数或持久统计发生改变")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(path(parser.parse_args().config).read_text()))
