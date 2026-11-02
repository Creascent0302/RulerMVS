"""实际GPU冒烟结束后核查步数、两输出头梯度、旧参数冻结和严格恢复。"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/active/ruler_flow_20261030"


def run(settings=None):
    settings = settings or {}
    arms = settings.get("arms", ["control", "direct", "flow", "flow_no_direction"])
    records = {}
    for group in ("public", "carla"):
        schedule = None
        original_initial = json.loads((OUTPUT / f"smoke_{group}_control/initial.json").read_text())
        direct_initial = json.loads((OUTPUT / f"smoke_{group}_direct/initial.json").read_text())
        for arm in arms:
            name = f"smoke_{group}_{arm}"
            folder = OUTPUT / name
            read = lambda key: json.loads((folder / f"{key}.json").read_text())
            completed = read("completed")
            if completed["steps"] != 1 or not completed["gradients_finite"]:
                raise RuntimeError(f"{name}步数或梯度异常")
            if schedule is None:
                schedule = completed["schedule_sha256"]
            if schedule != completed["schedule_sha256"]:
                raise RuntimeError(f"{name}不是相同训练批次")
            checkpoint, modules = read("checkpoint_check"), read("new_module_checks")
            if not checkpoint["passed"] or not modules["passed"]:
                raise RuntimeError(f"{name}恢复或新头检查失败")
            if arm != "control":
                frozen = read("old_model_frozen")
                if not frozen["passed"] or completed["trainable_parameters"] != completed["new_parameter_count"]:
                    raise RuntimeError(f"{name}冻结规则失败")
                variant = read("settings")["variant"]
                expected_heads = {"velocity", "quality"} if variant.get("learn_quality", True) else {"velocity"}
                if set(modules["heads"]) != expected_heads:
                    raise RuntimeError(f"{name}缺独立输出头检查")
                initial = read("initial")
                if initial != direct_initial:
                    raise RuntimeError(f"{name}相同零初始化新头的初始指标不同")
                for split, domains in initial.items():
                    for domain, metrics in domains.items():
                        if metrics["stages_stride8"]["coarse"] != original_initial[split][domain]["stages_stride8"]["coarse"]:
                            raise RuntimeError(f"{name}改变了原coarse指标")
            records[name] = {"passed": True, "trainable_parameters": completed["trainable_parameters"],
                             "checkpoint": checkpoint, "heads": modules["heads"]}
    (OUTPUT / settings.get("report", "smoke_verification.json")).write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(records)}组真实GPU冒烟及恢复检查全部通过", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    run(json.loads((ROOT / args.config).read_text()) if args.config else None)
