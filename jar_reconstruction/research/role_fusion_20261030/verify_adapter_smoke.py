"""两组只训练新增参数的实际GPU冒烟检查。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/active/role_fusion_20261030"


def run():
    records = {}
    for group in ("public", "carla"):
        folder = OUTPUT / f"smoke_{group}_both_adapters"
        done = json.loads((folder / "completed.json").read_text())
        frozen = json.loads((folder / "old_model_frozen.json").read_text())
        modules = json.loads((folder / "new_module_checks.json").read_text())
        if not (done["completed"] and done["gradients_finite"] and frozen["passed"] and modules["passed"]):
            raise RuntimeError(f"{group}新增参数独立训练冒烟未通过")
        if done["trainable_parameters"] != done["new_parameter_count"]:
            raise RuntimeError(f"{group}有旧参数进入训练")
        records[group] = {"completed": done, "old_model": frozen, "modules": modules}
    filename = OUTPUT / "adapter_smoke_verification.json"
    if filename.exists():
        raise FileExistsError("拒绝覆盖已有冒烟证据")
    filename.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print("两组真实GPU冒烟：仅新增参数更新，全部旧权重和统计保持", flush=True)


if __name__ == "__main__":
    run()
