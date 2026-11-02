"""八组真实前后向完成后，核对恒等初值、训练顺序和冻结编码器。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/active/role_fusion_20261030"


def run():
    records = {}
    for group in ("public", "carla"):
        control = OUTPUT / f"smoke_{group}_control"
        initial = json.loads((control / "initial.json").read_text())
        plan = json.loads((control / "plan.json").read_text())
        for arm in ("control", "context", "carry", "both"):
            name = f"smoke_{group}_{arm}"
            folder = OUTPUT / name
            done = json.loads((folder / "completed.json").read_text())
            current_plan = json.loads((folder / "plan.json").read_text())
            current_initial = json.loads((folder / "initial.json").read_text())
            settings = json.loads((folder / "settings.json").read_text())
            if not done["completed"] or not done["gradients_finite"]:
                raise RuntimeError(f"{name}没有完成有限梯度训练")
            if current_plan["schedule"] != plan["schedule"]:
                raise RuntimeError(f"{name}训练顺序不一致")
            if done["initial_encoder_sha256"] != done["final_encoder_sha256"]:
                raise RuntimeError(f"{name}冻结编码器已改变")
            if not all(value > 0 for value in done["new_branch_gradient_max"].values()):
                raise RuntimeError(f"{name}新增分支没有非零梯度")
            module_checks = None
            if done["new_parameter_count"]:
                module_checks = json.loads((folder / "new_module_checks.json").read_text())
                if not module_checks["passed"]:
                    raise RuntimeError(f"{name}新增独立模块未通过梯度和更新检查")
            identity = settings.get("initial_identity", True)
            tensors = None
            if identity:
                tensors = json.loads((folder / "initial_tensor_check.json").read_text())
                if current_initial != initial or not all(
                    item["tensors_bitwise_equal"] > 0 and item["real_foundation_blocks"] == 12
                    for item in tensors.values()
                ):
                    raise RuntimeError(f"{name}声明恒等但未通过完整网络初值检查")
            records[name] = {"initial_identity_expected": identity,
                             "initial_metrics_equal_control": current_initial == initial,
                             "initial_tensor_check": tensors,
                             "new_parameter_count": done["new_parameter_count"],
                             "new_module_checks": module_checks,
                             "new_branch_gradient_max": done["new_branch_gradient_max"]}
    filename = OUTPUT / "smoke_verification.json"
    if filename.exists():
        raise FileExistsError(f"拒绝覆盖冒烟记录：{filename}")
    filename.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print("八组真实GPU冒烟与初始化/冻结编码器/新增梯度检查通过", flush=True)


if __name__ == "__main__":
    run()
