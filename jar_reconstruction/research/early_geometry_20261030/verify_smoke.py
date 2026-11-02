"""只有实际十组 GPU 冒烟全部完成，才允许进入固定预算训练。"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/active/early_geometry_20261030"
ARMS = ("control", "appearance", "early", "late", "both")


def run():
    records = {}
    for group in ("public", "carla"):
        control = OUTPUT / f"smoke_{group}_control"
        initial = json.loads((control / "initial.json").read_text())
        plan = json.loads((control / "plan.json").read_text())
        for arm in ARMS:
            name = f"smoke_{group}_{arm}"
            folder = OUTPUT / name
            done = json.loads((folder / "completed.json").read_text())
            same = json.loads((folder / "initial.json").read_text()) == initial
            current_plan = json.loads((folder / "plan.json").read_text())
            tensors = json.loads((folder / "initial_tensor_check.json").read_text())
            assert done["completed"] and done["gradients_finite"] and same, name
            assert current_plan["schedule"] == plan["schedule"], name
            assert done["initial_encoder_sha256"] == done["final_encoder_sha256"], name
            assert all(row["real_foundation_blocks"] == 12 and row["tensors_bitwise_equal"] > 0
                       for row in tensors.values()), name
            assert all(value > 0 for value in done["new_branch_gradient_max"].values()), name
            records[name] = {"initial_metrics_identical": same, "initial_tensors": tensors,
                             "new_branch_gradient_max": done["new_branch_gradient_max"],
                             "old_encoders_unchanged": True, "new_parameters": done["new_parameter_count"]}
    (OUTPUT / "smoke_verification.json").write_text(json.dumps(records, indent=2) + "\n")
    print("十组真实GPU前后向、初值逐张量一致、冻结参数校验全部通过")


if __name__ == "__main__":
    run()
