"""由机制证据注册的最后六组短训；不改变原八组配置或产物。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path("research/ruler_flow_20261030")
OUTPUT = Path("runs/active/ruler_flow_20261030")


def generate():
    folder = ROOT / STUDY / "configs"
    arms = {
        "flow_anchor": {"objective": "flow", "anchor_fraction": .5},
        "direct_rank_fixed": {"objective": "direct", "learn_quality": False},
        "flow_rank_fixed": {"objective": "flow", "learn_quality": False},
    }
    queues = {"followup_smoke": [], "followup_formal": []}
    for group_index, group in enumerate(("public", "carla")):
        base = json.loads((folder / f"{group}_flow.json").read_text())
        for arm_index, (arm, changes) in enumerate(arms.items()):
            name = f"{group}_{arm}"
            settings = dict(base, output=str(OUTPUT / name), device=f"cuda:{3 * group_index + arm_index}",
                            variant={**base["variant"], **changes},
                            evaluation_steps=[1] if arm.startswith("direct") else [1, 2, 4])
            (folder / f"{name}.json").write_text(json.dumps(settings, indent=2) + "\n")
            smoke = dict(settings, smoke=True, smoke_steps=1, smoke_validation_samples=2,
                         output=str(OUTPUT / f"smoke_{name}"))
            (folder / f"smoke_{name}.json").write_text(json.dumps(smoke, indent=2) + "\n")
            queues["followup_formal"].append([str(STUDY / "configs" / f"{name}.json")])
            queues["followup_smoke"].append([str(STUDY / "configs" / f"smoke_{name}.json")])
    for phase, workers in queues.items():
        (folder / f"queue_{phase}.json").write_text(json.dumps({
            "name": phase + "_queue", "output": str(OUTPUT / "logs"), "workers": workers,
        }, indent=2) + "\n")
    (folder / "verify_followup.json").write_text(json.dumps({
        "arms": list(arms), "report": "followup_smoke_verification.json",
    }, indent=2) + "\n")
    (folder / "summary_followup.json").write_text(json.dumps({
        "output": str(OUTPUT), "arms": ["control", "direct", "flow", "flow_no_direction", *arms],
        "report": "summary_followup.json",
    }, indent=2) + "\n")
    for group in ("public", "carla"):
        settings = json.loads((folder / f"latency_{group}.json").read_text())
        settings["methods"].pop("reduced4")
        settings["methods"]["direct_rank_fixed_steps1"] = {
            "checkpoint": str(OUTPUT / f"{group}_direct_rank_fixed/final.pt"), "steps": 1,
        }
        for steps in (1, 2):
            settings["methods"][f"flow_rank_fixed_steps{steps}"] = {
                "checkpoint": str(OUTPUT / f"{group}_flow_rank_fixed/final.pt"), "steps": steps,
            }
        (folder / f"latency_final_{group}.json").write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    generate()
