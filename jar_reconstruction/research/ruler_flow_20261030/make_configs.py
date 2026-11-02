"""写入固定对照配置；只生成配置，不运行训练。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path("research/ruler_flow_20261030")
OUTPUT = Path("runs/active/ruler_flow_20261030")


def generate():
    folder = ROOT / STUDY / "configs"
    folder.mkdir(parents=True, exist_ok=True)
    variants = {
        "control": {},
        "direct": {"objective": "direct", "inference_steps": 1, "direction": True, "sigma": .05, "hidden": 32},
        "flow": {"objective": "flow", "inference_steps": 1, "direction": True, "sigma": .05, "hidden": 32},
        "flow_no_direction": {"objective": "flow", "inference_steps": 1, "direction": False, "sigma": .05, "hidden": 32},
    }
    queues = {"smoke": [], "formal": []}
    for group_index, group in enumerate(("public", "carla")):
        for arm_index, (arm, variant) in enumerate(variants.items()):
            name = f"{group}_{arm}"
            settings = {
                "checkpoint": f"runs/active/v2_restart_20261029/{'public' if group == 'public' else 'carla_new'}/best.pt",
                "data_config": f"research/v2_restart_20261029/configs/train_{'public' if group == 'public' else 'carla_new'}.json",
                "output": str(OUTPUT / name), "device": f"cuda:{4 * group_index + arm_index}",
                "seed": 20261030, "train_samples_per_domain": 128 if group == "public" else 497,
                "epochs": 1, "batch_size": 2, "lr": 1e-5 if group == "public" else 5e-6,
                "new_lr": .001, "weight_decay": .0001, "grad_clip": 1.,
                "actual_fine_risk": True, "correspondence_weight": 0., "augmentation": True,
                "workers": 2, "threads": 2, "minimum_free_gib": 200,
                "variant": variant, "aux_weight": 1.,
                "evaluation_steps": [1, 2, 4] if arm.startswith("flow") else [1],
                "evaluate_reduced4": arm == "direct",
            }
            (folder / f"{name}.json").write_text(json.dumps(settings, indent=2) + "\n")
            smoke = dict(settings, smoke=True, smoke_steps=1, smoke_validation_samples=2,
                         output=str(OUTPUT / f"smoke_{name}"), evaluate_reduced4=False)
            (folder / f"smoke_{name}.json").write_text(json.dumps(smoke, indent=2) + "\n")
            queues["formal"].append([str(STUDY / "configs" / f"{name}.json")])
            queues["smoke"].append([str(STUDY / "configs" / f"smoke_{name}.json")])
    for phase, workers in queues.items():
        (folder / f"queue_{phase}.json").write_text(json.dumps({
            "name": phase + "_queue", "output": str(OUTPUT / "logs"), "workers": workers,
        }, indent=2) + "\n")
    for group in ("public", "carla"):
        original = f"runs/active/v2_restart_20261029/{'public' if group == 'public' else 'carla_new'}/best.pt"
        methods = {
            "frozen": {"checkpoint": original},
            "control": {"checkpoint": str(OUTPUT / f"{group}_control/final.pt")},
            "reduced4": {"checkpoint": original, "variant": variants["direct"], "steps": 0},
            "direct_steps1": {"checkpoint": str(OUTPUT / f"{group}_direct/final.pt"), "steps": 1},
        }
        for steps in (1, 2, 4):
            methods[f"flow_steps{steps}"] = {"checkpoint": str(OUTPUT / f"{group}_flow/final.pt"), "steps": steps}
        methods["flow_no_direction_steps1"] = {
            "checkpoint": str(OUTPUT / f"{group}_flow_no_direction/final.pt"), "steps": 1,
        }
        (folder / f"latency_{group}.json").write_text(json.dumps({
            "output": str(OUTPUT / f"latency_{group}.json"), "device": "cuda:0",
            "data_config": f"research/v2_restart_20261029/configs/train_{'public' if group == 'public' else 'carla_new'}.json",
            "datasets": ["TartanAir_forest", "VKITTI2"] if group == "public" else ["CARLA_new_stereo_up"],
            "methods": methods, "warmup": 10, "repeats": 40,
            "reference_modes": ["single"] if group == "public" else ["single", "all"],
        }, indent=2) + "\n")


if __name__ == "__main__":
    generate()
