"""冻结八个配置及同预算清单；仅CPU，不读RGB/深度，不启动训练。"""

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json
from research.near_evidence_20261030.pilot import prepare
from research.paper_ablation_20261102.geometry import SETTINGS, DESCRIPTIONS


def main():
    folder = Path(__file__).parent
    output = ROOT / "runs/active/rulermvs_v6_20261101/full_training/paper_geometry_ablation"
    configs = folder / "configs"
    configs.mkdir(exist_ok=True)
    plan = {"completed": False, "prepared": True, "test_accessed": False,
            "selection": "fixed_last_step", "seed": 20261102, "phases": {},
            "scope": "同V5起点、冻结编码器的独立几何用途适配；不声称从零消融"}
    recovery = ROOT / "runs/active/rulermvs_v6_20261101/full_training/evaluation_recovery/configs"
    for phase, name, samples in (("public", "public", 128), ("carla", "carla_new", 256)):
        checkpoint = ROOT / f"runs/active/rulermvs_v5_20261031/{name}/best.pt"
        data = ROOT / f"research/rulermvs_v6_20261101/configs/data_{phase}.json"
        common = {
            "seed": 20261102, "checkpoint": str(checkpoint), "data_config": str(data),
            "device": "cuda:0", "variant": "parent", "grouped_loss": True,
            "train_samples_per_domain": samples, "batch_size": 2, "epochs": 2,
            "lr": 1e-5, "new_lr": 1e-3, "weight_decay": 1e-4,
            "trainable_prefixes": ["decoder."], "minimum_free_gib": 200,
        }
        selected, _, _, val_plan, schedule = prepare(json.loads(data.read_text()), common)
        phase_plan = {"checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
                      "data_config_sha256": digest(data), "training": selected,
                      "validation": val_plan, "schedule": schedule, "jobs": {}}
        for ablation in SETTINGS:
            config = dict(common, geometry_ablation=ablation,
                          output=str(output / "training" / phase / ablation),
                          experiment_notes=[DESCRIPTIONS[ablation],
                              "所有组相同候选数、可训练参数集合、优化器、样本顺序和固定末步规则。",
                              "CARLA从对应V5 CARLA起点独立适配；不是接续公共组。"])
            filename = configs / f"geometry_{phase}_{ablation}.json"
            if filename.exists() and json.loads(filename.read_text()) != config:
                raise FileExistsError(f"配置已有不同内容，不覆盖: {filename}")
            write_json(filename, config)
            phase_plan["jobs"][ablation] = {
                "config": str(filename), "config_sha256": digest(filename),
                "output": config["output"], "model_changes": SETTINGS[ablation],
            }
        plan["phases"][phase] = phase_plan
        evaluation = json.loads((recovery / f"{name}_main_val.json").read_text())
        evaluation["methods"] = {
            f"geometry_{ablation}": {
                "kind": "own", "weights": str(output / "training" / phase / ablation / "final.pt"),
                "input_mode": "equal_pixels", "inference": {"spatial_layout": True},
            } for ablation in SETTINGS
        }
        evaluation.update(output=str(output / "full_val" / phase), gpus=[0],
                          own_weights=evaluation["methods"]["geometry_control"]["weights"],
                          evaluation_split="val")
        evaluation.pop("speed", None)
        for domain in evaluation["datasets"]:
            domain["manifests"] = {k: v for k, v in domain["manifests"].items() if k in ("train", "val")}
        evaluation["protocol_notes"] = [
            "只用完整共同VAL；固定末步，不访问TEST或重新挑best。",
            "三用途单因素消融；TRAIN/VAL起点/候选预算相同；不是从零训练。",
            "几何关闭是有针对性的现有配置，不删除精确投影或有效掩码。",
        ]
        filename = configs / f"geometry_{phase}_full_val.json"
        write_json(filename, evaluation)
        phase_plan["full_val_config"] = str(filename)
    write_json(output / "plan.json", plan)
    print(json.dumps({phase: {"steps_per_group": row["schedule"]["steps"],
                             "exposures_per_group": row["schedule"]["exposures"],
                             "groups": len(row["jobs"])}
                      for phase, row in plan["phases"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
