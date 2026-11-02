"""登记30个同预算训练、完整VAL与串行阶段队列；不启动GPU。"""

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json
from research.near_evidence_20261030.pilot import prepare


def run():
    here = Path(__file__).parent
    output = ROOT / "runs/active/rulermvs_v6_20261101/full_training/paper_handoff_ablation"
    configs = here / "configs"
    configs.mkdir(exist_ok=True)
    jobs, plans = [], {}
    variants = ("v5", "marginal", "constant", "parent", "parent_oldpoint")
    for phase in ("public", "carla"):
        original = ROOT / f"research/rulermvs_v6_20261101/configs/pilot_{phase}_parent.json"
        base = json.loads(original.read_text())
        base.update(train_samples_per_domain=128 if phase == "public" else 256)
        data = json.loads(Path(base["data_config"]).read_text())
        for seed in (20261102, 20261103, 20261104):
            settings = dict(base, seed=seed)
            selection, _, _, validation, schedule = prepare(data, settings)
            key = f"{phase}_{seed}"
            plans[key] = {"training": selection, "validation": validation, "schedule": schedule,
                          "checkpoint_sha256": digest(base["checkpoint"])}
            for variant in variants:
                name = f"{phase}_{variant}_seed{seed}"
                settings = dict(base, seed=seed, variant=variant,
                                grouped_loss=variant not in ("v5", "parent_oldpoint"),
                                output=str(output / name), device="cuda:0")
                filename = configs / ("handoff_" + name + ".json")
                write_json(filename, settings)
                jobs.append({"name": name, "config": str(filename), "output": settings["output"],
                             "script": str(here / "handoff_train.py")})
    geometry = []
    for phase in ("public", "carla"):
        for variant in ("control", "no_source_geometry", "no_scale_routing", "no_jacobian_width"):
            filename = configs / f"geometry_{phase}_{variant}.json"
            if filename.exists():
                settings = json.loads(filename.read_text())
                geometry.append({"name": f"geometry_{phase}_{variant}", "config": str(filename),
                                 "output": settings["output"], "script": str(here / "geometry_train.py")})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preregistered_plan.json", {
        "seeds": [20261102, 20261103, 20261104], "variants": variants,
        "plans": plans, "selection": "fixed_last_step", "test_accessed": False,
        "phases_independent": True, "encoder_and_bridge_frozen": True,
        "constant_gate": "1可学习标量，sigmoid(-2)初值与129参数自适应门一致",
        "training_reference": "MixedDataset按(index+epoch)在有深度的相机间轮换；不是仅训练上参考",
        "objective": "v5/parent_oldpoint为原点目标，其余相同族目标",
        "full_validation": "所有末步权重均评估7957公共VAL或411新CARLA VAL；不使用TEST选模型",
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in here.glob("handoff_*.py")},
    })
    smoke_jobs = []
    for variant in ("constant", "parent_oldpoint"):
        original = next(job for job in jobs if job["name"] == f"public_{variant}_seed20261102")
        settings = json.loads(Path(original["config"]).read_text())
        name = "smoke_" + variant
        settings.update(output=str(output / name), smoke=True, smoke_steps=2, smoke_validation_samples=1)
        filename = configs / ("handoff_" + name + ".json")
        write_json(filename, settings)
        smoke_jobs.append(dict(original, name=name, config=str(filename), output=settings["output"]))
    shared_jobs = []
    for rank in range(7):
        filename = configs / f"handoff_shared_val_rank{rank}.json"
        worker_output = str(output / "shared_val" / f"rank{rank}")
        write_json(filename, {"rank": rank, "world": 7, "jobs": jobs + geometry, "output": worker_output})
        shared_jobs.append({"name": f"rank{rank}", "config": str(filename), "output": worker_output,
                            "script": str(here / "handoff_sharedval.py")})
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for filename in here.glob("*.py"):
        shutil.copy2(filename, snapshot / filename.name)
    write_json(output / "queue.json", {"output": str(output), "smoke_jobs": smoke_jobs, "train_jobs": jobs + geometry,
                                       "validation_jobs": jobs + geometry, "shared_validation_jobs": shared_jobs,
                                       "gpus": list(range(7))})
    print(json.dumps({"train_jobs": len(jobs + geometry), "full_val_jobs": len(jobs + geometry),
                      "public_steps": plans["public_20261102"]["schedule"]["steps"],
                      "carla_steps": plans["carla_20261102"]["schedule"]["steps"],
                      "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    run()
