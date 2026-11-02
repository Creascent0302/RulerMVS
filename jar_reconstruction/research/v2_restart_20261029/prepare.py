"""固定新训练配置、CARLA四参考VAL和曝光预算；不读取TEST深度。"""

import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.v2_training_20261028.sampling import RebalancedFullPassSampler

FOLDER = Path(__file__).parent
OUTPUT = ROOT / "runs/active/v2_restart_20261029"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def stage(number, name, epochs, decoder, fpn, matching, blocks=0, foundation=0., correspondence=.05):
    prefixes = ["decoder.", "matching.lateral.", "matching.smooth."]
    if matching:
        prefixes = ["decoder.", "matching."]
    return {
        "number": number, "name": name, "min_epochs": epochs, "max_epochs": epochs,
        "trainable_prefixes": prefixes, "foundation_blocks": blocks,
        "decoder_lr": decoder, "fpn_lr": fpn, "matching_lr": matching,
        "foundation_lr": foundation, "correspondence_weight": correspondence,
        "actual_fine_risk": number > 1,
    }


def budget(config):
    lengths = [len(Path(d["manifests"]["train"]).read_text().splitlines()) for d in config["datasets"]]
    sampler = RebalancedFullPassSampler(lengths, 0, 0, 8, 4, 1, config["seed"],
                                       domain_multipliers=[d["train_multiplier"] for d in config["datasets"]],
                                       domain_schedule="balanced_prefix")
    total = sum(d["total_exposures"] for d in sampler.domain_counts.values())
    epochs = sum(s["max_epochs"] for s in config["stages"])
    return {"unique_train_groups": sum(lengths), "updates_per_epoch": len(sampler.updates),
            "epochs": epochs, "total_updates": epochs * len(sampler.updates),
            "exposures_per_epoch": total, "total_exposures": epochs * total,
            "domains": {config["datasets"][i]["name"]: dict(values,
                        fraction=values["total_exposures"] / total)
                        for i, values in sampler.domain_counts.items()}}


def main():
    if (OUTPUT / "public/last.pt").exists() or (OUTPUT / "carla_new/last.pt").exists():
        raise RuntimeError("新训练已存在checkpoint，禁止重新生成或改变训练配置")
    previous = ROOT / "research/v2_training_20261028/configs"
    public = json.loads((previous / "train_public.json").read_text())
    public.pop("teacher_checkpoint")
    public.update(phase="public", seed=20261029, output=str(OUTPUT / "public"),
                  initial_model_checkpoint="", loss={})
    public["train"].update(stage_policy="continuous_new", full_validation_every=2,
                           validation_samples_per_domain=64, checkpoint_every=200,
                           domain_schedule="balanced_prefix")
    public["evaluation"] = {"precision": "fp32", "test_after_training": False}
    multipliers = [1., 1., 8., 1., 1.5, 1., 4.]
    for domain, multiplier in zip(public["datasets"], multipliers):
        domain["train_multiplier"] = multiplier
    public["stages"] = [
        stage(1, "新匹配金字塔与候选解码预热", 3, 2e-4, 2e-4, 0., correspondence=.1),
        stage(2, "匹配骨干与标尺候选联合学习", 5, 1e-4, 1e-4, 1e-5),
        stage(3, "低速适配ViT末四块并收敛", 12, 5e-5, 5e-5, 5e-6, 4, 5e-7, .02),
    ]
    carla = json.loads((previous / "train_carla_new.json").read_text())
    carla.pop("teacher_checkpoint")
    carla.update(phase="carla_new", seed=20261030, output=str(OUTPUT / "carla_new"),
                 initial_model_checkpoint=str(OUTPUT / "public/best.pt"), loss={})
    carla["train"].update(stage_policy="continuous_new", full_validation_every=1,
                          validation_samples_per_domain=32, domain_schedule="balanced_prefix")
    carla["evaluation"] = {"precision": "fp32", "test_after_training": False}
    carla["stages"] = [
        stage(1, "新CARLA候选与金字塔适配", 6, 1e-5, 5e-6, 0., correspondence=.02),
        stage(2, "四参考匹配适配并收敛", 18, 5e-6, 2e-6, 2e-7, correspondence=.02),
    ]
    base = carla["datasets"][0]
    source = ROOT / "runs/active/v2_training_20261028/carla_prepared/val_supervised.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    validation_domains = []
    for reference, camera in enumerate(base["camera"]["wide_ids"]):
        selected = [row for row in rows if camera in row["depth_supervised_camera_ids"]
                    and row["valid_camera_count"] >= 2]
        path = OUTPUT / "manifests" / ("val_" + camera + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
        domain = copy.deepcopy(base)
        domain.update(name="CARLA_new_" + camera, reference_index=reference, selection_depth_scale=10.)
        domain["manifests"]["val"] = str(path)
        validation_domains.append(domain)
    carla["validation_datasets"] = validation_domains
    plans = {}
    for name, config in (("public", public), ("carla_new", carla)):
        write(FOLDER / "configs" / ("train_" + name + ".json"), config)
        plans[name] = budget(config)
        smoke = copy.deepcopy(config)
        smoke["output"] = str(OUTPUT / ("smoke_" + name))
        smoke["smoke"] = True
        smoke["train"].update(validation_samples_per_domain=1, workers=0, checkpoint_every=1)
        for item in smoke["stages"]:
            item["min_epochs"] = item["max_epochs"] = 1
        if name == "carla_new":
            smoke["initial_model_checkpoint"] = str(OUTPUT / "smoke_public/final.pt")
        write(FOLDER / "configs" / ("smoke_" + name + ".json"), smoke)
    plans["carla_validation_groups"] = {
        d["name"]: len(Path(d["manifests"]["val"]).read_text().splitlines()) for d in validation_domains}
    plans["initialization_sha256"] = {
        key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for key, path in public["initialization"].items()}
    write(OUTPUT / "preparation.json", plans)
    print(json.dumps(plans, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
