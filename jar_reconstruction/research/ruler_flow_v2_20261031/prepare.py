"""固定独立训练配置与新架构迁移权重；不访问 TEST 深度。"""

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest, write_json
from jar.pretraining.runner import atomic_save
from jar.runtime import seed_all
from research.v2_restart_20261029.prepare import budget

FOLDER = Path(__file__).parent
OUTPUT = ROOT / "runs/active/ruler_flow_v2_20261031"
PREVIOUS = ROOT / "research/v2_restart_20261029/configs"
V2_PUBLIC = ROOT / "runs/active/v2_restart_20261029/public/best.pt"
V2_CARLA = ROOT / "runs/active/v2_restart_20261029/carla_new/best.pt"


def stage(number, name, epochs, prefixes, flow_lr, decoder_lr, fpn_lr,
          matching_lr=0., foundation_blocks=0, foundation_lr=0., correspondence=.02):
    return {
        "number": number, "name": name, "min_epochs": epochs, "max_epochs": epochs,
        "trainable_prefixes": prefixes, "foundation_blocks": foundation_blocks,
        "flow_lr": flow_lr, "decoder_lr": decoder_lr, "fpn_lr": fpn_lr,
        "matching_lr": matching_lr, "foundation_lr": foundation_lr,
        "flow_weight": 1., "correspondence_weight": correspondence,
        "actual_fine_risk": True,
    }


def make_configs():
    public = json.loads((PREVIOUS / "train_public.json").read_text())
    public.update(architecture="ruler_flow_v2", seed=20261031,
                  output=str(OUTPUT / "public"),
                  initial_model_checkpoint=str(OUTPUT / "initial_public.pt"))
    public["flow"] = {
        "objective": "meanflow", "hidden": 48, "inference_steps": 1,
        "noise_sigma": 0., "instantaneous_fraction": .5,
        "start_weight": 1., "bridge_weight": .25,
    }
    public["stages"] = [
        stage(1, "原十二候选证据保持，仅学习新流场", 1,
              ["decoder.flow_field."], 1e-3, 0., 0., correspondence=0.),
        stage(2, "流场与原解码及匹配金字塔共同适配", 2,
              ["decoder.", "matching.lateral.", "matching.smooth."],
              3e-4, 1e-5, 1e-5),
        stage(3, "匹配主干与ViT末四块低速联合收敛", 8,
              ["decoder.", "matching."], 1e-4, 5e-6, 5e-6,
              matching_lr=5e-7, foundation_blocks=4, foundation_lr=5e-8),
    ]
    public["train"].update(full_validation_every=1, checkpoint_every=200)
    carla = json.loads((PREVIOUS / "train_carla_new.json").read_text())
    carla.update(architecture="ruler_flow_v2", seed=20261101,
                 output=str(OUTPUT / "carla_new"),
                 initial_model_checkpoint=str(OUTPUT / "public/best.pt"))
    carla["flow"] = copy.deepcopy(public["flow"])
    carla["stages"] = [
        stage(1, "新公共Flow迁移CARLA四参考", 6,
              ["decoder.", "matching.lateral.", "matching.smooth."],
              1e-4, 1e-5, 5e-6),
        stage(2, "CARLA匹配与Flow共同收敛", 18,
              ["decoder.", "matching."], 5e-5, 5e-6, 2e-6, matching_lr=2e-7),
    ]
    configs = {"public": public, "carla_new": carla}
    for phase, config in configs.items():
        config["evaluation"]["test_after_training"] = False
        filename = FOLDER / "configs" / ("train_" + phase + ".json")
        if (OUTPUT / phase / "last.pt").exists():
            raise RuntimeError("训练已启动，禁止重新生成配置")
        write_json(filename, config)
        smoke = copy.deepcopy(config)
        smoke.update(smoke=True, output=str(OUTPUT / ("smoke_" + phase)))
        smoke["train"].update(workers=0, checkpoint_every=1,
                              validation_samples_per_domain=1, log_every=1)
        for item in smoke["stages"]:
            item["min_epochs"] = item["max_epochs"] = 1
        if phase == "carla_new":
            smoke["initial_model_checkpoint"] = str(OUTPUT / "smoke_public/final.pt")
        write_json(FOLDER / "configs" / ("smoke_" + phase + ".json"), smoke)
    for phase, checkpoint in (("public", V2_PUBLIC), ("carla_new", V2_CARLA)):
        for variant in ("control", "direct", "meanflow"):
            settings = {
                "checkpoint": str(checkpoint),
                "data_config": str(PREVIOUS / ("train_" + phase + ".json")),
                "output": str(OUTPUT / "pilot" / (phase + "_" + variant)),
                "variant": variant, "flow": copy.deepcopy(public["flow"]),
                "seed": 20261031, "batch_size": 2, "epochs": 1,
                "train_samples_per_domain": 128 if phase == "public" else 497,
                "workers": 2, "threads": 2, "device": "cuda:0",
                "lr": 1e-5 if phase == "public" else 5e-6,
                "new_lr": 1e-3, "aux_weight": 1., "weight_decay": 1e-4,
                "minimum_free_gib": 200, "augmentation": True,
            }
            settings["flow"]["objective"] = "direct" if variant == "direct" else "meanflow"
            write_json(FOLDER / "configs" / ("pilot_" + phase + "_" + variant + ".json"), settings)
            smoke = copy.deepcopy(settings)
            smoke.update(smoke=True, smoke_steps=2, workers=0,
                         output=str(OUTPUT / "pilot_smoke" / (phase + "_" + variant)))
            write_json(FOLDER / "configs" / ("pilot_smoke_" + phase + "_" + variant + ".json"), smoke)
    return configs


def migrate(config):
    from research.ruler_flow_v2_20261031.model import EvidenceFlowModel

    destination = OUTPUT / "initial_public.pt"
    if destination.exists():
        record = torch.load(destination, map_location="cpu", weights_only=False, mmap=True)
        if record["config"] != config or record["source_sha256"] != digest(V2_PUBLIC):
            raise ValueError("已有迁移权重的配置或原权重不同，拒绝覆盖")
        return
    seed_all(config["seed"])
    model = EvidenceFlowModel(config, initialize=False)
    old = torch.load(V2_PUBLIC, map_location="cpu", weights_only=False, mmap=True)
    model.load_v2_state(old["model"])
    record = {"architecture": model.architecture, "model": model.state_dict(),
              "config": config, "source_checkpoint": str(V2_PUBLIC),
              "source_sha256": digest(V2_PUBLIC),
              "selected": "strict_V2_migration_new_zero_residual_not_trained"}
    atomic_save(record, destination, config["train"]["minimum_free_gib"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migrate", action="store_true", help="同时生成独立迁移权重")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configs = make_configs()
    write_json(OUTPUT / "training_plan.json", {
        "public": budget(configs["public"]), "carla_new": budget(configs["carla_new"]),
        "original_weights": {str(p): digest(p) for p in (V2_PUBLIC, V2_CARLA)},
        "selection": "完整VAL选本次best，实际末步另存final；不读取TEST选架构或超参数",
        "warm_start": "仅继承本项目独立训练V2；不加载baseline任务权重或教师",
    })
    if args.migrate:
        migrate(configs["public"])


if __name__ == "__main__":
    main()
