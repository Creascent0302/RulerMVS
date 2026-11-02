"""生成 V5 独立配置和零残差迁移权重；已有配置保留，测试标签不参与准备。"""

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
OUTPUT = ROOT / "runs/active/rulermvs_v5_20261031"
PREVIOUS = ROOT / "research/v2_restart_20261029/configs"
V2_PUBLIC = ROOT / "runs/active/v2_restart_20261029/public/best.pt"
V2_CARLA = ROOT / "runs/active/v2_restart_20261029/carla_new/best.pt"


def read(filename):
    return json.loads(Path(filename).read_text())


def stage(number, name, epochs, prefixes, bridge_lr, decoder_lr, fpn_lr,
          matching_lr=0., foundation_blocks=0, foundation_lr=0., correspondence=.02):
    return {
        "number": number, "name": name, "min_epochs": epochs, "max_epochs": epochs,
        "trainable_prefixes": prefixes, "foundation_blocks": foundation_blocks,
        "bridge_lr": bridge_lr, "decoder_lr": decoder_lr, "fpn_lr": fpn_lr,
        "matching_lr": matching_lr, "foundation_lr": foundation_lr,
        "bridge_detach_backbone": True,
        "correspondence_weight": correspondence, "actual_fine_risk": True,
    }


def make_configs():
    public = read(PREVIOUS / "train_public.json")
    public.update(architecture="rulermvs_v5", seed=20261031,
                  output=str(OUTPUT / "public"),
                  initial_model_checkpoint=str(OUTPUT / "initial_public.pt"))
    public["bridge"] = {
        "enabled": True, "layers": ["layer1", "layer2"],
        "bottleneck": 32, "hidden": 128, "subcells": 2,
        "detach_backbone": True, "residual_scale": .1,
    }
    public["stages"] = [
        stage(1, "冻结V2，仅训练CNN到ViT的新连接", 1,
              ["bridge."], 1e-3, 0., 0., correspondence=0.),
        stage(2, "新连接与原候选解码及金字塔共同适配", 2,
              ["bridge.", "decoder.", "matching.lateral.", "matching.smooth."],
              3e-4, 1e-5, 1e-5),
        stage(3, "匹配深层与ViT末四块低速联合收敛", 6,
              ["bridge.", "decoder.", "matching.lateral.", "matching.smooth.",
               "matching.backbone.layer3.", "matching.backbone.layer4."],
              1e-4, 5e-6, 5e-6, matching_lr=5e-7,
              foundation_blocks=4, foundation_lr=5e-8),
    ]
    public["train"].update(full_validation_every=1, checkpoint_every=200, minimum_free_gib=200)
    carla = read(PREVIOUS / "train_carla_new.json")
    carla.update(architecture="rulermvs_v5", seed=20261101,
                 output=str(OUTPUT / "carla_new"),
                 initial_model_checkpoint=str(OUTPUT / "public/best.pt"))
    carla["bridge"] = copy.deepcopy(public["bridge"])
    carla["stages"] = [
        stage(1, "公共V5适配新CARLA四参考", 6,
              ["bridge.", "decoder.", "matching.lateral.", "matching.smooth."],
              1e-4, 1e-5, 5e-6),
        stage(2, "CARLA深层匹配与连接低速收敛", 18,
              ["bridge.", "decoder.", "matching.lateral.", "matching.smooth.",
               "matching.backbone.layer3.", "matching.backbone.layer4."],
              5e-5, 5e-6, 2e-6, matching_lr=2e-7),
    ]
    configs = {}
    for phase, default in (("public", public), ("carla_new", carla)):
        default["evaluation"]["test_after_training"] = False
        filename = FOLDER / "configs" / ("train_" + phase + ".json")
        # 参数以文件为准，重复准备不覆盖人工调整。
        config = read(filename) if filename.exists() else default
        if config.get("architecture") != "rulermvs_v5":
            raise ValueError(f"{filename}不是V5配置")
        if not filename.exists():
            write_json(filename, config)
        configs[phase] = config
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
        for variant in ("control", "layer1", "layer12"):
            bridge = copy.deepcopy(configs["public"]["bridge"])
            bridge["enabled"] = variant != "control"
            bridge["layers"] = ["layer1"] if variant == "layer1" else ["layer1", "layer2"]
            pilot = {
                "checkpoint": str(checkpoint),
                "data_config": str(PREVIOUS / ("train_" + phase + ".json")),
                "output": str(OUTPUT / "pilot" / (phase + "_" + variant)),
                "variant": variant, "bridge": bridge,
                "seed": 20261031, "batch_size": 2, "epochs": 1,
                "train_samples_per_domain": 128 if phase == "public" else 497,
                "workers": 2, "threads": 2, "device": "cuda:0",
                "lr": 1e-5 if phase == "public" else 5e-6,
                "new_lr": 1e-3, "weight_decay": 1e-4,
                "minimum_free_gib": 200, "augmentation": True,
            }
            filename = FOLDER / "configs" / ("pilot_" + phase + "_" + variant + ".json")
            if not filename.exists():
                write_json(filename, pilot)
    return configs


def migrate(config):
    from research.rulermvs_v5_20261031.model import V5Model

    destination = OUTPUT / "initial_public.pt"
    if destination.exists():
        record = torch.load(destination, map_location="cpu", weights_only=False, mmap=True)
        if record["config"] != config or record["source_sha256"] != digest(V2_PUBLIC):
            raise ValueError("已有迁移权重的配置或源权重不同，拒绝覆盖")
        return
    seed_all(config["seed"])
    model = V5Model(config, initialize=False)
    old = torch.load(V2_PUBLIC, map_location="cpu", weights_only=False, mmap=True)
    model.load_v2_state(old["model"])
    atomic_save({
        "architecture": model.architecture, "model": model.state_dict(), "config": config,
        "source_checkpoint": str(V2_PUBLIC), "source_sha256": digest(V2_PUBLIC),
        "selected": "strict_V2_migration_zero_bridge_not_trained",
    }, destination, config["train"]["minimum_free_gib"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migrate", action="store_true", help="同时生成独立的V2→V5迁移权重")
    args = parser.parse_args()
    if any((OUTPUT / phase / "last.pt").exists() for phase in ("public", "carla_new")):
        raise RuntimeError("正式训练已经启动，禁止重新准备或迁移")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configs = make_configs()
    write_json(OUTPUT / "training_plan.json", {
        "public": budget(configs["public"]), "carla_new": budget(configs["carla_new"]),
        "original_weights": {str(p): digest(p) for p in (V2_PUBLIC, V2_CARLA)},
        "selection": "完整VAL选本次best，实际末步另存final；TEST不选择结构或权重",
        "warm_start": "仅继承本项目独立训练V2，不加载baseline任务权重或教师",
        "gradient_policy": "连接始终停止回传ResNet输入特征；layer1/2冻结，ViT仍向连接回传梯度",
        "stage_rationale": [
            "1轮新连接独立学习，使输入补充先适配成熟下游，保护原候选判断。",
            "2轮候选解码/FPN适配，让新编码信息参与原候选判断与细化。",
            "6轮低学习率解冻深层，兼顾分布适配与保留通用表示；具体轮数是待验证预算，不声称理论最优。",
        ],
        "comparison_boundary": "V5增加热启动训练预算，完整TEST相对原V2不构成单一结构因果；短预算同训练对照另存。",
    })
    if args.migrate:
        migrate(configs["public"])


if __name__ == "__main__":
    main()
