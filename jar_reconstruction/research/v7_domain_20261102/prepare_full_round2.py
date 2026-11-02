"""读取一个明确的实验配方，生成独立全训配置和初始化；本脚本不启动训练。"""

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.v7_domain_20261102.full_training_round2 import convert_v6_checkpoint


def make_config(recipe, phase):
    phase_name = "public" if phase == "public" else "carla_new"
    source = ROOT / f"research/rulermvs_v6_20261101/configs/train_{phase_name}.json"
    config = json.loads(source.read_text())
    kind = recipe["prior_kind"]
    architecture = {"fixed": "rulermvs_v7", "coordinate": "rulermvs_v7_coordinate"}[kind]
    settings = recipe["phases"][phase]
    output = Path(recipe["output"])
    config.update(architecture=architecture, prior_kind=kind,
                  v7={"projective_center": False, "context_prior": True},
                  seed=recipe.get("seed", 20261102), output=str(output / phase),
                  initial_model_checkpoint=str(output / "initialization" / f"{phase}.pt"),
                  domain_augmentation=recipe.get("domain_augmentation", True),
                  augmentation_options=copy.deepcopy(recipe.get("augmentation_options", {})),
                  synthetic_pair=copy.deepcopy(recipe.get("synthetic_pair", {"enabled": False})),
                  retention_weight=settings.get("retention_weight", 0.), retention_margin=0.,
                  retention_checkpoint=settings["v6_checkpoint"])
    stages = []
    for number, item in enumerate(settings["stages"], 1):
        stage = {
            "number": number, "name": item["name"], "min_epochs": item["epochs"],
            "max_epochs": item["epochs"], "trainable_prefixes": item["trainable_prefixes"],
            "foundation_blocks": 0, "bridge_detach_backbone": True,
            "correspondence_weight": item.get("correspondence_weight", 0.),
        }
        for name in ("decoder", "handoff", "bridge", "fpn", "matching", "foundation", "center", "prior"):
            stage[name + "_lr"] = item.get(name + "_lr", 0.)
        stages.append(stage)
    config["stages"] = stages
    config["evaluation"]["test_after_training"] = False
    config["protocol_notes"] = [
        "独立V7实验，公共和CARLA分别从各自V6 best初始化，旧V6不覆盖。",
        "保留公共全域TRAIN覆盖与原重放倍率；新模块only阶段不更新已有V6参数。",
        "TRAIN顺序为原数据增强一次→共同像素→可选几何再生成→灰度/真实子集。",
        "再生成只用TRAIN标签，VAL/推理不生成，不增加传感器或GT输入。",
        "本配置不能保证旧域无退化；固定完整VAL选best，末步final另存，不按TEST选优。",
        "本入口是否已通过真实DDP冒烟，应查实际运行报告，不由配置存在推出。",
    ]
    return config


def main(recipe):
    output = Path(recipe["output"])
    if not output.resolve().is_relative_to(Path("/data")):
        raise ValueError("完整训练产物必须位于数据盘")
    directory = Path(recipe["config_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    for phase in ("public", "carla"):
        config = make_config(recipe, phase)
        filename = directory / ("train_" + phase + ".json")
        if filename.exists():
            raise FileExistsError(filename)
        if recipe.get("prepare_initial_weights", True):
            convert_v6_checkpoint(recipe["phases"][phase]["v6_checkpoint"],
                                  config["initial_model_checkpoint"], recipe["prior_kind"])
        filename.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    main(json.loads(Path(parser.parse_args().recipe).read_text()))
