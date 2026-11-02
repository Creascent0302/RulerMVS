"""同V5起点的候选交接消融；独立工厂，不改冻结主模型或训练循环。"""

import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from jar.pretraining.comparison import digest, write_json
from research.rulermvs_v6_20261101 import pilot
from research.rulermvs_v6_20261101.model import V6Model, build_from_checkpoint
from research.paper_ablation_20261102.handoff_evaluate import evaluate


VARIANTS = {
    "v5": "V5原点读出与点目标，同预算继续训练",
    "marginal": "仅细候选族质量边缘化、族内均值、族目标",
    "constant": "族读出加一个可学习空间常数门、族目标",
    "parent": "族读出加129参数质量自适应门、族目标",
    "parent_oldpoint": "质量自适应门，保留V5点目标以分离目标贡献",
}


class ConstantQualityGate(nn.Module):
    """单一可学习系数；与自适应门同初值sigmoid(-2)，不读取图像质量。"""

    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, parent, fine, source_null, source_valid):
        return self.logit.float().sigmoid().expand_as(parent[:, :1])


def build_ablation(saved, variant):
    if variant not in VARIANTS:
        raise ValueError(f"未知交接消融: {variant}")
    if variant != "constant":
        return build_from_checkpoint(saved, "parent" if variant == "parent_oldpoint" else variant)
    architecture = "rulermvs_v6_constant_ablation"
    if saved["architecture"] == architecture:
        config = copy.deepcopy(saved["config"])
        model = V6Model(config, initialize=False)
        model.decoder.handoff.gate = ConstantQualityGate()
        model.load_state_dict(saved["model"], strict=True)
    else:
        if saved["architecture"] != "rulermvs_v5":
            raise ValueError("常数门实验必须从共同V5初始化或恢复自己的checkpoint")
        model, config = build_from_checkpoint(saved, "parent")
        model.decoder.handoff.gate = ConstantQualityGate()
    model.architecture = architecture
    config["architecture"] = architecture
    config["handoff"]["gate_type"] = "learned_spatial_constant"
    model.config = config
    return model, config


def run(settings):
    settings = copy.deepcopy(settings)
    variant = settings["variant"]
    expected_grouped = variant not in ("v5", "parent_oldpoint")
    if settings["grouped_loss"] != expected_grouped:
        raise ValueError("预登记读出/目标组合不一致")
    output = Path(settings["output"])
    if not output.resolve().is_relative_to(Path("/data")):
        raise ValueError("权重与日志必须保存到数据盘")
    sources = [Path(__file__), Path(__file__).with_name("handoff_evaluate.py"),
               Path(pilot.__file__), ROOT / "research/rulermvs_v6_20261101/model.py",
               ROOT / "research/rulermvs_v6_20261101/handoff.py",
               ROOT / "research/rulermvs_v6_20261101/objective.py"]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    with patch.object(pilot, "build_from_checkpoint", build_ablation), patch.object(pilot, "evaluate", evaluate):
        pilot.run(settings)
    if hashes != {str(path.relative_to(ROOT)): digest(path) for path in sources}:
        raise RuntimeError("运行中源码改变，保留结果但不标记同版本完成")
    write_json(output / "handoff_audit.json", {
        "completed": True, "variant": variant, "description": VARIANTS[variant],
        "source_sha256": hashes, "checkpoint_sha256": digest(settings["checkpoint"]),
        "test_accessed": False, "selection": "fixed_last_step",
        "causal_scope": "共同V5权重起点、冻结编码器、相同适配预算；不是从零训练",
        "objective_difference": "v5与parent_oldpoint用原点目标，其余三组用相同族目标",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
