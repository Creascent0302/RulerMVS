"""RulerMVS-V5：独立 CNN→ViT 模块，复用 V2 连续阶段训练流程。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.runner import run
from research.v2_restart_20261029.training import RestartHooks


def bridge_optimizer(model, stage, config):
    """新增连接、原编码器和原解码器分别使用明确的学习率。"""
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("bridge."):
            kind = "bridge"
        elif name.startswith("foundation."):
            kind = "foundation"
        elif name.startswith("matching.backbone."):
            kind = "matching"
        elif name.startswith("matching."):
            kind = "fpn"
        else:
            kind = "decoder"
        decay = 0. if parameter.ndim <= 1 or name.endswith("bias") else config["train"]["weight_decay"]
        groups.setdefault((kind, decay), []).append(parameter)
    return torch.optim.AdamW([
        {"params": values, "lr": stage[kind + "_lr"],
         "initial_lr": stage[kind + "_lr"], "weight_decay": decay, "group_name": kind}
        for (kind, decay), values in groups.items()
    ])


class V5Hooks(RestartHooks):
    optimizer_factory = staticmethod(bridge_optimizer)

    def model_factory(self, config, initialize=True):
        from research.rulermvs_v5_20261031.model import V5Model

        self.model = V5Model(config, initialize)
        return self.model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["architecture"] != "rulermvs_v5" or config["train"]["stage_policy"] != "continuous_new":
        raise ValueError("独立入口只接受 rulermvs_v5 连续训练配置")
    if config.get("teacher_checkpoint"):
        raise ValueError("本轮不使用 baseline 或旧模型教师")
    run(config, args.resume or None, hooks=V5Hooks(config))


if __name__ == "__main__":
    main()
