"""独立 Evidence-Flow 八卡训练；复用已验证的数据、存储和连续阶段流程。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.runner import run
from research.v2_restart_20261029.training import RestartHooks


def flow_optimizer(model, stage, config):
    """新速度场和成熟主干分别设置学习率，不更改原层结构。"""
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("decoder.flow_field."):
            kind = "flow"
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


class FlowHooks(RestartHooks):
    optimizer_factory = staticmethod(flow_optimizer)

    def model_factory(self, config, initialize=True):
        from research.ruler_flow_v2_20261031.model import EvidenceFlowModel

        self.model = EvidenceFlowModel(config, initialize)
        return self.model

    def objective(self, prediction, inputs, targets, domain, settings, correspondence_weight=0.):
        from research.ruler_flow_v2_20261031.losses import flow_objective

        terms = super().objective(prediction, inputs, targets, domain, settings, correspondence_weight)
        weight = float(self.model.active_stage.get("flow_weight", 1.))
        auxiliary = flow_objective(self.model, prediction, inputs, targets, domain, self.config["flow"])
        terms["native_total"] = terms["total"]
        terms["total"] = terms["total"] + weight * auxiliary["total"]
        terms.update({"flow_" + key: value for key, value in auxiliary.items()})
        return terms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["architecture"] != "ruler_flow_v2" or config["train"]["stage_policy"] != "continuous_new":
        raise ValueError("独立入口只接受 ruler_flow_v2 连续训练配置")
    if config.get("teacher_checkpoint"):
        raise ValueError("本轮不使用 baseline 或旧模型教师")
    run(config, args.resume or None, hooks=FlowHooks(config))


if __name__ == "__main__":
    main()
