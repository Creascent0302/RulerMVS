"""V6独立全量训练：复用原DDP/恢复流程，仅注册模型、目标和验证协议。

连续阶段使用上阶段末步参数，但重新创建AdamW及阶段学习率计划。
last.pt保存恢复状态；best.pt按本轮完整VAL选优；final.pt始终保存末步。
"""

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch import nn

from jar.pretraining.comparison import equal_pixel_inputs
from jar.pretraining.runner import run
from jar.runtime import capture_rng, restore_rng
from research.v2_restart_20261029.training import RestartHooks
from research.rulermvs_v6_20261101.model import V6Model
from research.rulermvs_v6_20261101.objective import objective


def regional_selection_risk(rows, domains, weights):
    """域内按非空距离段归一，域间等权；只用于选优，不改变训练参数。"""
    if set(weights) - {"full", "near", "mid", "far"}:
        raise ValueError("选优分区仅支持full/near/mid/far")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("选优分区权重须为有限非负值")
    domain_scores = []
    for row, domain in zip(rows, domains):
        scale = domain.get("selection_depth_scale", 10. * domain["depth_scale"])
        weighted_sum, active_weight = 0., 0.
        for name, weight in weights.items():
            region = row[name]
            if weight == 0 or not region["pixels"]:
                continue
            risk = (math.log1p(region["mae"] / scale)
                    + math.log1p(region["rmse"] / scale)
                    + math.log1p(region["abs_rel"])) / 3.
            weighted_sum += weight * risk
            active_weight += weight
        if active_weight <= 0:
            raise ValueError(f"{domain.get('name', 'dataset')}没有正权重且有效的选优分区")
        domain_scores.append(weighted_sum / active_weight)
    return sum(domain_scores) / len(domain_scores)


def optimizer_for(model, stage, config):
    """六类参数独立学习率；新门控不会误用成熟解码器的学习率。"""
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("decoder.handoff."):
            kind = "handoff"
        elif name.startswith("bridge."):
            kind = "bridge"
        elif name.startswith("foundation."):
            kind = "foundation"
        elif name.startswith("matching.backbone."):
            kind = "matching"
        elif name.startswith("matching."):
            kind = "fpn"
        else:
            kind = "decoder"
        lr = float(stage[kind + "_lr"])
        if lr <= 0:
            raise ValueError(f"已解冻的{kind}须配置正学习率；冻结请修改trainable_prefixes")
        decay = 0. if parameter.ndim <= 1 or name.endswith("bias") else config["train"]["weight_decay"]
        groups.setdefault((kind, lr, decay), []).append(parameter)
    if not groups:
        raise ValueError("本阶段没有可训练参数")
    return torch.optim.AdamW([
        {"params": values, "lr": lr, "initial_lr": lr,
         "weight_decay": decay, "group_name": kind}
        for (kind, lr, decay), values in groups.items()
    ])


@contextmanager
def single_model_validation(model, runtime):
    """验证共同使用rank0的缓冲量，结束后恢复各rank状态和训练随机流。

    现有匹配骨干BN一直处于eval；仍显式同步，避免未来模块变化后
    再出现各rank用不同BN的混合模型验证。动态深度范围缓冲量也恢复。
    """
    rng = capture_rng(runtime)
    modes = {module: module.training for module in model.modules()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    try:
        model.eval()
        if runtime.distributed:
            for buffer in model.buffers():
                dist.broadcast(buffer, src=0)
        yield
    finally:
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                if name in buffers:
                    buffer.copy_(buffers[name])
        for module, mode in modes.items():
            module.training = mode
        restore_rng(rng, runtime)


class EqualPixelEvaluation(nn.Module):
    """仅转换输入图像信息；K、外参、掩码、标签均不修改。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, domain, reference=0):
        return self.model(equal_pixel_inputs(inputs), domain, reference)


class V6Hooks(RestartHooks):
    optimizer_factory = staticmethod(optimizer_for)

    def __init__(self, config):
        super().__init__(config)
        self.input_mode = config["input_mode"]
        if self.input_mode not in ("native", "equal_pixels"):
            raise ValueError("input_mode须明确为native或equal_pixels")

    def model_factory(self, config, initialize=True):
        self.model = V6Model(config, initialize)
        return self.model

    def prepare_batch(self, batch, domain):
        if self.input_mode == "equal_pixels":
            batch = dict(batch, inputs=equal_pixel_inputs(batch["inputs"]))
        return batch, domain

    def objective(self, prediction, inputs, targets, domain, settings, correspondence_weight=0.):
        return objective(prediction, inputs, targets, domain, settings, grouped=True,
                         correspondence_weight=correspondence_weight)

    def evaluate(self, model, dataset, config, runtime, limit=None, split="val"):
        evaluated = EqualPixelEvaluation(model) if self.input_mode == "equal_pixels" else model
        with single_model_validation(model, runtime):
            result = super().evaluate(evaluated, dataset, config, runtime, limit, split)
        weights = config["evaluation"].get("selection_region_weights", {"full": 1.})
        result["selection_risk_full_only"] = result["selection_risk"]
        result["selection_risk"] = regional_selection_risk(result["domains"], dataset.domains, weights)
        result["selection_region_weights"] = dict(weights)
        result["selection_rule"] = (
            "各域等权；域内非空full/near/mid/far按配置权重重新归一；"
            "段内log1p(MAE/s)、log1p(RMSE/s)、log1p(AbsRel)等权，"
            "s沿用TRAIN尺度。仅选择本轮best，不回滚或保证各指标同时改善。")
        result.update(input_mode=self.input_mode, validation_model="rank0_parameters_and_buffers",
                      training_state_restored=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["architecture"] != "rulermvs_v6" or config["train"]["stage_policy"] != "continuous_new":
        raise ValueError("独立入口只接受rulermvs_v6连续训练配置")
    if config.get("teacher_checkpoint"):
        raise ValueError("V6本轮不使用baseline或其他模型教师")
    run(config, args.resume or None, hooks=V6Hooks(config))


if __name__ == "__main__":
    main()
