"""V7 独立八卡入口：复用已验证DDP/恢复流程，保持V6默认入口不变。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest
from jar.pretraining.runner import atomic_save, run
from research.rulermvs_v6_20261101.training import V6Hooks
from research.v7_domain_20261102.augmentation import augment_inputs
from research.v7_domain_20261102.model import V7Model, build_model
from research.v7_domain_20261102.train import retention_penalty


def optimizer_for(model, stage, config):
    """保持原六类学习率，独立增加新中心与粗先验两类。"""
    groups = {}
    prefixes = (("decoder.center.", "center"), ("decoder.context_prior.", "prior"),
                ("decoder.handoff.", "handoff"), ("bridge.", "bridge"),
                ("foundation.", "foundation"), ("matching.backbone.", "matching"),
                ("matching.", "fpn"))
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        kind = next((kind for prefix, kind in prefixes if name.startswith(prefix)), "decoder")
        lr = float(stage[kind + "_lr"])
        if lr <= 0:
            raise ValueError(f"可训练{kind}需要正学习率；禁用请配置trainable_prefixes")
        decay = 0. if parameter.ndim <= 1 or name.endswith("bias") else config["train"]["weight_decay"]
        groups.setdefault((kind, lr, decay), []).append(parameter)
    if not groups:
        raise ValueError("当前阶段没有可训练参数")
    return torch.optim.AdamW([
        {"params": values, "lr": lr, "initial_lr": lr,
         "weight_decay": decay, "group_name": kind}
        for (kind, lr, decay), values in groups.items()
    ])


def prepare_initial_checkpoint(source, destination, variant, minimum_free_gib=200):
    """独立转换V6到选定V7；新模块恒等初始化，原文件绝不覆盖。

    已完成的V7短训权重也可转换为相同架构初始化，但应在配置中披露。
    这里只转换权重包装，训练分配与学习率由独立训练配置决定。
    """
    source, destination = Path(source), Path(destination)
    if variant not in ("projective", "prior", "combined"):
        raise ValueError("全量V7转换需明确projective/prior/combined结构")
    if destination.exists() or source.resolve() == destination.resolve():
        raise FileExistsError("初始化转换要求新的输出路径，不能覆盖旧权重")
    if not destination.resolve().is_relative_to(Path("/data")):
        raise ValueError("转换权重必须保存至数据盘")
    saved = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    expected_options = {"projective_center": variant in ("projective", "combined"),
                        "context_prior": variant in ("prior", "combined")}
    if saved["architecture"] == "rulermvs_v7" and saved["config"]["v7"] != expected_options:
        raise ValueError("已有V7权重的结构与转换variant不一致，不能用名称覆盖实际架构")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261102)
        model, config = build_model(saved, variant)
    if model.architecture != "rulermvs_v7":
        raise ValueError("全量V7初始化只接受已选定的新增结构；V6控制不伪装为新结构")
    payload = {"architecture": model.architecture, "model": model.state_dict(), "config": config,
               "source_checkpoint": str(source.resolve()), "source_sha256": digest(source),
               "conversion_variant": variant, "trained_during_conversion": False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(payload, destination, minimum_free_gib)
    return payload["source_sha256"]


class V7Hooks(V6Hooks):
    optimizer_factory = staticmethod(optimizer_for)

    def __init__(self, config):
        super().__init__(config)
        self.teacher = None
        self.augmentation_statistics = None

    def model_factory(self, config, initialize=True):
        self.model = V7Model(config, initialize)
        weight = float(config.get("retention_weight", 0.))
        if weight < 0:
            raise ValueError("retention_weight不能为负")
        if weight:
            filename = config.get("retention_checkpoint")
            if not filename:
                raise ValueError("启用质量保持需要明确指定本项目V6权重")
            saved = torch.load(filename, map_location="cpu", weights_only=False, mmap=True)
            if saved["architecture"] != "rulermvs_v6":
                raise ValueError("质量保持只接受旧V6，不加载baseline教师")
            with torch.random.fork_rng(devices=[]):
                self.teacher, _ = build_model(saved, "control")
            self.teacher.requires_grad_(False).eval()
        return self.model

    def prepare_batch(self, batch, domain):
        batch, domain = super().prepare_batch(batch, domain)
        if self.config.get("domain_augmentation", False):
            inputs, self.augmentation_statistics = augment_inputs(
                batch["inputs"], **self.config.get("augmentation_options", {}))
            batch = dict(batch, inputs=inputs)
        return batch, domain

    def objective(self, prediction, inputs, targets, domain, settings, correspondence_weight=0.):
        terms = super().objective(prediction, inputs, targets, domain, settings, correspondence_weight)
        if self.teacher is not None:
            device = inputs["rgb"].device
            if next(self.teacher.parameters()).device != device:
                self.teacher.to(device)
            with torch.no_grad():
                old = self.teacher(inputs, domain)
            penalty = retention_penalty(prediction, old, inputs, targets, domain,
                                        self.config.get("retention_margin", .01))
            terms["retention"] = penalty
            terms["total"] = terms["total"] + self.config["retention_weight"] * penalty
        if self.augmentation_statistics is not None:
            record = self.augmentation_statistics
            terms["grayscale_fraction"] = record["grayscale"].float().mean()
            terms["retained_views"] = record["retained_views"].float().mean()
        return terms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["architecture"] != "rulermvs_v7" or config["train"]["stage_policy"] != "continuous_new":
        raise ValueError("V7入口只接受独立rulermvs_v7连续新训练配置")
    if config.get("teacher_checkpoint"):
        raise ValueError("不使用baseline教师；质量保持另用retention_checkpoint明确旧V6来源")
    run(config, args.resume or None, hooks=V7Hooks(config))


if __name__ == "__main__":
    main()
