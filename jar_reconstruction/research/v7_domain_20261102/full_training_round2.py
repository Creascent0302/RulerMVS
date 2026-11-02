"""固定/连续坐标先验的独立全量入口；未验证的DDP运行不由代码存在推出。"""

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
from research.v7_domain_20261102.full_training import optimizer_for
from research.v7_domain_20261102.model import V7Model
from research.v7_domain_20261102.model_round2 import V7CoordinateModel, build_model
from research.v7_domain_20261102.retention_round2 import retention_penalty
from research.v7_domain_20261102.synthetic_pair_bilinear import augment_training_batch


def model_factory(config, initialize):
    classes = {"fixed": V7Model, "coordinate": V7CoordinateModel}
    kind = config["prior_kind"]
    if kind not in classes or config["architecture"] != classes[kind].architecture:
        raise ValueError("先验类型与真实architecture不一致")
    if config["v7"] != {"projective_center": False, "context_prior": True}:
        raise ValueError("本入口仅含一种粗先验，不静默混入投影中心")
    return classes[kind](config, initialize=initialize)


def convert_v6_checkpoint(source, destination, kind, minimum_free_gib=200):
    """公共/CARLA分别从旧V6转换，新增结构不继承本轮短训结果。"""
    if kind not in ("fixed", "coordinate"):
        raise ValueError("kind仅支持fixed或coordinate")
    source, destination = Path(source), Path(destination)
    if destination.exists() or source.resolve() == destination.resolve():
        raise FileExistsError("禁止覆盖已有权重，请使用独立的新初始化路径")
    if not destination.resolve().is_relative_to(Path("/data")):
        raise ValueError("初始化权重必须保存到数据盘")
    saved = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    if saved["architecture"] != "rulermvs_v6":
        raise ValueError("本轮全训须分别从对应V6开始，不暗用短训或baseline权重")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261102)
        model, config = build_model(saved, kind)
    config["prior_kind"] = kind
    payload = {"architecture": model.architecture, "model": model.state_dict(), "config": config,
               "source_checkpoint": str(source.resolve()), "source_sha256": digest(source),
               "conversion_variant": kind, "trained_during_conversion": False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(payload, destination, minimum_free_gib)


class V7Round2Hooks(V6Hooks):
    optimizer_factory = staticmethod(optimizer_for)

    def __init__(self, config):
        super().__init__(config)
        self.teacher = None
        self.batch_statistics = {}

    def model_factory(self, config, initialize=True):
        self.model = model_factory(config, initialize)
        if config.get("retention_weight", 0.) < 0:
            raise ValueError("保持损失权重不能为负")
        if config.get("retention_weight", 0.):
            saved = torch.load(config["retention_checkpoint"], map_location="cpu",
                               weights_only=False, mmap=True)
            with torch.random.fork_rng(devices=[]):
                self.teacher, _ = build_model(saved, "control")
            self.teacher.requires_grad_(False).eval()
        return self.model

    def prepare_batch(self, batch, domain):
        # 原MixedDataset只执行一次原亮度/丢一源；此处不重复调用它。
        batch, domain = super().prepare_batch(batch, domain)
        inputs, targets = batch["inputs"], batch["targets"]
        self.batch_statistics = {}
        synthetic = dict(self.config.get("synthetic_pair", {}))
        if synthetic.pop("enabled", False):
            inputs, targets, statistics = augment_training_batch(
                inputs, targets, domain, generator=torch.default_generator, **synthetic)
            active = statistics["active"].float()
            self.batch_statistics["synthetic_fraction"] = active.mean()
            self.batch_statistics["synthetic_source_coverage"] = (
                statistics["source_coverage"].float() * active).sum() / active.sum().clamp_min(1)
        if self.config.get("domain_augmentation", False):
            inputs, statistics = augment_inputs(inputs, **self.config.get("augmentation_options", {}))
            self.batch_statistics["grayscale_fraction"] = statistics["grayscale"].float().mean()
        self.batch_statistics["retained_views"] = inputs["view_mask"].sum(1).float().mean()
        return dict(batch, inputs=inputs, targets=targets), domain

    def objective(self, prediction, inputs, targets, domain, settings, correspondence_weight=0.):
        terms = super().objective(prediction, inputs, targets, domain, settings, correspondence_weight)
        if self.teacher is not None:
            device = inputs["rgb"].device
            if next(self.teacher.parameters()).device != device:
                self.teacher.to(device)
            with torch.no_grad():
                old = self.teacher(inputs, domain)
            penalty = retention_penalty(prediction, old, inputs, targets, domain,
                                        self.config.get("retention_margin", 0.))
            terms["retention"] = penalty
            terms["total"] = terms["total"] + self.config["retention_weight"] * penalty
        terms.update(self.batch_statistics)
        return terms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["train"]["stage_policy"] != "continuous_new":
        raise ValueError("本入口采用独立连续新训练，不回写或回滚V6")
    if config.get("teacher_checkpoint"):
        raise ValueError("不使用baseline教师，保持项只接受明确的旧V6权重")
    run(config, args.resume or None, hooks=V7Round2Hooks(config))
