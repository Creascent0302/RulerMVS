"""V2 独立重训：通用初始化、连续训练、本次运行内选优，不使用旧模型教师。"""

import argparse
import functools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from jar.pretraining.data import collate_mixed
from jar.pretraining.losses import depth_objective
from jar.pretraining.runner import run
from jar.runtime import move, precision_context
from research.near_mid_20261027.training import distance_masks
from research.v2_training_20261028.objectives import fine_actual_regret
from research.v2_training_20261028.sampling import RebalancedFullPassSampler
from research.v2_training_20261028.training import V2TrainingModel, v2_optimizer


def selection_risk(rows, domains):
    """三个指标等权、各域等权；尺度来自 TRAIN 统计或预登记常量。

    log1p 保持每个指标的单调性，缓解极端域压倒其他域；不保证每个域都改善。
    CARLA 四参考各算一域，避免某个相机有效像素较多而主导选模。
    """
    values = []
    for row, domain in zip(rows, domains):
        scale = domain.get("selection_depth_scale", 10. * domain["depth_scale"])
        full = row["full"]
        values.append((math.log1p(full["mae"] / scale)
                       + math.log1p(full["rmse"] / scale)
                       + math.log1p(full["abs_rel"])) / 3.)
    return sum(values) / len(values)


class RestartHooks:
    optimizer_factory = staticmethod(v2_optimizer)

    def __init__(self, config):
        self.config = config
        self.full_sampler = functools.partial(
            RebalancedFullPassSampler,
            domain_multipliers=[d.get("train_multiplier", 1.) for d in config["datasets"]],
            domain_schedule="balanced_prefix",
        )

    def model_factory(self, config, initialize=True):
        self.model = V2TrainingModel(config, initialize)
        return self.model

    def sampler_factory(self, *args, **kwargs):
        sampler = self.full_sampler(*args, **kwargs)
        limit = self.config["train"].get("development_updates_per_epoch")
        if limit:
            sampler.updates = sampler.updates[:limit]
        return sampler

    def objective(self, prediction, inputs, targets, domain, settings, correspondence_weight=0.):
        terms = depth_objective(prediction, inputs, targets, domain, settings, correspondence_weight)
        if self.model.active_stage.get("actual_fine_risk", False):
            risk = fine_actual_regret(prediction, inputs, targets, domain)
            terms["total"] = terms["total"] + .1 * (risk["new_risk"] - risk["old_risk"])
            terms["fine_actual_risk"] = risk["new_risk"]
        return terms

    @staticmethod
    def selection_score(result):
        return result["selection_risk"]

    @torch.no_grad()
    def evaluate(self, model, dataset, config, runtime, limit=None, split="val"):
        model.eval()
        rows = []
        for domain_index, domain in enumerate(dataset.domains):
            length = len(dataset.datasets[domain_index])
            indices = list(range(length)) if limit is None else torch.linspace(
                0, length - 1, min(length, limit)).long().tolist()
            keys = [(domain_index, index, -1) for index in indices[runtime.rank::runtime.world_size]]
            loader = DataLoader(dataset, batch_size=1, sampler=keys, collate_fn=collate_mixed,
                                num_workers=0, pin_memory=True)
            sums = torch.zeros(4, 6, dtype=torch.float64, device=runtime.device)
            frames = torch.zeros((), dtype=torch.long, device=runtime.device)
            for batch in loader:
                batch = move(batch, runtime.device)
                inputs, targets = batch["inputs"], batch["targets"]
                with precision_context(runtime.device, config["evaluation"].get("precision", "fp32")):
                    prediction = model(inputs, domain)
                estimate = prediction["depth"].float() * domain["depth_scale"]
                truth = targets["depth"][:, :1].float()
                valid = targets["depth_valid"][:, :1] & inputs["image_mask"][:, :1]
                valid &= inputs["view_mask"][:, :1, None, None] & torch.isfinite(truth) & (truth > 0)
                if domain.get("eval_range"):
                    low, high = domain["eval_range"]
                    valid &= (truth >= low) & (truth <= high)
                if not bool((torch.isfinite(estimate[valid]) & (estimate[valid] > 0)).all()):
                    raise RuntimeError(f"{domain['name']} 的验证预测包含非有限或非正深度")
                for index, mask in enumerate((valid, *distance_masks(truth, valid, domain))):
                    error = (estimate[mask].double() - truth[mask].double()).abs()
                    ratio = torch.maximum(estimate[mask] / truth[mask], truth[mask] / estimate[mask])
                    sums[index] += torch.stack((
                        error.new_tensor(error.numel()), error.sum(), error.square().sum(),
                        (error / truth[mask]).sum(), (ratio < 1.25).sum(), prediction["valid"][mask].sum(),
                    ))
                frames += 1
            if runtime.distributed:
                dist.all_reduce(sums)
                dist.all_reduce(frames)
            regions = {}
            for name, values in zip(("full", "near", "mid", "far"), sums.cpu().tolist()):
                count, absolute, square, relative, delta, coverage = values
                regions[name] = {"pixels": int(count)}
                if count:
                    regions[name].update(mae=absolute / count, rmse=math.sqrt(square / count),
                                         abs_rel=relative / count, delta1=delta / count,
                                         coverage=coverage / count)
            if not regions["full"]["pixels"]:
                raise RuntimeError(f"{domain['name']}/{split} 没有有效监督")
            rows.append({"dataset": domain["name"], "frames": int(frames),
                         "units": domain["geometry_units"], **regions})
        model.train()
        return {"split": split, "full": limit is None, "domains": rows,
                "selection_risk": selection_risk(rows, dataset.domains),
                "selection_rule": "各域等权平均 log1p(MAE/s)、log1p(RMSE/s)、log1p(AbsRel)，s为TRAIN中位数或预登记尺度"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config["train"]["stage_policy"] != "continuous_new":
        raise ValueError("本入口只允许本次新训练连续进行，不允许守门回退")
    if config.get("teacher_checkpoint"):
        raise ValueError("独立重训不加载旧 V2 或 baseline 教师")
    if config["phase"] == "public" and config.get("initial_model_checkpoint"):
        raise ValueError("公共阶段必须从通用初始化开始，不加载任何 V2 任务权重")
    run(config, args.resume or None, hooks=RestartHooks(config))


if __name__ == "__main__":
    main()
