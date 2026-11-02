"""独立25%短基线TRAIN数据对照；不修改正在运行的训练/模型文件。"""

import argparse
from collections import Counter
import copy
from functools import partial
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import get_worker_info

from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from research.v7_domain_20261102 import train, train_round2
from research.v7_domain_20261102.synthetic_pair_bilinear import augment_training_batch


def augmentation_seed(seed, domain_index, index, epoch):
    """不消耗DataLoader/模型随机流，也不依赖worker调度顺序。"""
    return int(seed) + 70000019 + int(epoch) * 1000003 + int(index) * 137 + int(domain_index) * 10007


class SyntheticDataset(MixedDataset):
    def __init__(self, config, split="train", *, options=None, audit_folder=None):
        super().__init__(config, split)
        self.synthetic_options = dict(options or {})
        self.audit_folder = Path(audit_folder) if audit_folder is not None else None

    def __getitem__(self, key):
        sample = super().__getitem__(key)
        domain_index, index, epoch = key
        if self.split != "train" or epoch < 0:
            return sample
        seed = augmentation_seed(self.config["seed"], domain_index, index, epoch)
        generator = torch.Generator().manual_seed(seed)
        # collate保留CARLA每样本camera_ids顺序；公共域张量也使用同一实现。
        batch = collate_mixed([sample])
        augmented, targets, record = augment_training_batch(
            batch["inputs"], batch["targets"], self.domains[domain_index],
            generator=generator, **self.synthetic_options)
        if not torch.equal(targets["depth"][:, 0], batch["targets"]["depth"][:, 0]):
            raise RuntimeError("再生成修改了参考GT")
        result = dict(sample)
        result["inputs"] = {name: value[0] for name, value in augmented.items()}
        result["targets"] = {name: value[0] for name, value in targets.items()}
        if self.audit_folder is not None:
            worker = get_worker_info()
            worker_id = worker.id if worker is not None else "main"
            self.audit_folder.mkdir(parents=True, exist_ok=True)
            requested = bool(torch.rand(1, 3, generator=torch.Generator().manual_seed(seed))[0, 0]
                             < self.synthetic_options.get("probability", .25))
            row = {
                "domain_index": int(domain_index), "dataset": self.domains[domain_index]["name"],
                "index": int(index), "epoch": int(epoch), "seed": seed,
                "requested": requested, "synthetic": bool(record["active"][0]),
                "source_coverage": float(record["source_coverage"][0]),
                "baseline_data_units": float(record["baseline_data_units"][0]),
                "baseline_normalized": float(record["baseline_normalized"][0]),
                "retained_views_before_outer_augmentation": int(record["retained_views"][0]),
                "reference_GT_unchanged": True,
            }
            # 每个worker单独文件，避免多进程行交织；默认flush/close即时落盘。
            with (self.audit_folder / f"worker_{worker_id}.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return result


def exposure_audit(output, settings):
    plan = json.loads((output / "plan.json").read_text())
    expected = Counter(tuple(key) for batch in plan["schedule"]["batches"] for key in batch)
    rows = [json.loads(line) for path in sorted((output / "synthetic_exposures").glob("*.jsonl"))
            for line in path.read_text().splitlines() if line]
    observed = Counter((row["domain_index"], row["index"], row["epoch"]) for row in rows)
    if observed != expected:
        raise RuntimeError("实际数据读取曝光与预登记训练schedule不一致")
    probability = settings.get("synthetic_options", {}).get("probability", .25)
    for row in rows:
        seed = augmentation_seed(settings["seed"], row["domain_index"], row["index"], row["epoch"])
        requested = bool(torch.rand(1, 3, generator=torch.Generator().manual_seed(seed))[0, 0] < probability)
        if row["requested"] != requested or (row["synthetic"] and not requested):
            raise RuntimeError("合成抽签不满足独立预登记随机流")
    domains = {}
    for row in rows:
        item = domains.setdefault(row["dataset"], {"exposures": 0, "requested": 0,
                                                    "synthetic": 0, "original": 0,
                                                    "coverage_sum_synthetic": 0.})
        item["exposures"] += 1
        item["requested"] += int(row["requested"])
        item["synthetic"] += int(row["synthetic"])
        item["original"] += int(not row["synthetic"])
        if row["synthetic"]:
            item["coverage_sum_synthetic"] += row["source_coverage"]
    for item in domains.values():
        item["mean_source_coverage_synthetic"] = (
            item["coverage_sum_synthetic"] / item["synthetic"] if item["synthetic"] else None)
    return {"schedule_matches": True, "exposures": len(rows),
            "synthetic": sum(row["synthetic"] for row in rows),
            "original": sum(not row["synthetic"] for row in rows), "datasets": domains}


def run(settings):
    settings = copy.deepcopy(settings)
    if settings["variant"] != "coordinate":
        raise ValueError("本数据对照仅采用同一coordinate模型")
    options = settings.setdefault("synthetic_options", {
        "probability": .25, "baseline_norm": [.03, .3],
        "keep_other_sources": False, "surface_relative_tolerance": .01,
        "minimum_weight": .5, "centroid_tolerance": .25})
    output = Path(settings["output"])
    sources = [Path(__file__), Path(__file__).with_name("synthetic_pair_bilinear.py"),
               Path(__file__).with_name("synthetic_pair.py"),
               Path(__file__).with_name("test_synthetic_pair_bilinear.py")]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    constructor = partial(SyntheticDataset, options=options, audit_folder=output / "synthetic_exposures")
    with patch.object(train, "MixedDataset", constructor):
        train_round2.run(settings)
    completed = json.loads((output / "completed.json").read_text())
    if not all(completed["checkpoint_checks"].values()):
        raise RuntimeError("checkpoint或推理标签隔离检查失败")
    if hashes != {str(path.relative_to(ROOT)): digest(path) for path in sources}:
        raise RuntimeError("再生成源码运行中改变")
    for path in sources:
        (output / "source_snapshot" / path.name).write_bytes(path.read_bytes())
    audit = exposure_audit(output, settings)
    write_json(output / "synthetic_audit.json", {
        "completed": True, "test_accessed": False, "source_sha256": hashes,
        "options": options, "exposure": audit,
        "dataset_override": "只在本进程替换train.MixedDataset；VAL和epoch<0完全绕过",
        "scope": "相对第二轮同配置仅新增25%几何再生成；同时改变源数/覆盖，非纯基线因果隔离",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
