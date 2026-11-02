"""独立部署路径完整TEST：复用冻结GT/指标，省略KD表面，不进行计时或选权重。"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import add_statistics, digest, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move, seed_all
from research.rulermvs_v6_20261101.evaluation import Models
from research.rulermvs_v6_20261101.speed_final import initialize, load
from research.near_mid_evaluation_20261028.evaluate import predict_with_stages
from research.near_mid_evaluation_20261028.metrics import depth_metrics, depth_sums, ground_truth, regions
from tools.smoke_outdoor_datasets import fingerprint


def path(value):
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def read(filename):
    return json.loads(path(filename).read_text())


def immutable(filename, value):
    if filename.exists() and read(filename) != value:
        raise ValueError("冻结产物不同，拒绝覆盖：" + str(filename))
    if not filename.exists():
        write_json(filename, value)


class FastModels(Models):
    def get(self, method, domain):
        settings = self.config["methods"][method]
        if method != "v6_fast" or settings["kind"] != "own":
            raise ValueError("此独立入口只测V6快速路径")
        key = settings["weights"]
        if key not in self.loaded:
            self.loaded[key] = load({"v6_checkpoint": key, "gpu": self.device.index,
                                     "fast": self.config["fast"]}, "v6_fast")
        return self.loaded[key]


def prepare(settings):
    main = read(settings["evaluation_plan"])
    timing = read(settings["timing_config"])
    output = path(settings["output"])
    groups, source_hashes = {}, {}
    for phase in ("public", "carla_new"):
        source = main["groups"][phase + "_main"]
        original = read(source["config"])
        eager_audit = read(path(source["output"]) / "full_test/audit.json")
        if not read(path(source["output"]) / "full_test/summary.json")["completed"]:
            raise ValueError("需要已完成的eager完整TEST对照")
        source_hashes.update(eager_audit["source_sha256"])
        config = copy.deepcopy(original)
        config["methods"] = {"v6_fast": copy.deepcopy(original["methods"]["v6_best"])}
        config.update(output=str(output / phase), split="test", gpu=settings["gpu"],
                      loader_workers=settings["loader_workers"], shard_size=settings["shard_size"],
                      fast=copy.deepcopy(timing["fast"]), surface_evaluation=False)
        # 缓存目录及编译开关与独占测速完全一致，保持同一部署实现。
        if phase == "carla_new" and config["methods"]["v6_fast"]["weights"] != timing["v6_checkpoint"]:
            raise ValueError("完整TEST与测速CARLA权重不一致")
        config_file = output / "configs" / (phase + ".json")
        immutable(config_file, config)
        groups[phase] = {"config": str(config_file), "eager_folder": str(path(source["output"]) / "full_test"),
                         "config_sha256": digest(config_file), "frames": source["frames"],
                         "weights_sha256": digest(config["methods"]["v6_fast"]["weights"]),
                         "manifests_sha256": eager_audit["manifest_sha256"]}
    source_hashes.update(main["extension_source_sha256"])
    for filename in (Path(__file__), ROOT / "research/rulermvs_v6_20261101/inference.py",
                     ROOT / "research/rulermvs_v6_20261101/speed_final.py"):
        source_hashes[str(filename.relative_to(ROOT))] = digest(filename)
    plan = {"settings": settings, "groups": groups, "source_sha256": source_hashes,
            "fast": timing["fast"], "selection": "same_frozen_VAL_best_as_main_TEST",
            "timing_config_sha256": digest(path(settings["timing_config"])),
            "scope": "完整TEST深度及共同stride8阶段，GT掩码不变；不计速度，不计算或伪造表面指标。"}
    verify(plan)
    immutable(output / "plan.json", plan)
    return plan


def verify(plan):
    for filename, expected in plan["source_sha256"].items():
        if digest(path(filename)) != expected:
            raise ValueError("源码发生改变：" + filename)
    for item in plan["groups"].values():
        config = read(item["config"])
        if digest(path(item["config"])) != item["config_sha256"]:
            raise ValueError("部署测试配置改变")
        if digest(config["methods"]["v6_fast"]["weights"]) != item["weights_sha256"]:
            raise ValueError("部署测试权重改变")
        for domain in config["datasets"]:
            if digest(domain["manifests"]["test"]) != item["manifests_sha256"][domain["name"]]:
                raise ValueError("部署测试清单改变")


def run_group(plan, phase):
    item = plan["groups"][phase]
    config = read(item["config"])
    initialize(config)
    seed_all(config["seed"])
    device = torch.device("cuda", config["gpu"])
    output = path(config["output"]) / "full_test"
    audit = {"config": config, "source_sha256": plan["source_sha256"],
             "weights_sha256": {"v6_fast": item["weights_sha256"]},
             "manifest_sha256": item["manifests_sha256"]}
    protocol = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    immutable(output / "audit.json", audit)
    eager = read(path(item["eager_folder"]) / "summary.json")
    models = FastModels(config, device)
    summary = {"completed": True, "protocol": protocol, "split": "test", "surface_computed": False,
               "selection": plan["selection"], "datasets": {}}
    for domain in config["datasets"]:
        dataset = build_dataset(dataset_config(domain, config), "test", load_targets=True)
        if len(dataset) != item["frames"][domain["name"]]:
            raise ValueError("完整TEST样本数不符")
        merged = {"indices": [], "depth": {}, "stages": {}}
        for start in range(0, len(dataset), config["shard_size"]):
            indices = list(range(start, min(start + config["shard_size"], len(dataset))))
            filename = output / domain["name"] / f"part_{start // config['shard_size']:04d}.json"
            if filename.exists():
                record = read(filename)
                if not record.get("completed") or record["protocol"] != protocol:
                    raise ValueError("已有快速TEST分片不完整或协议不同，保留现场")
            else:
                loader = DataLoader(dataset, batch_size=1, sampler=indices, num_workers=config["loader_workers"],
                                    pin_memory=True, collate_fn=collate_frames)
                frames, totals = [], {"indices": [], "depth": {}, "stages": {}}
                for index, batch in zip(indices, loader):
                    inputs = move(batch["inputs"], device)
                    with torch.inference_mode():
                        depth, valid, stages = predict_with_stages(models, "v6_fast", config["methods"]["v6_fast"], domain, inputs)
                    truth, mask = ground_truth(batch, domain)
                    masks = regions(truth, mask, domain)
                    metrics = {region: depth_sums(depth, valid, truth, selected) for region, selected in masks.items()}
                    stages["final"] = depth[..., ::8, ::8]
                    stage_metrics = {}
                    for name, value in stages.items():
                        target, selected = truth[..., ::8, ::8], mask[..., ::8, ::8]
                        if value.shape != target.shape:
                            raise ValueError("部署阶段格点不同")
                        stage_metrics[name] = depth_sums(value, torch.isfinite(value) & (value > 0), target, selected)
                    totals["indices"].append(index)
                    for region, values in metrics.items():
                        add_statistics(totals["depth"].setdefault(region, {}), values)
                    for name, values in stage_metrics.items():
                        add_statistics(totals["stages"].setdefault(name, {}), values)
                    frame = {"index": index, "valid_views": int(inputs["view_mask"].sum()),
                             "methods": {"v6_fast": {"depth": metrics, "stages": stage_metrics}}}
                    if index in (0, len(dataset) // 2, len(dataset) - 1):
                        frame["input_sha256"] = fingerprint(batch["inputs"])
                    frames.append(frame)
                record = {"completed": True, "protocol": protocol, "totals": totals, "frames": frames}
                write_json(filename, record)
            if record["totals"]["indices"] != indices:
                raise ValueError("部署分片样本次序不符")
            merged["indices"].extend(indices)
            for kind in ("depth", "stages"):
                for name, values in record["totals"][kind].items():
                    add_statistics(merged[kind].setdefault(name, {}), values)
            print(json.dumps({"phase": phase, "dataset": domain["name"], "done": len(merged["indices"]), "total": len(dataset)}), flush=True)
        values = {"frames": len(merged["indices"]), "depth": {key: depth_metrics(value) for key, value in merged["depth"].items()},
                  "stages_stride8": {key: depth_metrics(value) for key, value in merged["stages"].items()}}
        reference = eager["datasets"][domain["name"]]["methods"]["v6_best"]
        for region, metrics in values["depth"].items():
            if metrics["gt_pixels"] != reference["depth"][region]["gt_pixels"]:
                raise ValueError("部署TEST和原完整TEST的GT分母不相同")
        summary["datasets"][domain["name"]] = {"units": domain["geometry_units"], "num_views": domain["num_views"],
                                                "methods": {"v6_fast": values}}
    verify(plan)
    write_json(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_suffix(".json")))
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    plan = prepare(read(args.config))
    if args.prepare:
        print(json.dumps({"prepared": True, "frames": {phase: sum(value["frames"].values()) for phase, value in plan["groups"].items()}}))
        return
    for phase in plan["groups"]:
        run_group(plan, phase)
    write_json(path(plan["settings"]["output"]) / "completed.json", {"completed": True, "test_selected_model": False,
               "same_fast_path_as_timing": True, "surface_computed": False, "test_references": sum(sum(value["frames"].values()) for value in plan["groups"].values())})


if __name__ == "__main__":
    main()
