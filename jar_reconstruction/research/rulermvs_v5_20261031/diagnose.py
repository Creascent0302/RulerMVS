"""V2/V5同VAL格点的候选供给与选择诊断；GT只在前向结束后参与统计。"""

import argparse
import copy
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.runtime import move, seed_all
from research.near_evidence_20261030.pilot import path, prepare
from research.near_mid_evaluation_20261028.metrics import distance_limits, ground_truth
from research.rulermvs_v5_20261031.pilot import load_model
from tools.smoke_outdoor_datasets import offline_only


def candidate_values(prediction, truth, valid, domain):
    """误差单位还原为各域原单位；十二离散候选覆盖不等于区间覆盖。"""
    scale = float(prediction["unit_scale"])
    fine = prediction["fine"]
    coarse = prediction["coarse"]["depth"][:, 0] * scale
    actual = fine["depth"][:, 0, ::2, ::2] * scale
    eligible = fine["valid"][..., ::2, ::2]
    values = {"coarse_winner_MAE": (coarse - truth).abs(),
              "fine_actual_MAE": (actual - truth).abs(),
              "final_stride8_MAE": (prediction["depth"][:, 0, ::8, ::8] * scale - truth).abs()}
    for label, key in (("sampling", "sampling_depth"), ("continuous", "candidate_depth")):
        candidates = fine[key][..., ::2, ::2] * scale
        errors = (candidates - truth[:, None]).abs().masked_fill(~eligible, torch.inf)
        oracle = errors.amin(1)
        values[label + "_oracle_MAE"] = oracle
        values[label + "_coverage_rel_5pct"] = (oracle <= truth * .05).float()
        values[label + "_coverage_abs_0p1_scale"] = (oracle <= .1 * domain["depth_scale"]).float()
    values["fine_selection_regret"] = values["fine_actual_MAE"] - values["continuous_oracle_MAE"]
    if any(value.shape != truth.shape or not bool(torch.isfinite(value[valid]).all())
           for value in values.values()):
        raise RuntimeError("共同GT格点上预测或候选缺失/非有限，拒绝静默删除难例")
    return values


def accumulate(result, mask, values):
    result["pixels"] = result.get("pixels", 0) + int(mask.sum())
    sums = result.setdefault("sums", {})
    for name, value in values.items():
        sums[name] = sums.get(name, 0.) + float(value[mask].double().sum())


@torch.inference_mode()
def evaluate(model, dataset, plan, device):
    report = {}
    for number, domain in enumerate(dataset.domains):
        selected = plan[domain["name"]]
        near, far = distance_limits(domain)
        partitions = {name: {} for name in ("all", "development", "confirmation")}
        for index in selected["indices"]:
            batch = move(collate_mixed([dataset[(number, index, -1)]]), device)
            prediction = model(batch["inputs"], domain, reference=0)
            truth, valid = ground_truth(batch, dict(domain, reference_index=0))
            truth, valid = truth[:, 0, ::8, ::8], valid[:, 0, ::8, ::8]
            values = candidate_values(prediction, truth, valid, domain)
            masks = {"full": valid, "near": valid & (truth < near),
                     "mid": valid & (truth >= near) & (truth < far), "far": valid & (truth >= far)}
            partition = "development" if index in selected["development"] else "confirmation"
            for name in ("all", partition):
                for region, mask in masks.items():
                    accumulate(partitions[name].setdefault(region, {}), mask, values)
        for groups in partitions.values():
            for item in groups.values():
                item["means"] = {key: total / item["pixels"] if item["pixels"] else None
                                 for key, total in item["sums"].items()}
        report[domain["name"]] = {"units": domain["geometry_units"], "frames": len(selected["indices"]),
                                   "distance_limits": [near, far], "absolute_tolerance": .1 * domain["depth_scale"],
                                   "partitions": partitions}
    return report


def run(config):
    offline_only()
    seed_all(config.get("seed", 20261031))
    torch.set_num_threads(config.get("threads", 2))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config.get("device", "cuda:0"))
    destination = path(config["output"])
    if destination.exists():
        raise FileExistsError(destination)
    report = {"completed": False, "test_accessed": False, "stride": 8, "groups": {},
              "protocol": "同VAL16/同GT格点像素加权；oracle为fine十二离散点，不是top4或区间覆盖；无源支持不删除GT"}
    for group in config["groups"]:
        settings = json.loads(path(group["settings"]).read_text())
        settings = dict(settings, smoke=False)
        source = copy.deepcopy(json.loads(path(settings["data_config"]).read_text()))
        for domain in source["datasets"] + source.get("validation_datasets", []):
            domain["manifests"] = {key: value for key, value in domain["manifests"].items()
                                   if key in ("train", "val")}
        _, _, validation, plan, _ = prepare(source, settings)
        dataset = MixedDataset(validation, "val")
        result = {"plan": plan, "methods": {}}
        report["groups"][group["name"]] = result
        for name, checkpoint in group["methods"].items():
            model, record = load_model(checkpoint, device)
            if any(record["config"][key] != source[key] for key in ("image_size", "matching_image_size")):
                raise ValueError("权重与诊断数据输入尺寸不同")
            del record
            result["methods"][name] = {"checkpoint": str(path(checkpoint)),
                                       "sha256": digest(path(checkpoint)),
                                       "domains": evaluate(model, dataset, plan, device)}
            write_json(destination, report)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"完成候选诊断：{group['name']}/{name}", flush=True)
    report["completed"] = True
    write_json(destination, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
