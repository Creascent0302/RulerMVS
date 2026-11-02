"""固定新V2权重，仅在既定VAL development比较MAP与全候选L1 medoid。"""

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import Models, add_statistics, digest, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from research.near_mid_evaluation_20261028.metrics import (
    depth_metrics, depth_sums, distance_limits, ground_truth, regions,
)
from tools.benchmark_public_speed import prepare_model
from tools.compare_public_pretrained import configure
from tools.smoke_outdoor_datasets import offline_only

SOURCE = ROOT / "runs/active/v2_restart_20261029/evaluation/configs"
PLAN = ROOT / "runs/active/near_evidence_20261030/diagnostics/plan.json"
OUTPUT = ROOT / "runs/active/near_evidence_20261030/readout_probe"


def medoid_index(depth, probability, valid):
    """在已有连续候选中最小化期望L1；精确并列取较小深度以免依赖槽位顺序。"""
    weights = torch.where(valid, probability, 0)
    distance = (depth.unsqueeze(2) - depth.unsqueeze(1)).abs()
    risk = (distance * weights.unsqueeze(1)).sum(2)
    risk = risk.masked_fill(~valid, torch.inf)
    minimum = risk.amin(1, keepdim=True)
    tied_depth = depth.masked_fill((risk != minimum) | ~valid, torch.inf)
    index = tied_depth.argmin(1, keepdim=True)
    return index, risk


def unit_tests():
    def select(depths, weights):
        z = torch.tensor(depths, dtype=torch.float64).reshape(1, -1, 1, 1)
        p = torch.tensor(weights, dtype=torch.float64).reshape_as(z)
        index, risk = medoid_index(z, p, torch.ones_like(z, dtype=torch.bool))
        return float(z.gather(1, index)), risk.flatten()

    assert select([1, 4, 9], [.1, .7, .2])[0] == 4
    assert select([2, 8], [.5, .5])[0] in (2, 8)
    assert select([2, 5, 8], [.5, 0, .5])[0] in (2, 8)
    original = select([1, 3, 6, 10], [.2, .35, .3, .15])[0]
    assert select([6, 1, 10, 3], [.3, .2, .15, .35])[0] == original
    assert select([1, 3, 3, 6, 10], [.2, .175, .175, .3, .15])[0] == original
    generator = torch.Generator().manual_seed(20261030)
    for _ in range(40):
        z = torch.rand(2, 12, 3, 4, generator=generator, dtype=torch.float64) * 30
        p = torch.rand(z.shape, generator=generator, dtype=torch.float64).softmax(1)
        valid = torch.rand(z.shape, generator=generator) > .2
        index, risk = medoid_index(z, p, valid)
        chosen = z.gather(1, index)
        enumerated = torch.stack([
            ((z - z[:, k:k + 1]).abs() * p * valid).sum(1)
            for k in range(z.shape[1])
        ], 1).masked_fill(~valid, torch.inf)
        assert torch.allclose(risk, enumerated)
        assert torch.allclose(risk.gather(1, index), enumerated.amin(1, keepdim=True))
        assert bool((chosen == z).any(1).all())
    z = torch.tensor([1., 2.]).reshape(1, 2, 1, 1)
    index, _ = medoid_index(z, z * 0, torch.zeros_like(z, dtype=torch.bool))
    assert int(index) == 0
    return {"passed": 6, "random_enumeration_cases": 40,
            "checks": ["majority_mode", "no_interpolated_surface", "permutation",
                       "split_duplicate_mass", "enumerated_minimum_risk", "all_invalid_index_safe"]}


def install_readout(decoder):
    original = decoder._prediction
    state = {"method": "map"}

    def prediction(self, features, candidates, valid, similarity, support,
                   lower, upper, observation, use_offset=True):
        result = original(features, candidates, valid, similarity, support,
                          lower, upper, observation, use_offset)
        if state["method"] == "medoid" and observation["stride"] == 4:
            index, _ = medoid_index(result["candidate_depth"], result["probability"], result["valid"])
            result["selected_index"] = index
            selected = result["candidate_depth"].gather(1, index)
            result["depth"] = torch.where(result["ref_valid"], selected, 0)
            result["confidence"] = result["probability"].gather(1, index)
        return result

    decoder._prediction = types.MethodType(prediction, decoder)
    return state


def measure(native, batch, domain):
    truth, valid = ground_truth(batch, domain)
    masks = regions(truth, valid, domain)
    masks["near_edge"] = masks["near"] & masks["gradient_band"]
    scale = float(domain["depth_scale"])
    output = {"native_grids": {}, "common_stride8": {}}
    stages = {"coarse": (native["coarse"]["depth"], 8),
              "fine": (native["fine"]["depth"], 4), "final": (native["depth"], 1)}
    for name, (depth, stride) in stages.items():
        depth = depth * scale
        for protocol, step in (("native_grids", 1), ("common_stride8", 8 // stride)):
            value = depth[..., ::step, ::step]
            gt_stride = stride * step
            prediction_valid = torch.isfinite(value) & (value > 0)
            output[protocol][name] = {
                region: depth_sums(value, prediction_valid, truth[..., ::gt_stride, ::gt_stride],
                                   mask[..., ::gt_stride, ::gt_stride])
                for region, mask in masks.items()
            }
    return output


def add_measurement(total, current):
    for protocol, stages in current.items():
        for stage, region_values in stages.items():
            for region, values in region_values.items():
                target = total.setdefault(protocol, {}).setdefault(stage, {}).setdefault(region, {})
                add_statistics(target, values)


def summarize(total):
    return {protocol: {stage: {region: depth_metrics(sums) for region, sums in values.items()}
                       for stage, values in stages.items()} for protocol, stages in total.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-tests", action="store_true")
    args = parser.parse_args()
    tests = unit_tests()
    print(json.dumps({"unit_tests": tests}), flush=True)
    if args.unit_tests:
        return
    offline_only()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"拒绝覆盖已有试验：{OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    public, carla, others = [json.loads((SOURCE / (name + ".json")).read_text())
                             for name in ("public", "carla", "carla_other_refs")]
    jobs = [(d, public, "v2_restart_public_best") for d in public["datasets"]]
    jobs += [(d, c, "v2_restart_carla_best") for c in (carla, others) for d in c["datasets"]]
    jobs.sort(key=lambda item: {"Hypersim_subset": 0, "CARLA_new_upper": 1}.get(item[0]["name"], 2))
    original_plan = json.loads(PLAN.read_text())
    plan = {name: entry["development"] for name, entry in original_plan.items()}
    config = dict(public, tf32=False)
    device = configure(config, 0)
    report = {
        "completed": False, "unit_tests": tests, "domains": {}, "selected_indices": plan,
        "test_accessed": False, "confirmation_accessed": False,
        "protocol": "11域既定VAL每域偶位8DEV；新best固定权重；FP32/TF32off；同帧共享encode，只有fine读出改变。",
        "timing_note": "单帧共享编码；每域先预热两次decoder，逐帧交替顺序计时，非完整网络延迟结论。",
        "sources": {str(Path(__file__).relative_to(ROOT)): digest(__file__),
                    "plan_sha256": digest(PLAN)},
        "checkpoint_sha256": {}, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(device),
    }
    write_json(OUTPUT / "summary.json", report)
    for domain, source_config, method in jobs:
        local = copy.deepcopy(source_config)
        local["methods"] = {method: local["methods"][method]}
        for item in local["datasets"]:
            item["manifests"] = {"val": item["manifests"]["val"]}
        local_domain = copy.deepcopy(domain)
        local_domain["manifests"] = {"val": domain["manifests"]["val"]}
        weights = local["methods"][method].get("weights", local["own_weights"])
        if weights not in report["checkpoint_sha256"]:
            report["checkpoint_sha256"][weights] = digest(weights)
        models = Models(local, device)
        dataset = build_dataset(dataset_config(local_domain, local), "val", load_targets=True)
        totals = {"map": {}, "medoid": {}}
        frames = []
        timing = {"map": [], "medoid": []}
        changes = {"pixels": 0, "changed_index": 0, "changed_depth": 0}
        for frame_number, index in enumerate(plan[domain["name"]]):
            batch = move(collate_frames([dataset[index]]), device)
            net = prepare_model(models, method, local["methods"][method], domain, batch["inputs"])
            if frame_number == 0:
                if net.decoder.fine_readout == "mode_mean":
                    raise ValueError("权重默认不是MAP，不能把mode_mean冒充原MAP")
                state = install_readout(net.decoder)
            with torch.inference_mode():
                scale = float(domain["depth_scale"])
                net.decoder.set_range(domain["near"] / scale, domain["far"] / scale)
                work = net.encode(batch["inputs"], domain, reference=domain["reference_index"])
                if frame_number == 0:
                    for name in ("map", "medoid"):
                        state["method"] = name
                        for _ in range(2):
                            net.decoder(work)
                native = {}
                order = ("map", "medoid") if frame_number % 2 == 0 else ("medoid", "map")
                for name in order:
                    state["method"] = name
                    torch.cuda.synchronize(device)
                    begin = time.perf_counter()
                    native[name] = net.decoder(work)
                    torch.cuda.synchronize(device)
                    timing[name].append((time.perf_counter() - begin) * 1000)
                # 只改fine决策：相同上游供给与分数必须逐位一致。
                for stage in ("coarse", "fine"):
                    for field in ("scores", "probability", "candidate_depth"):
                        if not torch.equal(native["map"][stage][field], native["medoid"][stage][field]):
                            raise AssertionError(f"读出外数值变化：{stage}.{field}")
                truth, valid = ground_truth(batch, domain)
                fine_valid = valid[..., ::4, ::4]
                changes["pixels"] += int(fine_valid.sum())
                changes["changed_index"] += int(((native["map"]["fine"]["selected_index"] !=
                                                  native["medoid"]["fine"]["selected_index"]) & fine_valid).sum())
                changes["changed_depth"] += int(((native["map"]["fine"]["depth"] !=
                                                  native["medoid"]["fine"]["depth"]) & fine_valid).sum())
                frame_metrics = {}
                for name in ("map", "medoid"):
                    measured = measure(native[name], batch, domain)
                    add_measurement(totals[name], measured)
                    frame_metrics[name] = summarize(measured)
            frames.append({"index": index, "metrics": frame_metrics})
            print(json.dumps({"domain": domain["name"], "index": index,
                              "near_final_MAE": {name: values["native_grids"]["final"]["near"]["MAE"]
                                                 for name, values in frame_metrics.items()}}, ensure_ascii=False), flush=True)
            del batch, work, native
        result = {"units": domain["geometry_units"], "distance_limits": distance_limits(domain),
                  "method": method, "frames": frames, "selection_changes": changes,
                  "metrics": {name: summarize(total) for name, total in totals.items()},
                  "statistics": totals, "decoder_wall_ms": timing,
                  "decoder_median_ms": {name: statistics.median(values) for name, values in timing.items()}}
        report["domains"][domain["name"]] = result
        write_json(OUTPUT / (domain["name"] + ".json"), result)
        write_json(OUTPUT / "summary.json", report)
        models.loaded.clear()
        del net, models
        torch.cuda.empty_cache()
    report["completed"] = True
    write_json(OUTPUT / "summary.json", report)


if __name__ == "__main__":
    main()
