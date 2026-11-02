"""仅为消融评测复用相同的冻结编码；逐模型完整前向对齐，不作为速度结果。"""

import argparse
import copy
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import add_statistics, digest, equal_pixel_inputs, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.runtime import move, seed_all
from research.near_evidence_20261030.pilot import parameter_hash
from research.near_mid_evaluation_20261028.metrics import depth_sums, distance_limits, ground_truth
from research.paper_ablation_20261102.geometry import build_geometry_model
from research.paper_ablation_20261102.handoff_fullval import finish
from research.paper_ablation_20261102.handoff_train import build_ablation
from tools.smoke_outdoor_datasets import offline_only


def frozen_hash(model):
    # named_buffers包含非持久buffer，不能仅hash state_dict。
    values = {"parameter:" + name: value for name, value in model.named_parameters()
              if not name.startswith("decoder.")}
    values.update({"buffer:" + name: value for name, value in model.named_buffers()
                   if not name.startswith("decoder.")})
    return parameter_hash(values)


def tensor_leaves(value, name=""):
    if isinstance(value, torch.Tensor):
        return {name: value}
    children = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, (tuple, list)) else []
    result = {}
    for key, child in children:
        result.update(tensor_leaves(child, name + "/" + str(key)))
    return result


def metric_record(prediction, truth, masks, scale):
    depth = prediction["depth"].cpu() * scale
    valid = prediction["valid"].cpu()
    stages = {"depth": depth, "coarse_stride8": prediction["coarse"]["depth"].cpu() * scale,
              "fine_stride8": prediction["fine"]["depth"][..., ::2, ::2].cpu() * scale,
              "final_stride8": depth[..., ::8, ::8]}
    raw = {}
    for stage, estimate in stages.items():
        full = stage == "depth"
        target = truth if full else truth[..., ::8, ::8]
        good = valid if full else torch.isfinite(estimate) & (estimate > 0)
        for region, mask in masks.items():
            raw.setdefault(stage, {})[region] = depth_sums(
                estimate, good, target, mask if full else mask[..., ::8, ::8])
    target = truth[..., ::8, ::8]
    quality = {"coarse_good": (stages["coarse_stride8"] - target).abs() <= .05 * target,
               "fine_good": (stages["fine_stride8"] - target).abs() <= .05 * target}
    return raw, quality


def add_record(total, raw):
    for stage, regions in raw.items():
        for region, sums in regions.items():
            add_statistics(total.setdefault(stage, {}).setdefault(region, {}), sums)


def paired_counts(method, baseline, masks):
    both_good = method["coarse_good"] & baseline["coarse_good"]
    both_bad = ~method["coarse_good"] & ~baseline["coarse_good"]
    values = {
        "both_coarse_good": both_good,
        "baseline_fine_bad_on_both_good": both_good & ~baseline["fine_good"],
        "method_fine_bad_on_both_good": both_good & ~method["fine_good"],
        "baseline_bad_to_method_good": both_good & ~baseline["fine_good"] & method["fine_good"],
        "baseline_good_to_method_bad": both_good & baseline["fine_good"] & ~method["fine_good"],
        "both_coarse_bad": both_bad,
        "baseline_corrected_on_both_bad": both_bad & baseline["fine_good"],
        "method_corrected_on_both_bad": both_bad & method["fine_good"],
        "only_method_coarse_good": method["coarse_good"] & ~baseline["coarse_good"],
        "only_baseline_coarse_good": baseline["coarse_good"] & ~method["coarse_good"],
    }
    return {region: {"pixels": int(mask[..., ::8, ::8].sum()),
                     **{key: int((value & mask[..., ::8, ::8]).sum()) for key, value in values.items()}}
            for region, mask in masks.items()}


def finish_pairs(raw):
    result = {}
    for region, values in raw.items():
        good, bad = values["both_coarse_good"], values["both_coarse_bad"]
        result[region] = {"raw": values,
            "baseline_destroyed_on_common_good": values["baseline_fine_bad_on_both_good"] / good if good else None,
            "method_destroyed_on_common_good": values["method_fine_bad_on_both_good"] / good if good else None,
            "baseline_corrected_on_common_bad": values["baseline_corrected_on_both_bad"] / bad if bad else None,
            "method_corrected_on_common_bad": values["method_corrected_on_both_bad"] / bad if bad else None}
    return result


def load_phase(jobs, phase, device):
    models, records, expected, expected_data = {}, {}, None, None
    for job in jobs:
        settings = json.loads(Path(job["config"]).read_text())
        actual_phase = "public" if Path(settings["data_config"]).stem == "data_public" else "carla"
        if actual_phase != phase:
            continue
        folder = Path(job["output"])
        audit = "geometry_audit.json" if "geometry_ablation" in settings else "handoff_audit.json"
        if not (folder / audit).exists():
            raise RuntimeError(f"{job['name']}训练未完成")
        saved = torch.load(folder / "final.pt", map_location="cpu", weights_only=False, mmap=True)
        if "geometry_ablation" in settings:
            model, config = build_geometry_model(saved, settings["geometry_ablation"])
        else:
            model, config = build_ablation(saved, settings["variant"])
        signature = frozen_hash(model)
        if expected is not None and signature != expected:
            raise ValueError("冻结参数或buffer不相同，禁止共享编码")
        expected = signature
        data_protocol = {"datasets": config.get("validation_datasets", config["datasets"]),
                         "image_size": config["image_size"], "matching_image_size": config["matching_image_size"]}
        if expected_data is not None and data_protocol != expected_data:
            raise ValueError("图像尺寸或VAL数据协议不相同，禁止共用评测输入")
        expected_data = data_protocol
        model.requires_grad_(False).to(device).eval()
        install_position_cache(model.foundation.model)
        # 保留每个完整模型，首样例真实调用各自forward验证，不用替换模块伪造一致。
        models[job["name"]] = model
        records[job["name"]] = {"config": copy.deepcopy(config), "output": str(folder),
                                "checkpoint_sha256": digest(folder / "final.pt"), "frozen_sha256": signature,
                                "settings": settings}
        del saved
    return models, records


@torch.inference_mode()
def run(settings):
    offline_only()
    seed_all(20261102)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)
    device = "cuda:0"
    output = Path(settings["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rank, world = settings["rank"], settings["world"]
    report = {"completed": False, "split": "full_val", "test_accessed": False,
              "rank": rank, "world": world, "models": {}, "checks": []}
    for phase in ("public", "carla"):
        models, records = load_phase(settings["jobs"], phase, device)
        config = copy.deepcopy(next(iter(records.values()))["config"])
        config["datasets"] = config.pop("validation_datasets", config["datasets"])
        for domain in config["datasets"]:
            domain["manifests"] = {"val": domain["manifests"]["val"]}
        for name, info in records.items():
            report["models"][name] = {"phase": phase, "datasets": {},
                                      **{key: info[key] for key in ("output", "checkpoint_sha256", "frozen_sha256")}}
        dataset = MixedDataset(config, "val")
        encoder = next(iter(models.values()))
        for domain_index, domain in enumerate(dataset.domains):
            length = len(dataset.datasets[domain_index])
            indices = list(range(rank, length, world))
            keys = [(domain_index, index, -1) for index in indices]
            loader = DataLoader(dataset, batch_size=1, sampler=keys, collate_fn=collate_mixed,
                                num_workers=2, pin_memory=True)
            scale = float(domain["depth_scale"])
            for model in models.values():
                model.decoder.set_range(domain["near"] / scale, domain["far"] / scale)
            totals = {name: {} for name in models}
            paired = {name: {} for name in models}
            handles = {name: (output / (name + "__" + domain["name"] + ".jsonl")).open("w") for name in models}
            try:
                for offset, (index, batch) in enumerate(zip(indices, loader)):
                    inputs = equal_pixel_inputs(move(batch["inputs"], device))
                    encoded = encoder.encode(inputs, domain, reference=0)
                    snapshot = {name: value.clone() for name, value in tensor_leaves(encoded).items()} if offset == 0 else None
                    truth, valid = ground_truth(batch, dict(domain, reference_index=0))
                    near, far = distance_limits(domain)
                    masks = {"full": valid, "near": valid & (truth < near),
                             "mid": valid & (truth >= near) & (truth < far), "far": valid & (truth >= far)}
                    quality = {}
                    for name, model in models.items():
                        prediction = model.decoder(encoded)
                        if offset == 0:
                            complete = model(inputs, domain, reference=0)
                            for key in ("depth", "valid"):
                                torch.testing.assert_close(prediction[key], complete[key], rtol=0, atol=0)
                            for stage in ("coarse", "fine"):
                                torch.testing.assert_close(prediction[stage]["depth"], complete[stage]["depth"], rtol=0, atol=0)
                            for key, value in tensor_leaves(encoded).items():
                                torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)
                            report["checks"].append({"model": name, "domain": domain["name"], "index": index,
                                                     "full_forward_bitwise_equal": True, "encoded_unchanged": True})
                            del complete
                        raw, quality[name] = metric_record(prediction, truth, masks, scale)
                        add_record(totals[name], raw)
                        handles[name].write(json.dumps({"index": index, "raw": raw}) + "\n")
                        del prediction
                    for name in models:
                        seed = records[name]["settings"]["seed"]
                        baseline = f"geometry_{phase}_control" if name.startswith("geometry_") else f"{phase}_v5_seed{seed}"
                        for region, values in paired_counts(quality[name], quality[baseline], masks).items():
                            add_statistics(paired[name].setdefault(region, {}), values)
                    del inputs, encoded, snapshot
                    if (offset + 1) % 50 == 0:
                        for handle in handles.values():
                            handle.flush()
                        print(json.dumps({"rank": rank, "phase": phase, "domain": domain["name"],
                                          "frames": offset + 1, "total": len(indices)}), flush=True)
            finally:
                for handle in handles.values():
                    handle.close()
            for name, raw in totals.items():
                report["models"][name]["datasets"][domain["name"]] = {
                    "frames": len(indices), "indices": indices, "total_frames": length,
                    "manifest_sha256": digest(domain["manifests"]["val"]), "units": domain["geometry_units"],
                    "raw": raw, "frame_log": str(output / (name + "__" + domain["name"] + ".jsonl")),
                    "paired_candidates_raw": paired[name],
                }
            write_json(output / "progress.json", report)
        models.clear()
        del encoder, model, models, records, dataset
        gc.collect()
        torch.cuda.empty_cache()
    report["completed"] = True
    report["source_sha256"] = digest(__file__)
    write_json(output / "completed.json", report)


def merge(settings):
    output = Path(settings["output"])
    workers = [json.loads((Path(job["output"]) / "completed.json").read_text())
               for job in settings["shared_validation_jobs"]]
    if not all(worker["completed"] for worker in workers):
        raise ValueError("共享完整VAL存在未完成分片")
    for name, info in workers[0]["models"].items():
        records = [worker["models"][name] for worker in workers]
        if len({record["checkpoint_sha256"] for record in records}) != 1:
            raise ValueError("分片权重不一致")
        report = {"completed": True, "split": "full_val", "test_accessed": False,
                  "checkpoint_sha256": info["checkpoint_sha256"], "frozen_sha256": info["frozen_sha256"],
                  "datasets": {}, "selection": "fixed_last_step", "input_mode": "equal_pixels",
                  "shared_frozen_encoder": True, "not_a_latency_benchmark": True}
        for domain, first in info["datasets"].items():
            shards = [record["datasets"][domain] for record in records]
            indices = [index for shard in shards for index in shard["indices"]]
            if sorted(indices) != list(range(first["total_frames"])):
                raise ValueError("完整VAL分片重复或漏帧")
            if len({shard["manifest_sha256"] for shard in shards}) != 1:
                raise ValueError("分片清单不一致")
            raw = {}
            paired = {}
            for shard in shards:
                add_record(raw, shard["raw"])
                for region, values in shard["paired_candidates_raw"].items():
                    add_statistics(paired.setdefault(region, {}), values)
            report["datasets"][domain] = {"frames": len(indices), "units": first["units"],
                "manifest_sha256": first["manifest_sha256"], "raw": raw, "metrics": finish(raw),
                "paired_candidates_stride8": finish_pairs(paired),
                "frame_logs": [shard["frame_log"] for shard in shards]}
        destination = Path(info["output"]) / "full_val.json"
        if destination.exists():
            raise FileExistsError(destination)
        write_json(destination, report)
    write_json(output / "shared_val_completed.json", {"completed": True, "models": len(workers[0]["models"]),
        "workers": len(workers), "forward_checks": sum(len(worker["checks"]) for worker in workers),
        "source_sha256": digest(__file__), "test_accessed": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
