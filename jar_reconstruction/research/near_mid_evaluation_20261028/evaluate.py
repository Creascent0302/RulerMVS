"""固定权重的全量测试：同帧输入复用，逐域深度、阶段和可见表面指标。"""

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import Models, add_statistics, digest, equal_pixel_inputs, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from tools.compare_public_pretrained import configure
from tools.smoke_outdoor_datasets import fingerprint, offline_only
from research.near_mid_evaluation_20261028.metrics import (
    VisibleSurface, depth_metrics, depth_sums, ground_truth, regions, surface_metrics,
)


def finish(record):
    return {"frames": len(record["indices"]),
            "depth": {name: depth_metrics(sums) for name, sums in record["depth"].items()},
            "stages_stride8": {name: depth_metrics(sums) for name, sums in record["stages"].items()},
            "surface": surface_metrics(record["surface"], record["thresholds"])}


def predict_with_stages(models, name, settings, domain, inputs):
    """只接收inputs；同像素对照须真正替换匹配图，不能绕过适配层。"""
    stages = {}
    if settings["kind"] == "own":
        work = equal_pixel_inputs(inputs) if settings.get("input_mode") == "equal_pixels" else inputs
        native = models.get(name, domain)(work, domain, reference=domain["reference_index"])
        scale = native["unit_scale"]
        prediction = {"depth": native["depth"] * scale, "valid": native["valid"]}
        for stage, step in (("coarse", 1), ("fine", 2)):
            stages[stage] = (native[stage]["depth"][..., ::step, ::step] * scale).cpu()
    else:
        prediction = models.forward(name, domain, inputs)
    depth, valid = models.convert(prediction, inputs)
    return depth.cpu(), valid.cpu(), stages


def run_part(config, task, models, datasets, device, output, protocol):
    domain_id, part, indices = task
    domain = config["datasets"][domain_id]
    path = output / domain["name"] / f"part_{part:04d}.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("protocol") == protocol and old.get("completed"):
            return {"event": "cached", "domain": domain["name"], "part": part}
    if domain_id not in datasets:
        datasets[domain_id] = build_dataset(dataset_config(domain, config), config["split"], load_targets=True)
    loader = DataLoader(datasets[domain_id], batch_size=1, sampler=indices,
                        num_workers=config["loader_workers"], pin_memory=True, collate_fn=collate_frames)
    totals, records = {}, []
    begin = time.monotonic()
    for index, batch in zip(indices, loader):
        inputs = move(batch["inputs"], device)
        truth, mask = ground_truth(batch, domain)
        masks = regions(truth, mask, domain)
        surface = VisibleSurface(truth, mask, batch["inputs"]["K"][0, domain["reference_index"]],
                                 domain, config["surface_queries"])
        frame = {"index": index, "valid_views": int(inputs["view_mask"].sum()), "methods": {}}
        if index in (0, len(datasets[domain_id]) // 2, len(datasets[domain_id]) - 1):
            frame["input_sha256"] = fingerprint(batch["inputs"])
        for name, settings in config["methods"].items():
            with torch.inference_mode():
                depth, valid, stages = predict_with_stages(models, name, settings, domain, inputs)
            statistics = {region: depth_sums(depth, valid, truth, selected) for region, selected in masks.items()}
            surface_sums = surface.measure(depth, valid)
            stages["final"] = depth[..., ::8, ::8]
            coarse_truth, coarse_mask = truth[..., ::8, ::8], mask[..., ::8, ::8]
            stage_sums = {}
            for stage, value in stages.items():
                if value.shape != coarse_truth.shape:
                    raise ValueError(f"阶段格点不符：{stage} {value.shape} != {coarse_truth.shape}")
                stage_sums[stage] = depth_sums(value, torch.isfinite(value) & (value > 0), coarse_truth, coarse_mask)
            record = totals.setdefault(name, {"indices": [], "depth": {}, "surface": {}, "stages": {},
                                              "thresholds": surface.thresholds})
            record["indices"].append(index)
            for region, item in statistics.items():
                add_statistics(record["depth"].setdefault(region, {}), item)
            for stage, item in stage_sums.items():
                add_statistics(record["stages"].setdefault(stage, {}), item)
            add_statistics(record["surface"], surface_sums)
            frame["methods"][name] = {"depth": statistics, "surface": surface_sums, "stages": stage_sums}
        records.append(frame)
    write_json(path, {"completed": True, "protocol": protocol, "totals": totals, "frames": records})
    return {"event": "complete", "domain": domain["name"], "part": part, "frames": len(indices),
            "seconds": round(time.monotonic() - begin, 2)}


def worker(config, gpu, tasks, results, output, protocol):
    offline_only()
    device = configure(config, gpu)
    models, datasets = Models(config, device), {}
    while (task := tasks.get()) is not None:
        try:
            result = run_part(config, task, models, datasets, device, Path(output), protocol)
        except Exception:
            result = {"event": "failed", "domain": config["datasets"][task[0]]["name"],
                      "part": task[1], "error": traceback.format_exc()}
            write_json(Path(output) / "failures" / f"{result['domain']}_{task[1]}.json", result)
        results.put(dict(result, gpu=gpu))


def merge(config, tasks, output, protocol):
    report = {"completed": True, "protocol": protocol, "split": config["split"], "datasets": {}}
    for domain_id, domain in enumerate(config["datasets"]):
        selected = [task for task in tasks if task[0] == domain_id]
        expected = sorted(index for task in selected for index in task[2])
        totals = {}
        for _, part, _ in selected:
            source = json.loads((output / domain["name"] / f"part_{part:04d}.json").read_text())
            assert source["protocol"] == protocol and source["completed"]
            for method, values in source["totals"].items():
                target = totals.setdefault(method, {"indices": [], "depth": {}, "stages": {}, "surface": {},
                                                    "thresholds": values["thresholds"]})
                target["indices"].extend(values["indices"])
                for kind in ("depth", "stages"):
                    for region, item in values[kind].items():
                        add_statistics(target[kind].setdefault(region, {}), item)
                add_statistics(target["surface"], values["surface"])
        for value in totals.values():
            assert sorted(value["indices"]) == expected and len(set(value["indices"])) == len(expected)
        for region in ("full", "near", "mid", "far", "gradient_band", "smooth"):
            assert len({v["depth"][region]["gt_pixels"] for v in totals.values()}) == 1
        report["datasets"][domain["name"]] = {"units": domain["geometry_units"], "num_views": domain["num_views"],
                                                "methods": {name: finish(value) for name, value in totals.items()}}
    write_json(output / "summary.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true", help="仅VAL首中末，单独目录")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    config["split"] = "val" if args.smoke else config.get("evaluation_split", "test")
    if config["split"] not in ("val", "test"):
        raise ValueError("固定权重评测仅允许val或test")
    if args.smoke:
        config["gpus"] = config["gpus"][:2]
        config["loader_workers"] = 0
    output = Path(config["output"]) / ("smoke_val" if args.smoke else "full_" + config["split"])
    output.mkdir(parents=True, exist_ok=True)
    sources = list((ROOT / "jar/pretraining").glob("*.py"))
    sources += list(Path(__file__).parent.glob("*.py"))
    sources += [ROOT / "jar/models/ruler_direct.py", ROOT / "official_methods/metrics.py"]
    sources += [ROOT / "jar/data.py", ROOT / "jar/multifocal/data.py"]
    sources += list((ROOT / "official_methods/methods/mvsanywhere").rglob("*.py"))
    audit = {"config": config, "weights_sha256": {name: digest(value["weights"]) for name, value in config["methods"].items()},
             "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sources},
             "manifest_sha256": {d["name"]: digest(d["manifests"][config["split"]]) for d in config["datasets"]}}
    protocol = hashlib.sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    write_json(output / "audit.json", audit)
    tasks = []
    for domain_id, domain in enumerate(config["datasets"]):
        length = len(build_dataset(dataset_config(domain, config), config["split"], load_targets=True))
        indices = sorted(set((0, length // 2, length - 1))) if args.smoke else list(range(length))
        for start in range(0, len(indices), config["shard_size"]):
            tasks.append((domain_id, start // config["shard_size"], indices[start:start + config["shard_size"]]))
    context = mp.get_context("spawn")
    pending, results = context.Queue(), context.Queue()
    for task in tasks:
        pending.put(task)
    for _ in config["gpus"]:
        pending.put(None)
    processes = [context.Process(target=worker, args=(config, gpu, pending, results, str(output), protocol))
                 for gpu in config["gpus"]]
    for process in processes:
        process.start()
    failures = 0
    received = 0
    while received < len(tasks):
        try:
            result = results.get(timeout=30)
        except queue.Empty:
            if all(not process.is_alive() for process in processes):
                raise RuntimeError("评测进程退出但分片未全部返回")
            continue
        received += 1
        failures += result["event"] == "failed"
        print(json.dumps(dict(result, received=received, total=len(tasks)), ensure_ascii=False), flush=True)
    for process in processes:
        process.join()
    if failures or any(process.exitcode for process in processes):
        raise RuntimeError(f"存在{failures}个失败分片，不能宣称测试完成")
    merge(config, tasks, output, protocol)
    print(f"完成：{output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
