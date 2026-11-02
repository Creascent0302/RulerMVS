"""固定末步的完整VAL：相同冻结编码只计算一次，逐头独立预测。

这是离线评测优化，不是部署速度。每域首帧与每个完整模型逐位核对。
命令只接受配置文件；分片覆盖、模型和数据一致性在合并时检查。
"""

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
from jar.pretraining.comparison import digest, equal_pixel_inputs, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.runtime import move
from research.near_mid_evaluation_20261028.metrics import ground_truth, regions
from research.paper_ablation_20261102.handoff_fullval import finish
from research.paper_ablation_20261102.handoff_sharedval import (
    add_record, frozen_hash, metric_record, tensor_leaves,
)
from research.v7_domain_20261102.model import build_model
from tools.smoke_outdoor_datasets import offline_only


@torch.inference_mode()
def run(settings):
    offline_only()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(settings["device"])
    torch.cuda.set_device(device)
    output = Path(settings["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    models, records = {}, {}
    signature = None
    for job in settings["models"]:
        if job.get("completion") and not json.loads(Path(job["completion"]).read_text())["completed"]:
            raise RuntimeError("训练未完成")
        saved = torch.load(job["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
        model, _ = build_model(saved, job.get("variant", "control"))
        actual_signature = frozen_hash(model)
        if signature is not None and signature != actual_signature:
            raise ValueError("编码器不同，禁止共享特征")
        signature = actual_signature
        model.eval().requires_grad_(False).to(device)
        install_position_cache(model.foundation.model)
        models[job["name"]] = model
        records[job["name"]] = {"checkpoint": job["checkpoint"],
                                "sha256": digest(job["checkpoint"]), "datasets": {}}
        del saved
    config = json.loads(Path(settings["data_config"]).read_text())
    config["datasets"] = config.pop("validation_datasets", config["datasets"])
    for domain in config["datasets"]:
        domain["manifests"] = {"val": domain["manifests"]["val"]}
    dataset = MixedDataset(config, "val")
    encoder = next(iter(models.values()))
    checks = []
    for number, domain in enumerate(dataset.domains):
        length = len(dataset.datasets[number])
        indices = list(range(settings["rank"], length, settings["world"]))
        if settings.get("smoke"):
            indices = indices[:1]
        keys = [(number, index, -1) for index in indices]
        loader = DataLoader(dataset, batch_size=1, sampler=keys, collate_fn=collate_mixed,
                            num_workers=2, pin_memory=True)
        scale = float(domain["depth_scale"])
        for model in models.values():
            model.decoder.set_range(domain["near"] / scale, domain["far"] / scale)
        totals = {name: {} for name in models}
        with (output / (domain["name"] + ".jsonl")).open("w") as log:
            for offset, (index, batch) in enumerate(zip(indices, loader)):
                inputs = equal_pixel_inputs(move(batch["inputs"], device))
                work = encoder.encode(inputs, domain, reference=0)
                snapshot = {name: value.clone() for name, value in tensor_leaves(work).items()} if offset == 0 else {}
                truth, valid = ground_truth(batch, dict(domain, reference_index=0))
                masks = regions(truth, valid, domain)
                row = {"index": index, "models": {}}
                for name, model in models.items():
                    prediction = model.decoder(work)
                    if offset == 0:
                        complete = model(inputs, domain, reference=0)
                        for key in ("depth", "valid"):
                            torch.testing.assert_close(prediction[key], complete[key], rtol=0, atol=0)
                        for stage in ("coarse", "fine"):
                            torch.testing.assert_close(prediction[stage]["depth"], complete[stage]["depth"], rtol=0, atol=0)
                        for key, value in tensor_leaves(work).items():
                            torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)
                        checks.append({"model": name, "dataset": domain["name"], "index": index})
                        del complete
                    raw, _ = metric_record(prediction, truth, masks, scale)
                    add_record(totals[name], raw)
                    row["models"][name] = raw
                    del prediction
                log.write(json.dumps(row) + "\n")
                if offset % 50 == 0:
                    log.flush()
                    print(json.dumps({"domain": domain["name"], "frames": offset + 1,
                                      "total": len(indices), "rank": settings["rank"]}), flush=True)
        for name in models:
            records[name]["datasets"][domain["name"]] = {
                "indices": indices, "total": length, "units": domain["geometry_units"],
                "manifest_sha256": digest(domain["manifests"]["val"]), "raw": totals[name]}
    write_json(output / "completed.json", {"completed": True, "split": "val", "smoke": settings.get("smoke", False),
        "test_accessed": False, "models": records, "frozen_encoder_sha256": signature,
        "bitwise_checks": checks, "settings": settings, "source_sha256": digest(__file__)})


def merge(settings):
    reports = [json.loads((Path(folder) / "completed.json").read_text()) for folder in settings["shards"]]
    if not all(item["completed"] and not item["smoke"] for item in reports):
        raise ValueError("不是完整VAL分片")
    result = {"completed": True, "split": "full_val", "test_accessed": False, "models": {}}
    for name, first_model in reports[0]["models"].items():
        models = [report["models"][name] for report in reports]
        if len({item["sha256"] for item in models}) != 1:
            raise ValueError("分片权重不同")
        destination = {"checkpoint": first_model["checkpoint"], "sha256": first_model["sha256"], "datasets": {}}
        for domain, first in first_model["datasets"].items():
            parts = [model["datasets"][domain] for model in models]
            indices = [index for part in parts for index in part["indices"]]
            if sorted(indices) != list(range(first["total"])):
                raise ValueError("分片缺帧或重复")
            if len({part["manifest_sha256"] for part in parts}) != 1:
                raise ValueError("分片数据不同")
            raw = {}
            for part in parts:
                add_record(raw, part["raw"])
            destination["datasets"][domain] = {"frames": len(indices), "units": first["units"],
                "raw": raw, "metrics": finish(raw), "manifest_sha256": first["manifest_sha256"]}
        result["models"][name] = destination
    result["bitwise_checks"] = sum(len(report["bitwise_checks"]) for report in reports)
    write_json(settings["output"], result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    settings = json.loads(Path(parser.parse_args().config).read_text())
    if settings.get("merge"):
        merge(settings)
    else:
        run(settings)
