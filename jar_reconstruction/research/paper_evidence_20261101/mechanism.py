"""固定VAL上的候选交接诊断；推理干预仅分析依赖，不代替同预算重训消融。"""

import argparse
import copy
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import digest, equal_pixel_inputs, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from research.near_mid_evaluation_20261028.metrics import ground_truth, regions
from research.rulermvs_v6_20261101.diagnose import cpu_prediction, finish, update_record
from research.rulermvs_v6_20261101.speed_final import initialize, load


def run(settings):
    initialize(settings)
    output = Path(settings["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    plan = json.loads(Path(settings["evaluation_plan"]).read_text())
    indices_plan = json.loads(Path(settings["diagnostics"]).read_text())["plans"]
    report = {
        "completed": False, "split": "val", "test_accessed": False,
        "settings": settings, "datasets": {}, "weights_sha256": {},
        "notes": [
            "V5/V6分别使用其已完整训练的VAL-best；并非同训练预算因果消融。",
            "V6 parent/marginal/off共享同一权重和编码，属于推理依赖诊断。",
            "oracle读取标签只在前向结束后统计，不是可部署预测或真实性能。",
            "粗细阶段在相同stride8格点比较；全分辨率误差另列。",
            "候选族可能重叠或来自同一表面，族身份不是物理实例标签。",
        ],
    }
    device = "cuda:" + str(settings.get("gpu", 0))
    with torch.inference_mode():
        for phase in ("public", "carla_new"):
            config = json.loads(Path(plan["groups"][phase + "_main"]["val_config"]).read_text())
            paths = {v + "_checkpoint": config["methods"][v + "_best"]["weights"]
                     for v in ("v5", "v6")}
            models = {v: load(dict(settings, **paths), v + "_eager") for v in ("v5", "v6")}
            report["weights_sha256"][phase] = {k: digest(v) for k, v in paths.items()}
            for domain in config["datasets"]:
                name = domain["name"]
                # 沿用实验前已登记的16VAL索引，不能按本轮误差挑好看的样本。
                selected = indices_plan[name]
                if digest(domain["manifests"]["val"]) != selected["manifest_sha256"]:
                    raise ValueError(f"{name}的固定VAL清单已改变")
                dataset = build_dataset(dataset_config(domain, config), "val", load_targets=True)
                totals, frames = {}, []
                for index in selected["indices"]:
                    batch = collate_frames([dataset[index]])
                    inputs = equal_pixel_inputs(move(batch["inputs"], device))
                    domain_scale = float(domain["depth_scale"])
                    v5 = models["v5"](inputs, domain, domain["reference_index"])
                    predictions = {"v5_trained": cpu_prediction(v5)}
                    del v5
                    model = models["v6"]
                    model.decoder.set_range(domain["near"] / domain_scale, domain["far"] / domain_scale)
                    encoded = model.encode(inputs, domain, domain["reference_index"])
                    try:
                        for label, enabled, mode in (
                            ("v6_trained", True, "parent"),
                            ("v6_no_parent", True, "marginal"),
                            ("v6_flat_readout", False, "parent"),
                        ):
                            model.decoder.handoff_enabled = enabled
                            model.decoder.handoff.mode = mode
                            predictions[label] = cpu_prediction(model.decoder(encoded))
                    finally:
                        model.decoder.handoff_enabled = True
                        model.decoder.handoff.mode = "parent"
                    del encoded, inputs
                    for label in ("v6_no_parent", "v6_flat_readout"):
                        for stage in ("coarse",):
                            torch.testing.assert_close(predictions[label][stage], predictions["v6_trained"][stage], rtol=0, atol=0)
                        torch.testing.assert_close(predictions[label]["fine"]["candidate_depth"],
                                                   predictions["v6_trained"]["fine"]["candidate_depth"], rtol=0, atol=0)
                    truth, mask = ground_truth(batch, domain)
                    masks = regions(truth, mask, domain)
                    row = {"index": index, "valid_views": int(batch["inputs"]["view_mask"].sum()), "methods": {}}
                    for label, prediction in predictions.items():
                        record = {}
                        update_record(record, prediction, truth, masks, domain_scale, False)
                        row["methods"][label] = finish(record)
                        update_record(totals.setdefault(label, {}), prediction, truth, masks, domain_scale, False)
                    frames.append(row)
                result = {"units": domain["geometry_units"], "indices": selected["indices"],
                          "manifest_sha256": selected["manifest_sha256"],
                          "methods": {k: finish(v) for k, v in totals.items()}, "frames": frames}
                report["datasets"][name] = result
                write_json(output / "results.json", report)
                print(json.dumps({"completed_domain": name, "frames": len(frames)}, ensure_ascii=False), flush=True)
            models.clear()
            del model, models
            gc.collect()
            torch.cuda.empty_cache()
    report["completed"] = True
    report["source_sha256"] = {str(Path(__file__).relative_to(ROOT)): digest(__file__)}
    write_json(output / "results.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))
