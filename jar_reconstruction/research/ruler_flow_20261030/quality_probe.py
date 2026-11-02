"""固定权重屏蔽新增quality输出，隔离同一四峰轨迹上的评分作用；不训练。"""

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import add_statistics, digest, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.runtime import move, seed_all
from research.near_mid_evaluation_20261028.metrics import depth_metrics, depth_sums, ground_truth, regions
from research.ruler_flow_20261030.diagnostics import aggregate, analyze, truth_masks
from research.ruler_flow_20261030.pilot import load_model, path
from tools.smoke_outdoor_datasets import offline_only


@torch.inference_mode()
def run():
    offline_only()
    seed_all(20261030)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    study = ROOT / "runs/active/ruler_flow_20261030"
    output = study / "diagnosis/quality_probe.json"
    if output.exists():
        raise FileExistsError(output)
    prepared_file = study / "diagnosis/prepared.json"
    prepared = json.loads(prepared_file.read_text())
    plan = {
        "selection": {group: item["selection"] for group, item in prepared["groups"].items()},
        "prepared_sha256": digest(prepared_file), "source_sha256": digest(Path(__file__)),
        "weights_sha256": {group: digest(study / f"{group}_flow/final.pt") for group in prepared["groups"]},
        "device": str(device), "maximum_forwards": 32, "inference_steps": 1,
        "scope": "同已查看4DEV索引，原Flow1与仅屏蔽新增quality输出；无重训，不是去头重训结论。",
        "test_accessed": False, "training_performed": False,
    }
    write_json(output.with_name("quality_probe_plan.json"), plan)
    results, forwards = {}, 0
    for group, group_plan in prepared["groups"].items():
        training = json.loads((study / f"{group}_flow/settings.json").read_text())
        config = json.loads(path(training["data_config"]).read_text())
        config["datasets"] = copy.deepcopy(config.get("validation_datasets", config["datasets"]))
        config.pop("validation_datasets", None)
        config["datasets"] = [item for item in config["datasets"] if item["name"] in group_plan["selection"]]
        for domain in config["datasets"]:
            domain["manifests"] = {"val": domain["manifests"]["val"]}
            if digest(path(domain["manifests"]["val"])) != group_plan["selection"][domain["name"]]["manifest_sha256"]:
                raise RuntimeError("VAL清单改变")
        dataset = MixedDataset(config, "val")
        with torch.inference_mode(False):
            model, _ = load_model(study / f"{group}_flow/final.pt", device, inference_steps=1)
        for domain_index, domain in enumerate(dataset.domains):
            name = domain["name"]
            stage_rows = {key: [] for key in ("flow1", "quality_zero")}
            sums = {key: {} for key in stage_rows}
            checks = []
            for index in group_plan["selection"][name]["indices"]:
                batch = move(collate_mixed([dataset[(domain_index, index, -1)]]), device)
                baseline = model(batch["inputs"], domain, reference=0)
                handle = model.decoder.flow_head.quality.register_forward_hook(
                    lambda module, arguments, value: torch.zeros_like(value))
                try:
                    changed = model(batch["inputs"], domain, reference=0)
                finally:
                    handle.remove()
                forwards += 2
                candidate_same = torch.equal(baseline["fine"]["candidate_depth"], changed["fine"]["candidate_depth"])
                state_same = torch.equal(baseline["flow_state"], changed["flow_state"])
                if not candidate_same or not state_same:
                    raise RuntimeError("屏蔽quality改变了轨迹或候选，不能归因为评分")
                truth, masks = truth_masks(batch, domain)
                truth_full, valid_full = ground_truth(batch, dict(domain, reference_index=0))
                full_masks = regions(truth_full, valid_full, domain)
                valid = masks["full"]
                index_diff = (baseline["fine"]["selected_index"] != changed["fine"]["selected_index"])[:, 0]
                checks.append({"index": index, "candidate_depth_bitwise_equal": candidate_same,
                               "flow_state_bitwise_equal": state_same,
                               "changed_winners": int(index_diff[valid].sum()), "gt_pixels_stride4": int(valid.sum())})
                for method, prediction in (("flow1", baseline), ("quality_zero", changed)):
                    stage_rows[method].append(analyze(prediction, domain, truth, masks)[0])
                    estimate = prediction["depth"] * prediction["unit_scale"]
                    for region in ("near", "full"):
                        values = depth_sums(estimate, prediction["valid"], truth_full, full_masks[region])
                        add_statistics(sums[method].setdefault(region, {}), values)
            results[name] = {
                "units": domain["geometry_units"], "frames": len(checks), "checks": checks,
                "methods": {method: {"fine_stride4": aggregate(stage_rows[method]),
                                      "depth": {region: depth_metrics(values) for region, values in sums[method].items()}}
                            for method in stage_rows},
            }
            print(name, {method: results[name]["methods"][method]["depth"]["near"]["MAE"] for method in stage_rows}, flush=True)
        del model
    if forwards != 32:
        raise RuntimeError(f"实际前向数{forwards}不符合注册预算32")
    write_json(output, {"completed": True, "forwards": forwards, "domains": results,
                        "plan_sha256": digest(output.with_name("quality_probe_plan.json")), "scope": plan["scope"]})


if __name__ == "__main__":
    run()
