"""固定已训练MeanFlow权重，只清零末层输出，定位连续修正的实际作用。"""

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset
from jar.runtime import seed_all
from research.near_evidence_20261030.pilot import evaluate, parameter_hash
from research.ruler_flow_v2_20261031.pilot import load_model
from tools.smoke_outdoor_datasets import offline_only


def run_phase(folder, output, device):
    checkpoint = folder / "final.pt"
    checkpoint_hash = digest(checkpoint)
    plan = json.loads((folder / "plan.json").read_text())["validation"]
    model, record = load_model(checkpoint, device)
    config = copy.deepcopy(record["config"])
    config["datasets"] = copy.deepcopy(config.get("validation_datasets", config["datasets"]))
    for domain in config["datasets"]:
        domain["manifests"] = {"val": domain["manifests"]["val"]}
        if digest(domain["manifests"]["val"]) != plan[domain["name"]]["manifest_sha256"]:
            raise RuntimeError("VAL清单发生改变")
    dataset = MixedDataset(config, "val")
    original_lattice = model.decoder.depth_lattice.clone()
    original_range = (model.decoder.near, model.decoder.far)
    before = parameter_hash(model.state_dict())
    fingerprints = []

    def capture(module, arguments, prediction):
        fields = {
            "coarse_depth": prediction["coarse"]["depth"],
            "coarse_scores": prediction["coarse"]["scores"],
            "coarse_candidates": prediction["coarse"]["candidate_depth"],
            "fine_scores": prediction["fine"]["scores"],
            "fine_winner": prediction["fine"]["selected_index"],
            "fine_valid": prediction["fine"]["valid"],
            "fine_support": prediction["fine"]["support"],
        }
        fingerprints.append(parameter_hash(fields))

    capture_handle = model.register_forward_hook(capture)
    normal = evaluate(model, dataset, plan, device)
    normal_fingerprints = fingerprints.copy()
    fingerprints.clear()
    zero_handle = model.decoder.flow_field.output.register_forward_hook(
        lambda module, arguments, value: torch.zeros_like(value))
    try:
        zero = evaluate(model, dataset, plan, device)
    finally:
        zero_handle.remove()
        capture_handle.remove()
    if fingerprints != normal_fingerprints:
        raise RuntimeError("清零流速度后，原评分/候选身份发生变化")
    # set_range会重建运行时lattice；恢复它再比较整个持久state，禁止豁免学习参数。
    model.decoder.depth_lattice.copy_(original_lattice)
    model.decoder.near, model.decoder.far = original_range
    after = parameter_hash(model.state_dict())
    if after != before or digest(checkpoint) != checkpoint_hash:
        raise RuntimeError("反事实诊断改变了模型持久状态或磁盘权重")
    rows = []
    for split in normal:
        for name, results in normal[split].items():
            for region in ("full", "near"):
                observed = results["depth"][region]
                disabled = zero[split][name]["depth"][region]
                if observed["gt_pixels"] != disabled["gt_pixels"]:
                    raise RuntimeError("比较GT分母不同")
                rows.append({"split": split, "dataset": name, "units": results["units"],
                             "region": region, "frames": results["frames"],
                             "gt_pixels": observed["gt_pixels"],
                             "normal_mae": observed["MAE"], "zero_mae": disabled["MAE"],
                             "normal_rmse": observed["RMSE"], "zero_rmse": disabled["RMSE"],
                             "mae_change_normal_minus_zero": observed["MAE"] - disabled["MAE"],
                             "rmse_change_normal_minus_zero": observed["RMSE"] - disabled["RMSE"]})
    evidence = {
        "completed": True, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "state_hash_before": before, "state_hash_after": after,
        "runtime_lattice_restored_before_final_hash": True,
        "coarse_scores_winner_support_unchanged": True,
        "frames_per_variant": len(fingerprints), "forwards": 2 * len(fingerprints),
        "validation_plan": plan, "rows": rows, "normal": normal, "zero_velocity": zero,
        "test_accessed": False,
        "interpretation": "同一已训练权重的推理干预；与原V2差异仍含旧解码器共同训练，不等于删除流头后重新训练。",
    }
    write_json(output / (folder.name + ".json"), evidence)
    print(json.dumps({key: value for key, value in evidence.items()
                      if key not in ("normal", "zero_velocity", "validation_plan")}, ensure_ascii=False), flush=True)
    del model, record, dataset
    torch.cuda.empty_cache()
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    offline_only()
    seed_all(20261031)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    root = ROOT / "runs/active/ruler_flow_v2_20261031"
    output = root / "readout_probe"
    output.mkdir(parents=True, exist_ok=False)
    reports = [run_phase(root / "pilot" / (phase + "_meanflow"), output, device)
               for phase in ("public", "carla_new")]
    write_json(output / "summary.json", {
        "completed": True, "gpu": args.gpu, "precision": "fp32_tf32_off",
        "frames_per_variant": sum(report["frames_per_variant"] for report in reports),
        "forwards": sum(report["forwards"] for report in reports),
        "all_model_states_preserved": all(report["state_hash_before"] == report["state_hash_after"] for report in reports),
        "all_coarse_scores_winners_preserved": True,
        "test_accessed": False, "rows": [row for report in reports for row in report["rows"]],
        "source_sha256": digest(Path(__file__)),
    })


if __name__ == "__main__":
    main()
