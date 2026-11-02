"""用现有公共数据加载器读取每套 train/val/test 的首个样本，不运行模型。"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jar.data import build_dataset, collate_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.config).read_text())
    torch.set_num_threads(1)
    report = {"scope": "每数据集每划分首样本的CPU加载与标签隔离；未训练/未运行网络", "samples": []}
    for entry in plan["datasets"]:
        config = json.loads(Path(entry["own_config"]).read_text())
        config["data"]["root"] = entry["root"]
        config["data"].update({f"{split}_manifest": path for split, path in entry["manifests"].items()})
        for split in ("train", "val", "test"):
            dataset = build_dataset(config, split)
            sample = dataset[0]
            inputs_only = build_dataset(config, split, load_targets=False)[0]
            for key, value in sample["inputs"].items():
                if not torch.equal(value, inputs_only["inputs"][key]):
                    raise AssertionError(f"标签影响输入：{entry['name']}/{split}/{key}")
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise AssertionError(f"非有限输入：{entry['name']}/{split}/{key}")
            if inputs_only["targets"]:
                raise AssertionError("load_targets=False仍返回了标签")
            batch = collate_frames([sample])
            depth = batch["targets"]["depth"]
            valid = batch["targets"]["depth_valid"]
            if not valid.any() or not torch.isfinite(depth[valid]).all():
                raise AssertionError(f"没有有限深度监督：{entry['name']}/{split}")
            record = {
                "dataset": entry["name"], "split": split, "length": len(dataset),
                "geometry_units": entry["geometry_units"],
                "reference_index": entry["reference_index"],
                "inputs": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                           for key, value in batch["inputs"].items() if torch.is_tensor(value)},
                "depth_shape": list(depth.shape),
                "valid_depth_fraction": float(valid.float().mean()),
                "targets_removed_inputs_identical": True,
            }
            report["samples"].append(record)
            print(entry["name"], split, record["inputs"]["rgb"]["shape"], flush=True)
    path = Path(plan["storage_root"]) / "loader_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Audit:", path)


if __name__ == "__main__":
    main()
