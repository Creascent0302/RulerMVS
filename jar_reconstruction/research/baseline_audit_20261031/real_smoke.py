"""真实VAL接口冒烟：显式允许无任务权重的结构检查，绝不计算精度或计时排名。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from official_methods.data import FourCameraDataset, collate
from official_methods.postprocess import depth_on_image_grid
from official_methods.registry import build_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    suite = ROOT / "runs/active/baseline_audit_20261031/suite"
    records = []
    for path in sorted((suite / "configs" / args.method).glob("*.json")):
        config = json.loads(path.read_text())
        config["model"]["allow_random_init"] = True
        # 此入口明确为随机结构/已获权重的接口测试；不能作为evaluate入口使用。
        torch.manual_seed(1701)
        model = build_adapter(config).eval()
        dataset = FourCameraDataset(config, "val", load_targets=False)
        batch = collate([dataset[0]], expected_views=config["data"]["num_views"])
        assert batch["targets"] == {}
        with torch.no_grad():
            prediction = model(batch["inputs"])
            assert torch.isfinite(prediction["depth"]).all()
            converted = depth_on_image_grid(prediction, batch["inputs"])
        records.append({
            "dataset": config["dataset_name"], "split": "val", "sample": batch["metadata"],
            "input_shape": list(batch["inputs"]["rgb"].shape),
            "native_depth_shape": list(prediction["depth"].shape),
            "common_depth_shape": list(converted["depth"].shape),
            "finite": True, "declared_valid_fraction": float(converted["depth_valid"].float().mean()),
            "task_weights": config["model"].get("weights", ""),
            "geometry_units": config["data"]["geometry_units"], "labels_loaded": False,
            "performance_claim": "无；单帧数据/结构接口测试，不是精度或速度结果",
        })
        del model, prediction, converted, dataset, batch
        print(args.method, config["dataset_name"], "PASS", flush=True)
    if len(records) != 8:
        raise RuntimeError("当前套件应包含8个数据域；请先运行prepare.py")
    output = suite / f"{args.method}_val_inputs.json"
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
