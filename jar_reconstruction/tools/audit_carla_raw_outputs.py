"""对冻结模型单独审计原始输出；不加载GT、不重算成绩、不覆盖已有评价。"""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.carla_comparison import comparison_dataset, read_json
from jar.data import collate_frames
from jar.raw_output_audit import aggregate_raw_audits, raw_output_audit
from jar.runtime import move, seed_all
from tools.carla_compare import file_sha256, make_model, resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    experiment = read_json(args.config)
    protocol = read_json(experiment["protocol"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "frames.jsonl").exists() or (output / "raw_audit.json").exists():
        raise FileExistsError("请使用新目录，保留已有原始输出审计。")
    torch.set_num_threads(protocol["runtime"].get("threads", 2))
    seed_all(experiment.get("seed", 17))
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    weights = experiment.get("weights", "")
    model, model_config = make_model(experiment, weights, device)
    dataset = comparison_dataset(protocol, experiment["split"], load_targets=False)
    indices = experiment.get("frame_indices", list(range(len(dataset))))
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(dataset) for i in indices):
        raise ValueError("frame_indices必须为不重复的有效帧索引。")
    audits = []
    with torch.inference_mode(), (output / "frames.jsonl").open("w") as stream:
        for index in indices:
            batch = collate_frames([dataset[index]])
            if batch["targets"]:
                raise RuntimeError("仅审计入口不应加载任何监督标签。")
            inputs = move(batch["inputs"], device)
            prediction = model(inputs)
            audit = raw_output_audit(prediction)
            frame = dataset.frames[index]
            row = {
                "sample_id": f"{frame['sequence']}/{frame['frame_id']}",
                "dataset_index": index,
                "valid_camera_count": int(inputs["view_mask"].sum()),
                "raw_output_audit": audit,
            }
            audits.append(audit)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            print(
                f"{experiment['method']} {len(audits)}/{len(indices)} raw_nonfinite={audit['nonfinite_elements']}",
                flush=True,
            )
            del prediction, inputs, batch
    report = {
        "experiment": experiment,
        "device": str(device),
        "model_config": model_config,
        "checkpoint": weights,
        "checkpoint_sha256": file_sha256(weights) if weights else None,
        "pretrained_model_weights": model_config.get("model", {}).get("weights", ""),
        "manifest_sha256": file_sha256(protocol["data"]["manifests"][experiment["split"]]),
        "full_manifest": len(indices) == len(dataset),
        "frame_count": len(audits),
        "raw_output_audit": aggregate_raw_audits(audits),
        "source_sha256": {
            path: file_sha256(path)
            for path in ("tools/audit_carla_raw_outputs.py", "jar/raw_output_audit.py")
        },
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "note": "只重跑冻结模型的RGB+标定推理并统计原始数值有限性；未加载GT，未测精度、未更改已有成绩或权重。",
    }
    (output / "raw_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(output / "raw_audit.json")


if __name__ == "__main__":
    main()
