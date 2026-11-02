"""只读清单审计：检查所有源图的路径交叉，不把不同路径等同不同内容。"""

import hashlib
import json
from functools import lru_cache
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "research/rulermvs_v6_20261101/configs"
RUN = ROOT / "runs/active/rulermvs_v6_20261101/full_training"


def read_json(path):
    return json.loads(path.read_text())


@lru_cache(maxsize=None)
def canonical(root, value):
    path = Path(value)
    if not path.is_absolute():
        path = Path(root) / path
    return str(path.resolve())


def inspect(manifest, root):
    path = Path(manifest)
    images = set()
    sequences = set()
    samples = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        sequence = str(item["sequence"])
        sequences.add(sequence)
        samples.add((sequence, str(item["frame_id"])))
        for camera in item["cameras"]:
            if camera.get("rgb"):
                images.add(canonical(root, camera["rgb"]))
    return {
        "manifest": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "images": images,
        "sequences": sequences,
        "samples": samples,
    }


def main():
    output = RUN / "paper_split_audit"
    output.mkdir(exist_ok=True)
    results = {}
    for phase, group in (("public", "public_main"), ("carla_new", "carla_new_main")):
        training = read_json(TRAINING / f"train_{phase}.json")["datasets"]
        evaluation = read_json(RUN / f"evaluation_recovery/configs/{group}_test.json")
        for domain in evaluation["datasets"]:
            original = training[0] if phase == "carla_new" else next(
                item for item in training if item["name"] == domain["name"]
            )
            manifests = {
                "train_actual": original["manifests"]["train"],
                "val_train_config": original["manifests"]["val"],
                "val_common": domain["manifests"]["val"],
                "test_common": domain["manifests"]["test"],
            }
            splits = {name: inspect(path, domain["root"]) for name, path in manifests.items()}
            comparisons = {}
            for left, right in combinations(splits, 2):
                a, b = splits[left], splits[right]
                comparisons[f"{left}__{right}"] = {
                    "shared_rgb_paths": len(a["images"] & b["images"]),
                    "shared_sample_ids": len(a["samples"] & b["samples"]),
                    "shared_sequences": sorted(a["sequences"] & b["sequences"]),
                    "shared_rgb_examples": sorted(a["images"] & b["images"])[:5],
                }
            results[domain["name"]] = {
                "splits": {
                    name: {
                        "manifest": item["manifest"],
                        "sha256": item["sha256"],
                        "unique_rgb_paths": len(item["images"]),
                        "unique_sample_ids": len(item["samples"]),
                        "sequences": sorted(item["sequences"]),
                    }
                    for name, item in splits.items()
                },
                "comparisons": comparisons,
            }
    result = {
        "completed": True,
        "scope": "Actual current training manifests and frozen common evaluation manifests; all listed cameras",
        "limitations": [
            "只验证路径/样本ID/sequence交叉，没有读取RGB内容作哈希或近重复检测。",
            "不同sequence不保证不同物理世界；同地图不同轨迹不等于跨地图泛化。",
            "CARLA训练代码会扩展有标签参考；这里检查原采集所有源图，覆盖该扩展的图像来源。",
            "未审计通用预训练或baseline原始预训练的数据交叉。",
        ],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "datasets": results,
    }
    (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    lines = [
        "# 当前TRAIN与固定TEST清单的路径交叉审计", "",
        "检查所有源图而非仅参考图；包含真实训练清单中的缺视图采集。没有修改任何划分。", "",
        "|数据域|TRAIN采集/组|共同VAL|共同TEST|TRAIN/TEST共享RGB路径|TRAIN/TEST共享sequence|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in results.items():
        splits = item["splits"]
        pair = item["comparisons"]["train_actual__test_common"]
        counts = [splits[key]["unique_sample_ids"] for key in ("train_actual", "val_common", "test_common")]
        lines.append(f"|{name}|{counts[0]}|{counts[1]}|{counts[2]}|{pair['shared_rgb_paths']}|{len(pair['shared_sequences'])}|")
    lines.extend(["", *[f"- {note}" for note in result["limitations"]]])
    (output / "清单审计.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
