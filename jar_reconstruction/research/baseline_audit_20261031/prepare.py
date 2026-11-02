"""为六种官方深度方法生成当前八域的共同完整输入配置，不读取深度值。"""

import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from official_methods.data import FourCameraDataset, audit_split_sequences, collate

METHODS = ("fast_mvsnet", "simplerecon", "mvsanywhere", "mvsformerpp", "patchmatchnet", "diffmvs")
OUTPUT = ROOT / "runs/active/baseline_audit_20261031/suite"


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


def prepare_domain(domain):
    records, manifests = {}, {}
    for split, source in domain["manifests"].items():
        source = Path(source)
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        complete, excluded = [], []
        for row in rows:
            cameras = row["cameras"]
            valid = len(cameras) == domain["num_views"] and all(
                camera.get("rgb") and camera.get("valid", True) for camera in cameras
            )
            if valid:
                complete.append(row)
            else:
                excluded.append([row["sequence"], row["frame_id"]])
        if not complete:
            raise ValueError(f"{domain['name']}/{split}没有完整的真实输入")
        target = OUTPUT / "manifests" / domain["name"] / f"{split}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in complete))
        manifests[split] = str(target)
        records[split] = {
            "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "original_count": len(rows), "complete_count": len(complete), "excluded": excluded,
            "complete_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    return manifests, records


def main():
    domains = []
    for name in ("public", "carla"):
        source = ROOT / f"research/v2_evaluation_20261029/{name}.json"
        domains.extend(json.loads(source.read_text())["datasets"])
    report = {"selection": "仅根据真实RGB是否存在，不用GT深度值选择输入或范围", "datasets": {}}
    for domain in domains:
        manifests, records = prepare_domain(domain)
        settings = {
            "root": domain["root"], "num_views": domain["num_views"],
            "geometry_units": domain["geometry_units"],
            **{f"{split}_manifest": path for split, path in manifests.items()},
        }
        audit_split_sequences({"data": settings})
        configs = {}
        for method in METHODS:
            path = ROOT / f"official_methods/methods/{method}/configs/carla.json"
            config = json.loads(path.read_text())
            config.update(data=copy.deepcopy(settings), image_size=[256, 448], cpu_threads=2)
            options = config["model"]
            options.update(
                num_views=domain["num_views"], image_height=256, image_width=448,
                reference_indices=[domain["reference_index"]],
                depth_range=[domain["near"], domain["far"]],
                min_depth=domain["near"], max_depth=domain["far"],
            )
            evaluation = config.setdefault("evaluation", {})
            evaluation.update(reference_indices=[domain["reference_index"]],
                              depth_range=domain.get("eval_range", [domain["near"], domain["far"]]))
            evaluation.pop("roi", None)  # 城市场景的固定盒子不适用于另外七域。
            config["data"]["supervision_depth_range"] = evaluation["depth_range"]
            config["dataset_name"] = domain["name"]
            config["protocol_note"] = [
                "同一真实输入子集、256x448主图、同参考号，不复制视图或使用GT范围。",
                "范围来自冻结的TRAIN配置。共同数据迁移配置不是原论文原数据训练协议。",
                "SimpleRecon固定N的metadata MLP不可跨N共享任务权重。",
                "大部分配置未填写任务权重，正式infer/evaluate将报错而非输出随机模型指标。",
            ]
            if domain["geometry_units"] == "scene_unit":
                evaluation.pop("surface_thresholds_m", None)
                evaluation["surface_thresholds_scene_unit"] = [0.01, 0.05, 0.1]
            if method == "fast_mvsnet":
                options["weights"] = str(ROOT / "official_methods/methods/fast_mvsnet/upstream/outputs/pretrained.pth")
            elif method == "patchmatchnet":
                options["weights"] = str(ROOT / "official_methods/methods/patchmatchnet/upstream/checkpoints/model_000007.ckpt")
            elif method == "mvsanywhere":
                options["weights"] = str(ROOT / "runs/carla_research_20261011/cache/mvsanywhere_hero.ckpt")
            target = OUTPUT / "configs" / method / f"{domain['name']}.json"
            save(target, config)
            configs[method] = str(target)
        # 真实VAL文件仅检验接口；训练/TEST标签值不打开，绝不据此调范围。
        probe = json.loads(Path(configs["patchmatchnet"]).read_text())
        dataset = FourCameraDataset(probe, "val", load_targets=False)
        batch = collate([dataset[0]], expected_views=domain["num_views"])
        report["datasets"][domain["name"]] = {
            "units": domain["geometry_units"], "views": domain["num_views"],
            "reference": domain["reference_index"], "splits": records, "configs": configs,
            "val_inputs_shape": list(batch["inputs"]["rgb"].shape),
            "val_targets_empty": batch["targets"] == {},
        }
    save(OUTPUT / "preparation.json", report)
    print(OUTPUT / "preparation.json")


if __name__ == "__main__":
    main()
