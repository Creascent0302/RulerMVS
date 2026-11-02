"""锁定完成训练的权重、完整共同清单和评测协议，不按TEST选模型。"""

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest, write_json

HERE = Path(__file__).parent
OUTPUT = ROOT / "runs/active/baseline_full_comparison_20261101"
OLD = ROOT / "runs/active/v5_comparison_20261101"


def read(path):
    return json.loads(Path(path).read_text())


def main():
    config = read(ROOT / "research/v5_comparison_20261101/configs/carla_complete_equal_pixels.json")
    config["methods"] = {key: value for key, value in config["methods"].items()
                         if key in ("v5_best", "original_v2", "mvsanywhere_refined")}
    training = {}
    for method in ("patchmatchnet", "diffmvs"):
        folder = OLD / f"{method}_adapt"
        if not read(folder / "completed.json")["completed"]:
            raise RuntimeError(f"{method}训练/首轮TEST未完成")
        original_audit = read(folder / "audit.json")
        for source, expected in original_audit["source_sha256"].items():
            path = (OUTPUT / "training_validation_before_bn_fix.py"
                    if source == "official_methods/training_validation.py" else ROOT / source)
            if digest(path) != expected:
                raise RuntimeError(f"训练源码证据变化：{source}")
        checkpoints = {kind: torch.load(folder / f"training/{kind}.pt", map_location="cpu", weights_only=False)
                       for kind in ("best", "final")}
        if checkpoints["final"]["step"] != 384:
            raise RuntimeError("训练未达计划384步")
        same = all(torch.equal(value, checkpoints["final"]["model"][key])
                   for key, value in checkpoints["best"]["model"].items())
        buffers = checkpoints["best"]["state_by_rank"]
        different = [key for key, value in buffers[0]["buffers"].items()
                     if any(not torch.equal(value, rank["buffers"][key]) for rank in buffers[1:])]
        training[method] = {
            "completion": read(folder / "completed.json"),
            "training_report": read(folder / "training/train_report.json"),
            "best_final_parameters_equal": same,
            "best_step": checkpoints["best"]["step"],
            "different_rank_buffers": different,
            "selection_limitation": "旧DDP验证混合各rank的BN统计；本轮重新验证已保存的实际checkpoint，不将其称为所有历史迭代的最优模型。",
        }
        for kind in ("best", "final"):
            config["methods"][f"{method}_{kind}"] = {
                "kind": method, "weights": str(folder / f"training/{kind}.pt"),
                "project_checkpoint": True, "model": {"variant": method}, "sample_seed": 1701,
            }
    write_json(OUTPUT / "training_audit.json", training)

    sources = [HERE / "evaluate.py", ROOT / "research/v5_comparison_20261101/evaluate.py"]
    for method in ("patchmatchnet", "diffmvs"):
        sources.extend((ROOT / f"official_methods/methods/{method}").rglob("*.py"))
    config["extension_source_sha256"].update({str(p.relative_to(ROOT)): digest(p) for p in sources})
    config.update(seed=20261101, gpus=list(range(8)), shard_size=32, loader_workers=1)
    config["protocol_notes"] += [
        "完整共同四目；两新增方法固定逐RGB+K+T+参考号+seed的原生随机流，不依赖GPU分片顺序。",
        "主表保留原训练best命名，但其旧VAL混合BN；同时独立报告final和本次完整VAL，不按TEST换权重。",
        "新增baseline仅CARLA403帧先导训练，V2/V5有公共预训练，Hero官方多域预训练；非同预算架构排名。",
    ]
    upper = copy.deepcopy(config)
    upper.update(output=str(OUTPUT / "validation"), evaluation_split="val")
    upper["methods"] = {key: value for key, value in upper["methods"].items()
                        if value["kind"] in ("patchmatchnet", "diffmvs")}
    write_json(HERE / "configs/validation.json", upper)

    others = read(ROOT / "runs/active/rulermvs_v5_20261031/evaluation/configs/carla_other_refs_equal_pixels.json")
    counts = {}
    for domain in others["datasets"]:
        domain = copy.deepcopy(domain)
        rows = [json.loads(line) for line in Path(domain["manifests"]["test"]).read_text().splitlines()]
        good = [row for row in rows if len(row["cameras"]) == 4 and all(
            camera.get("rgb") and camera.get("valid", True) for camera in row["cameras"])]
        filename = OUTPUT / "manifests" / (domain["name"] + "_test.jsonl")
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in good))
        counts[domain["name"]] = {"original": len(rows), "complete": len(good), "manifest_sha256": digest(filename)}
        domain["manifests"]["test"] = str(filename)
        config["datasets"].append(domain)
    config.update(output=str(OUTPUT / "carla_four_references"), evaluation_split="test")
    write_json(HERE / "configs/carla.json", config)
    write_json(OUTPUT / "subset_audit.json", counts)

    repeats = copy.deepcopy(upper)
    repeats.update(output=str(OUTPUT / "stochastic_repeats"), evaluation_split="test")
    repeats["methods"] = {}
    for method in ("patchmatchnet", "diffmvs"):
        for seed in (2701, 3701):
            repeats["methods"][f"{method}_seed{seed}"] = dict(config["methods"][method + "_best"], sample_seed=seed)
    write_json(HERE / "configs/repeats.json", repeats)

    transfer = read(ROOT / "research/v5_comparison_20261101/configs/patchmatchnet_public.json")
    transfer.update(output=str(OUTPUT / "public_carla_transfer"), gpus=list(range(8)), evaluation_split="test")
    transfer["methods"] = {key: value for key, value in config["methods"].items()
                           if key in ("patchmatchnet_best", "diffmvs_best")}
    transfer["extension_source_sha256"] = config["extension_source_sha256"]
    transfer["protocol_notes"] = config["protocol_notes"] + ["此组仅测CARLA适配模型跨域迁移，不能混称公共预训练模型的同预算实验。"]
    write_json(HERE / "configs/public_transfer.json", transfer)

    timing_data = copy.deepcopy(config)
    timing_data["datasets"] = timing_data["datasets"][:1]
    timing_data["methods"] = {key: value for key, value in config["methods"].items() if not key.endswith("_final")}
    # 兼容上一轮四参考测速器的占位配置选择；进入测速即pop，并非新增模型。
    timing_data["methods"]["v5_final"] = copy.deepcopy(timing_data["methods"]["v5_best"])
    write_json(HERE / "configs/speed_data.json", timing_data)
    write_json(HERE / "configs/four_speed.json", {
        "data_config": str(HERE / "configs/speed_data.json"),
        "output": str(OUTPUT / "four_reference_latency.json"), "warmup": 10, "repeats": 40,
    })
    timing = {"data_config": str(HERE / "configs/speed_data.json"), "output": str(OUTPUT / "single_latency"),
              "own_weights": config["own_weights"], "gpu": 0, "cpu_threads": 2, "seed": 20261101,
              "warmup": 10, "repetitions": 40, "position_cache": True, "fast_cost_volume": True,
              "isolated_resources": True,
              "methods": {key: value for key, value in timing_data["methods"].items() if key != "v5_final"},
              "protocol_notes": config["protocol_notes"], "extension_source_sha256": config["extension_source_sha256"]}
    write_json(HERE / "configs/single_speed.json", timing)
    print(json.dumps({"training": {k: v['best_final_parameters_equal'] for k, v in training.items()}, "subsets": counts}))


if __name__ == "__main__":
    main()
