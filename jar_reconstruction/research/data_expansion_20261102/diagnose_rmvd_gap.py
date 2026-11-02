"""固定旧权重定位 courtyard 失效阶段，不训练、不选择测试参数。

内部尺度探针保持实际相机、搜索范围和输出单位不变。粗/细阶段先按整数
特征中心回到主图，再沿用既有原尺寸 GT 评测。刻度 oracle 只作 GT 诊断。
"""

import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from jar.data import build_dataset, collate_frames
from jar.geometry import pixel_grid, sample_maps
from jar.pretraining.comparison import add_statistics, digest, equal_pixel_inputs, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from research.near_mid_evaluation_20261028.metrics import depth_metrics, depth_sums
from research.rulermvs_v6_20261101.evaluation import Models
from tools.compare_public_pretrained import configure
from tools.smoke_outdoor_datasets import offline_only
from rmvd_subset import DATA


def stage_on_image(depth, size, stride):
    coordinates = pixel_grid(*size).reshape(1, -1, 2)
    return sample_maps(depth.cpu(), coordinates, stride=stride,
                       padding_mode="border").transpose(1, 2).reshape(1, 1, *size)


def score_native(prediction, truth):
    prediction = F.interpolate(prediction, truth.shape[-2:], mode="bilinear", align_corners=False)
    result = {}
    for start in range(0, truth.shape[-2], 128):
        section = (..., slice(start, start + 128), slice(None))
        target = truth[section]
        depth = prediction[section]
        valid = torch.isfinite(depth) & (depth > 0)
        mask = torch.isfinite(target) & (target > 0)
        add_statistics(result, depth_sums(depth, valid, target, mask))
    return result


def lattice_oracle(truth, lattice):
    result = {"count": 0, "absolute_sum": 0., "outside_count": 0}
    for start in range(0, truth.shape[-2], 128):
        values = truth[..., start:start + 128, :].numpy()
        values = values[np.isfinite(values) & (values > 0)].astype(np.float64)
        upper = np.searchsorted(lattice, values).clip(0, len(lattice) - 1)
        lower = (upper - 1).clip(0, len(lattice) - 1)
        error = np.minimum(abs(values - lattice[lower]), abs(values - lattice[upper]))
        result["count"] += len(values)
        result["absolute_sum"] += float(error.sum())
        result["outside_count"] += int(((values < lattice[0]) | (values > lattice[-1])).sum())
    return result


def main():
    offline_only()
    previous = json.loads((DATA / "evaluation_results.json").read_text())
    config_path = PROJECT / "runs/active/rulermvs_v6_20261101/full_training/paper_eth3d/evaluation.json"
    config = json.loads(config_path.read_text())
    settings = config["methods"]
    config["methods"] = {"v6_best": settings["v6_best"],
                         "mvsanywhere_firstpass": dict(settings["mvsanywhere_refined"], refinements=0)}
    domain = dict(root=str(DATA), reference_index=0, num_views=4, geometry_units="meters",
                  near=.1, far=100., depth_scale=1., name="ETH3D_courtyard_4views",
                  manifests={"test": str(DATA / "manifests/test_4views.jsonl")})
    device = configure(config, 7)
    models = Models(config, device)
    dataset = build_dataset(dataset_config(domain, config), "test", load_targets=False)
    rows = [json.loads(line) for line in Path(domain["manifests"]["test"]).read_text().splitlines()]
    size = tuple(config["image_size"])
    report = dict(completed=False, training=False, optimizer_steps=0, input_size=size,
                  protocol="固定八组四视图；原尺寸正值GT；尺度探针非参数选优",
                  source_sha256=digest(__file__), original_result_sha256=digest(DATA / "evaluation_results.json"),
                  weights_sha256=previous["weights_sha256"], records=[])
    totals, oracle_total = {}, {"count": 0, "absolute_sum": 0., "outside_count": 0}
    model = models.get("v6_best", domain)
    for index, row in enumerate(rows):
        batch = collate_frames([dataset[index]])
        assert not batch["targets"]
        inputs = move(batch["inputs"], device)
        work = equal_pixel_inputs(inputs)
        with torch.inference_mode():
            output = model(work, domain, reference=0)
            predictions = {"v6_final": output["depth"].cpu() * output["unit_scale"]}
            for stage, stride in (("coarse", 8), ("fine", 4)):
                predictions["v6_" + stage] = stage_on_image(
                    output[stage]["depth"] * output["unit_scale"], size, stride)
            lattice = (model.decoder.depth_lattice * output["unit_scale"]).cpu().double().numpy()
            del output
            shifts = {}
            for scale in (.5, 2.):
                probe = model(work, dict(domain, depth_scale=scale), reference=0)
                key = f"v6_internal_scale_{scale:g}"
                predictions[key] = probe["depth"].cpu() * probe["unit_scale"]
                shifts[key] = float((predictions[key] - predictions["v6_final"]).abs().mean())
                del probe
            prediction = models.forward("mvsanywhere_firstpass", domain, inputs)
            predictions["mvsanywhere_firstpass"] = models.convert(prediction, inputs)[0].cpu()
        # 所有推理完成后才读取本帧GT，不能用于选范围、候选或尺度。
        evaluation = row["evaluation"]
        raw = np.fromfile(evaluation["native_depth"], np.float32).reshape(evaluation["native_shape"])
        truth = torch.from_numpy(raw).view(1, 1, *raw.shape)
        record = dict(source_sample=row["source_sample"], metrics={}, internal_scale_output_shift=shifts)
        for name, depth in predictions.items():
            sums = score_native(depth, truth)
            add_statistics(totals.setdefault(name, {}), sums)
            record["metrics"][name] = depth_metrics(sums)
        oracle = lattice_oracle(truth, lattice)
        for key, value in oracle.items():
            oracle_total[key] += value
        record["global_lattice_oracle"] = oracle
        original = previous["datasets"][domain["name"]]["records"][index]["methods"]["v6_best"]["MAE"]
        record["prior_v6_mae_difference"] = record["metrics"]["v6_final"]["MAE"] - original
        if abs(record["prior_v6_mae_difference"]) > 1e-5:
            raise RuntimeError("固定V6复跑与原结果不一致，不能继续解释新诊断")
        report["records"].append(record)
        write_json(DATA / "gap_diagnosis_progress.json", report)
        print(index, {key: round(value["MAE"], 5) for key, value in record["metrics"].items()}, flush=True)
    report["metrics"] = {name: depth_metrics(sums) for name, sums in totals.items()}
    report["global_lattice_oracle"] = dict(
        **oracle_total, MAE=oracle_total["absolute_sum"] / oracle_total["count"],
        note="GT事后逐像素选最近全局刻度，忽略图像证据和空间网格；不是可部署模型精度")
    report["completed"] = True
    write_json(DATA / "gap_diagnosis.json", report)


if __name__ == "__main__":
    main()
