"""四个真实 TRAIN 样本的GPU再生成与可视化，仅检查增强几何。"""

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from jar.pretraining.comparison import digest, equal_pixel_inputs, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.runtime import move
from research.v7_domain_20261102.synthetic_pair import correspondence_targets, synthesize_pair


def geometry_audit(inputs, supervision, sample):
    reference = supervision["reference_index"]
    slot = supervision["source_slot"]
    valid = supervision["source_depth_valid"][sample]
    depth = supervision["source_depth"][sample].double()[valid]
    uv = supervision["projected_uv"][sample].double()[:, valid]
    homogeneous = torch.cat((uv, torch.ones_like(uv[:1])), 0)
    point = torch.linalg.solve(inputs["K"][sample, slot].double(), homogeneous) * depth
    source_transform = inputs["T_cam_from_rig"][sample, slot].double()
    reference_transform = inputs["T_cam_from_rig"][sample, reference].double()
    relative = source_transform @ torch.linalg.inv(reference_transform)
    recovered = relative[:3, :3].T @ (point - relative[:3, 3:4])
    projected = inputs["K"][sample, reference].double() @ recovered
    projected = projected[:2] / projected[2:3]
    index = supervision["reference_pixel_index"][sample][valid]
    height, width = valid.shape
    original = torch.stack((index % width, index // width)).double()
    y, x = torch.meshgrid(torch.arange(height, device=valid.device),
                          torch.arange(width, device=valid.device), indexing="ij")
    source_pixels = torch.stack((x, y)).double()[:, valid]
    return {
        "valid_source_pixels": int(valid.sum()),
        "continuous_roundtrip_max_pixels": float((projected - original).norm(dim=0).max()),
        "raster_quantization_max_pixels": float((uv - source_pixels).norm(dim=0).max()),
        "relative_rotation_max_difference": float((relative[:3, :3] - torch.eye(3, device=valid.device)).abs().max()),
        "relative_translation": relative[:3, 3].tolist(),
    }


def figure(path, original, augmented, target, supervision, sample, title):
    ref = original["rgb"][sample, 0].detach().cpu().permute(1, 2, 0)
    source = augmented["rgb"][sample, 1].detach().cpu().permute(1, 2, 0)
    depth = target["depth"][sample, 0].detach().cpu()
    mask = target["depth_valid"][sample, 0].detach().cpu()
    limit = float(torch.quantile(depth[mask], .95))
    source_depth = supervision["source_depth"][sample].detach().cpu()
    source_mask = supervision["source_depth_valid"][sample].detach().cpu()
    panels = [ref, source, source_mask, torch.where(mask, depth, float("nan")),
              torch.where(source_mask, source_depth, float("nan")),
              (~source_mask).float()]
    titles = ["TRAIN reference RGB", "Virtual short-baseline source", "Source valid mask",
              "Original reference Z-depth", "Splat source Z-depth", "Unknown holes (white)"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.5))
    for index, (axis, panel, name) in enumerate(zip(axes.flat, panels, titles)):
        options = {"cmap": "gray", "vmin": 0., "vmax": 1.} if index in (2, 5) else {}
        if index in (3, 4):
            options = {"cmap": "viridis", "vmin": 0., "vmax": limit}
        axis.imshow(panel, **options)
        axis.set_title(name, fontsize=10)
        axis.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    output = ROOT / "runs/active/v7_domain_20261102/synthetic_pair_smoke"
    if not output.resolve().is_relative_to("/data"):
        raise ValueError("合成产物须位于数据盘")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    device = torch.device("cuda:7")
    torch.cuda.set_device(device)
    records = []
    generator = torch.Generator().manual_seed(202611027)
    for name in ("public", "carla"):
        config_path = ROOT / f"research/rulermvs_v6_20261101/configs/data_{name}.json"
        config = json.loads(config_path.read_text())
        config["datasets"] = config["datasets"][:1]
        config.pop("validation_datasets", None)
        config.update(seed=20261102, augmentation=False)
        config["datasets"][0]["manifests"] = {
            "train": config["datasets"][0]["manifests"]["train"]}
        data = MixedDataset(config, "train")
        batch = move(collate_mixed([data[(0, index, -1)] for index in (0, 1)]), device)
        inputs = equal_pixel_inputs(batch["inputs"])
        inputs_before = {key: value.clone() for key, value in inputs.items() if torch.is_tensor(value)}
        target_before = {key: value.clone() for key, value in batch["targets"].items() if torch.is_tensor(value)}
        augmented, supervision, statistics = synthesize_pair(
            inputs, batch["targets"], data.domains[0], generator=generator)
        target_new = correspondence_targets(batch["targets"], supervision)
        assert all(torch.equal(value, inputs[key]) for key, value in inputs_before.items())
        assert all(torch.equal(value, batch["targets"][key]) for key, value in target_before.items())
        assert torch.equal(target_new["depth"][:, 0], batch["targets"]["depth"][:, 0])
        assert torch.equal(target_new["depth_valid"][:, 0], batch["targets"]["depth_valid"][:, 0])
        assert torch.isfinite(augmented["rgb"]).all()
        assert torch.isfinite(augmented["matching_rgb"]).all()
        for sample in range(2):
            audit = geometry_audit(augmented, supervision, sample)
            if audit["continuous_roundtrip_max_pixels"] > 2e-3:
                raise AssertionError(audit)
            if audit["raster_quantization_max_pixels"] > 2 ** -.5 + 1e-4:
                raise AssertionError(audit)
            baseline = float(statistics["baseline_data_units"][sample])
            title = f"{data.domains[0]['name']} TRAIN index {sample}; baseline {baseline:.4f} data units"
            image_path = output / f"{name}_{sample}.png"
            figure(image_path, inputs, augmented, batch["targets"], supervision, sample, title)
            row = {
                "dataset": data.domains[0]["name"], "split": "train", "index": sample,
                "metadata": batch["metadata"][sample], "geometry_units": data.domains[0]["geometry_units"],
                "manifest_sha256": digest(data.domains[0]["manifests"]["train"]),
                "source_coverage": float(statistics["source_coverage"][sample]),
                "collision_count": int(statistics["collisions"][sample]),
                "baseline_data_units": baseline,
                "baseline_normalized": float(statistics["baseline_normalized"][sample]),
                "retained_views": int(statistics["retained_views"][sample]),
                "geometry_audit": audit, "image": str(image_path),
            }
            records.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    torch.cuda.synchronize(device)
    sources = [Path(__file__), Path(__file__).with_name("synthetic_pair.py"),
               Path(__file__).with_name("test_synthetic_pair.py")]
    write_json(output / "completed.json", {
        "completed": True, "device": str(device), "GPU": torch.cuda.get_device_name(device),
        "samples": records, "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "test_accessed": False, "model_inference_run": False,
        "source_targets_require_explicit_mapping": True,
        "limits": ["仅点前向投放几何检查，不是新架构性能或真实植被生成。",
                   "最近栅格化存在亚像素量化；洞不填充，不作为已观察空闲空间。",
                   "参考RGB和GT保留，新源由TRAIN标签生成，绝不能用于评测输入。"],
    })


if __name__ == "__main__":
    main()
