"""四个TRAIN样本的双线性增强GPU检查；不运行模型或读取TEST。"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest, equal_pixel_inputs, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.runtime import move
from research.v7_domain_20261102.smoke_synthetic_pair import figure
from research.v7_domain_20261102.synthetic_pair import forward_splat as nearest_splat
from research.v7_domain_20261102.synthetic_pair_bilinear import (
    augment_training_batch, correspondence_targets, forward_splat, synthesize_pair,
)
from research.v7_domain_20261102.test_synthetic_pair import scene


def mapping_audit(inputs, reference_rgb, supervision, sample):
    mapping = supervision["mapping"]
    weight = mapping["weight"][sample].double()
    keep = weight > 0
    src_index = mapping["source_index"][sample]
    ref_index = mapping["reference_index"][sample]
    height, width = supervision["source_depth"][sample].shape
    sums = torch.zeros(height * width, dtype=torch.double, device=weight.device).scatter_add_(0, src_index, weight)
    valid = supervision["source_depth_valid"][sample].flatten()
    inverse = torch.zeros_like(sums).scatter_add_(0, src_index, weight / mapping["source_depth"][sample].double())
    expected_z = inverse[valid].reciprocal()
    actual_z = supervision["source_depth"][sample].double().flatten()[valid]
    reference = supervision["reference_index"]
    slot = supervision["source_slot"]
    intrinsic = inputs["K"][sample, reference].double()
    source_transform = inputs["T_cam_from_rig"][sample, slot].double()
    reference_transform = inputs["T_cam_from_rig"][sample, reference].double()
    relative = source_transform @ torch.linalg.inv(reference_transform)
    original_uv = torch.stack((ref_index[keep] % width, ref_index[keep] // width)).double()
    source_uv = torch.stack((mapping["projected_u"][sample, keep].double(), original_uv[1]))
    homogeneous = torch.cat((source_uv, torch.ones_like(source_uv[:1])), 0)
    points = torch.linalg.solve(intrinsic, homogeneous) * mapping["source_depth"][sample, keep].double()
    recovered = relative[:3, :3].T @ (points - relative[:3, 3:4])
    projected = intrinsic @ recovered
    recovered_uv = projected[:2] / projected[2:3]
    centroid = torch.zeros_like(sums).scatter_add_(0, src_index,
                                                   weight * mapping["projected_u"][sample].double())
    x = torch.arange(height * width, device=weight.device) % width
    minimum = torch.full_like(sums, torch.inf).scatter_reduce_(
        0, src_index, torch.where(keep, mapping["source_depth"][sample].double(), torch.inf), reduce="amin")
    maximum = torch.zeros_like(sums).scatter_reduce_(
        0, src_index, torch.where(keep, mapping["source_depth"][sample].double(), 0.), reduce="amax")
    colours = reference_rgb[sample].double().flatten(1)[:, ref_index]
    reconstructed = torch.zeros(3, height * width, device=weight.device, dtype=torch.double)
    reconstructed.scatter_add_(1, src_index[None].expand(3, -1), colours * weight[None])
    actual_rgb = inputs["rgb"][sample, slot].double().flatten(1)
    return {
        "normalization_max_error": float((sums[valid] - 1.).abs().max()),
        "invalid_contribution_mass_max": float(sums[~valid].abs().max()) if (~valid).any() else 0.,
        "mapping_colour_max_error": float((reconstructed - actual_rgb).abs().max()),
        "mapping_depth_relative_max_error": float(((actual_z - expected_z).abs() / expected_z).max()),
        "continuous_contribution_roundtrip_max_pixels": float((recovered_uv - original_uv).norm(dim=0).max()),
        "raster_centroid_max_pixels": float((centroid - x)[valid].abs().max()),
        "within_surface_depth_spread_max_fraction": float((maximum[valid] / minimum[valid] - 1.).max()),
        "valid_source_pixels": int(valid.sum()),
    }


def analytic_test(device):
    inputs, targets = scene()
    inputs, targets = move(inputs, device), move(targets, device)
    rgb = inputs["rgb"][:, 0]
    depth = torch.full_like(targets["depth"][:, 0], 8.)
    rows = []
    for delta in (.25, .5, .75):
        shift = torch.full((2,), delta * 8. / 32., device=device)
        output = forward_splat(rgb, depth, targets["depth_valid"][:, 0], inputs["K"][:, 0], shift)
        nearest = nearest_splat(rgb, depth, targets["depth_valid"][:, 0], inputs["K"][:, 0], shift)
        expected = (1. - delta) * rgb[..., 1:] + delta * rgb[..., :-1]
        valid = output["valid"][..., 1:] & nearest["valid"][..., 1:]
        new_error = (output["rgb"][..., 1:] - expected).abs().permute(0, 2, 3, 1)[valid].mean()
        old_error = (nearest["rgb"][..., 1:] - expected).abs().permute(0, 2, 3, 1)[valid].mean()
        assert new_error < 1e-6 and new_error < old_error * .01
        rows.append({"displacement_pixels": delta, "bilinear_rgb_MAE": float(new_error),
                     "nearest_rgb_MAE": float(old_error), "common_pixels": int(valid.sum())})
    return rows


def main():
    output = ROOT / "runs/active/v7_domain_20261102/synthetic_pair_bilinear_smoke"
    if not output.resolve().is_relative_to("/data"):
        raise ValueError("结果必须存入数据盘")
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:7")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    analytic = analytic_test(device)
    generator = torch.Generator().manual_seed(202611027)
    records = []
    for phase in ("public", "carla"):
        path = ROOT / f"research/rulermvs_v6_20261101/configs/data_{phase}.json"
        config = json.loads(path.read_text())
        config["datasets"] = config["datasets"][:1]
        config.pop("validation_datasets", None)
        config.update(seed=20261102, augmentation=False)
        config["datasets"][0]["manifests"] = {"train": config["datasets"][0]["manifests"]["train"]}
        data = MixedDataset(config, "train")
        batch = move(collate_mixed([data[(0, index, -1)] for index in (0, 1)]), device)
        inputs = equal_pixel_inputs(batch["inputs"])
        before_inputs = {k: v.clone() for k, v in inputs.items() if torch.is_tensor(v)}
        before_targets = {k: v.clone() for k, v in batch["targets"].items() if torch.is_tensor(v)}
        augmented, supervision, statistics = synthesize_pair(
            inputs, batch["targets"], data.domains[0], generator=generator)
        updated = correspondence_targets(batch["targets"], supervision)
        assert all(torch.equal(v, inputs[k]) for k, v in before_inputs.items())
        assert all(torch.equal(v, batch["targets"][k]) for k, v in before_targets.items())
        assert torch.equal(updated["depth"][:, 0], batch["targets"]["depth"][:, 0])
        assert torch.isfinite(augmented["matching_rgb"]).all()
        for sample in range(2):
            audit = mapping_audit(augmented, inputs["rgb"][:, 0], supervision, sample)
            assert audit["continuous_contribution_roundtrip_max_pixels"] < 2e-3, audit
            assert audit["normalization_max_error"] < 1e-5, audit
            assert audit["raster_centroid_max_pixels"] < .2501, audit
            assert audit["within_surface_depth_spread_max_fraction"] < .01001, audit
            assert audit["mapping_depth_relative_max_error"] < 1e-5, audit
            assert audit["mapping_colour_max_error"] < 1e-5, audit
            image_path = output / f"{phase}_{sample}.png"
            baseline = float(statistics["baseline_data_units"][sample])
            title = f"Bilinear {data.domains[0]['name']} TRAIN {sample}; baseline {baseline:.4f} data units"
            figure(image_path, inputs, augmented, batch["targets"], supervision, sample, title)
            record = {
                "dataset": data.domains[0]["name"], "split": "train", "index": sample,
                "metadata": batch["metadata"][sample], "manifest_sha256": digest(data.domains[0]["manifests"]["train"]),
                "baseline_data_units": baseline,
                "source_coverage": float(statistics["source_coverage"][sample]),
                "occluded_contributions": int(statistics["occluded_contributions"][sample]),
                "retained_views": int(statistics["retained_views"][sample]),
                "geometry_audit": audit, "image": str(image_path),
            }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        wrapped, wrapped_targets, counts = augment_training_batch(
            inputs, batch["targets"], data.domains[0], generator=torch.Generator().manual_seed(22), probability=1.)
        assert counts["synthetic_count"] == 2 and counts["original_count"] == 0
        assert torch.equal(wrapped_targets["depth"][:, 0], batch["targets"]["depth"][:, 0])
    torch.cuda.synchronize(device)
    sources = [Path(__file__), Path(__file__).with_name("synthetic_pair_bilinear.py"),
               Path(__file__).with_name("test_synthetic_pair_bilinear.py")]
    write_json(output / "completed.json", {
        "completed": True, "device": str(device), "GPU": torch.cuda.get_device_name(device),
        "samples": records, "analytic_subpixel_plane": analytic,
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "test_accessed": False, "model_inference_run": False,
        "limits": ["仅TRAIN再生成和几何/光度测试，不是网络精度结论。",
                   "近表面1%相容和0.25像素质心筛选是保守启发式，非真实连续曲面认证。",
                   "不会补洞或恢复从未观察到的纹理；监督按mapping生成。"],
    })


if __name__ == "__main__":
    main()
