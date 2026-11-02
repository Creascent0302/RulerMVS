"""公共协议与指标测试不依赖官方模型的重量级库。"""

import copy
import json
import tempfile
from pathlib import Path
import pytest
import torch
from jar.synthetic import make_synthetic_batch, SyntheticRigDataset
from jar.geometry import project, unproject, pixel_grid
from official_methods.data import (
    canonical_inputs,
    resize_sample,
    crop_sample,
    horizontal_flip_sample,
    FourCameraDataset,
    validate_occupancy,
)
from official_methods.postprocess import depth_on_image_grid, depth_points
from official_methods.metrics import (
    depth_statistics,
    finish_depth,
    surface_metrics,
    occupancy_statistics,
)
from official_methods.occupancy_labels import observed_occupancy
from official_methods.prediction_io import native_to_cpu, export_prediction


def test_external_interface_and_inverse():
    batch = make_synthetic_batch(N=4, H=32, W=48)
    inputs = batch["inputs"]
    external = dict(
        images=inputs["rgb"],
        intrinsics=inputs["K"],
        T_cam_to_rig=torch.linalg.inv(inputs["T_cam_from_rig"]),
    )
    fixed = canonical_inputs(external)
    torch.testing.assert_close(fixed["T_cam_from_rig"], inputs["T_cam_from_rig"])
    with pytest.raises(ValueError, match="四个真实"):
        canonical_inputs({k: v[:, :3] for k, v in fixed.items()})


def test_geometry_metric_depth_and_resize_centers():
    sample = SyntheticRigDataset(1, (32, 48), 4)[0]
    resized = resize_sample(sample, (64, 96))
    k = sample["inputs"]["K"][0]
    t = sample["inputs"]["T_cam_from_rig"][0]
    uv = torch.tensor([[12.0, 16.0], [24.0, 20.0]])
    xyz, _ = unproject(uv, torch.tensor([3.0, 7.0]), k, t)
    projected, z, valid, _ = project(xyz, resized["inputs"]["K"][0:1], t[None], (64, 96))
    torch.testing.assert_close(projected[0], (uv + 0.5) * 2 - 0.5)
    torch.testing.assert_close(z[0], torch.tensor([3.0, 7.0]))
    assert valid.all()


def test_no_automatic_synthetic():
    config = {"image_size": [32, 48], "data": {"root": "", "train_manifest": ""}}
    with pytest.raises(ValueError, match="nonempty"):
        FourCameraDataset(config, "train")
    assert FourCameraDataset(config, "train", synthetic=True)[0]["inputs"]["rgb"].shape == (
        4,
        3,
        32,
        48,
    )
    assert (
        FourCameraDataset(config, "train", synthetic=True, load_targets=False)[0]["targets"] == {}
    )


def test_crop_flip_geometry():
    sample = SyntheticRigDataset(1, (32, 48), 4)[0]
    changed = horizontal_flip_sample(crop_sample(sample, 2, 4, 28, 40))
    uv = torch.tensor([[12.0, 16.0]])
    xyz, _ = unproject(
        uv, torch.tensor([4.0]), sample["inputs"]["K"][0], sample["inputs"]["T_cam_from_rig"][0]
    )
    new_uv, z, valid, _ = project(
        xyz, changed["inputs"]["K"][0:1], changed["inputs"]["T_cam_from_rig"][0:1], (28, 40)
    )
    torch.testing.assert_close(new_uv[0], torch.tensor([[31.0, 14.0]]))
    assert valid.all()


def test_postprocess_identity_and_missing_intrinsics():
    batch = make_synthetic_batch(N=4, H=32, W=48)
    depth = batch["targets"]["depth"]
    pred = dict(depth=depth, reference_indices=[0, 1, 2, 3], depth_K=batch["inputs"]["K"])
    mapped = depth_on_image_grid(pred, batch["inputs"])
    torch.testing.assert_close(mapped["depth"], depth, atol=2e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="depth_K"):
        depth_on_image_grid(
            dict(depth=depth[..., ::2, ::2], reference_indices=[0, 1, 2, 3]), batch["inputs"]
        )


def test_depth_missing_and_outlier_denominators():
    gt = torch.tensor([2.0, 4.0, 8.0])
    pred = torch.tensor([2.0, float("nan"), 160.0])
    stats = depth_statistics(
        pred, gt, torch.ones(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool), 1, 80
    )
    metrics = finish_depth(stats)
    assert metrics["prediction_coverage"] == 2 / 3
    assert metrics["delta_1"] == 1 / 3
    assert metrics["RMSE_m"] > 100  # 大误差不能以越界为借口丢掉。


def test_surface_definition():
    points = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]])
    out = surface_metrics(points, points + torch.tensor([0, 0, 0.1]), [0.2])
    assert abs(out["Chamfer_L1_m"] - 0.1) < 1e-6
    assert out["Fscore_0.2m"] == 1
    empty = torch.empty(0, 3)
    assert surface_metrics(empty, points, [0.2])["Fscore_0.2m"] == 0
    assert surface_metrics(points, empty, [0.2])["Fscore_0.2m"] is None


def test_occupancy_unknown_not_free():
    target = dict(
        state=torch.tensor([[[0, 1, 2]]]), grid_origin=torch.zeros(3), voxel_size=torch.ones(3)
    )
    pred = dict(target, state=torch.tensor([[[0, 2, 0]]]))
    stats = occupancy_statistics(pred, target)
    assert stats["union"] == 1 and stats["intersection"] == 0
    assert stats["free_on_unknown"] == 1
    assert stats["known_voxels"] == 2


def test_synthetic_first_hit_occupancy():
    batch = make_synthetic_batch(N=4, H=32, W=48)
    occ = observed_occupancy(batch["inputs"], batch["targets"], [[0, 20], [-4, 4], [-2, 4]], 1.0)
    validate_occupancy(occ)
    assert occ["voxel_size"].stride() == (1,)  # DataLoader pin_memory 不接受 expand 的重叠存储。
    assert (occ["state"] == 0).any() and (occ["state"] == 1).any() and (occ["state"] == 2).any()
    # 背景墙之后的体素不能被射线穿透标为 free。
    assert not (occ["state"][18:] == 0).any()


def test_native_namedtuple_and_portable_export(tmp_path):
    from collections import namedtuple

    Gaussian = namedtuple("Gaussian", ["means", "scales"])
    native = Gaussian(torch.ones(2, 3, requires_grad=True), torch.zeros(2, 3))
    copied = native_to_cpu({"gaussians": [native]})["gaussians"][0]
    assert type(copied) is Gaussian and not copied.means.requires_grad
    batch = make_synthetic_batch(N=4, H=16, W=32)
    pred = dict(
        depth=batch["targets"]["depth"],
        depth_K=batch["inputs"]["K"],
        reference_indices=[0, 1, 2, 3],
        native={"known": torch.tensor(1.0)},
    )
    export_prediction(tmp_path / "sample", pred, {}, batch["inputs"], [], {"method": "test"})
    import numpy as np

    with np.load(tmp_path / "sample.npz", allow_pickle=False) as values:
        assert values["depth"].shape == (1, 4, 16, 32)
        assert "occupancy" not in values and "confidence" not in values
    assert torch.load(tmp_path / "sample.native.pt", weights_only=True)["known"] == 1
