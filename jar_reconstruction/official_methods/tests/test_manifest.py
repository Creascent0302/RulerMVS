"""实际临时 PNG/NPY/NPZ/JSONL 文件上的 CARLA 四目契约测试。"""

from __future__ import annotations

import copy
import json

import numpy as np
from PIL import Image
import pytest
import torch

from jar.synthetic import SyntheticRigDataset
from official_methods.data import FourCameraDataset, audit_split_sequences, collate


@pytest.fixture
def manifest_scene(tmp_path):
    sample = SyntheticRigDataset(1, (16, 24), 4, seed=1701)[0]
    frame = {
        "sequence": "TownTest/route_01/sequence_01",
        "frame_id": "42",
        "timestamp": 2.1,
        "cameras": [],
    }
    for camera in range(4):
        rgb = (
            (sample["inputs"]["rgb"][camera].permute(1, 2, 0).numpy() * 255)
            .round()
            .astype(np.uint8)
        )
        Image.fromarray(rgb).save(tmp_path / f"rgb_{camera}.png")
        np.save(tmp_path / f"label_depth_{camera}.npy", sample["targets"]["depth"][camera].numpy())
        np.save(
            tmp_path / f"label_valid_{camera}.npy", sample["targets"]["depth_valid"][camera].numpy()
        )
        frame["cameras"].append(
            {
                "camera_id": f"cam_{camera}",
                "frame_id": "42",
                "timestamp": 2.1,
                "rgb": f"rgb_{camera}.png",
                "K": sample["inputs"]["K"][camera].tolist(),
                "T_cam_from_rig": sample["inputs"]["T_cam_from_rig"][camera].tolist(),
                "targets": {
                    "frame_id": "42",
                    "depth": f"label_depth_{camera}.npy",
                    "depth_valid": f"label_valid_{camera}.npy",
                    "depth_type": "optical_z",
                },
            }
        )
    state = np.array([[[0, 1, 2], [2, 1, 0]]], dtype=np.int64)
    semantics = np.array([[[-1, 7, -1], [-1, 8, -1]]], dtype=np.int64)
    occupancy = {
        "state": state,
        "semantics": semantics,
        "grid_origin": np.array([1.0, -2.0, -1.0]),
        "voxel_size": np.array([0.25, 0.25, 0.5]),
    }
    np.savez(tmp_path / "label_occupancy.npz", **occupancy)
    frame["occupancy_targets"] = {"frame_id": "42", "path": "label_occupancy.npz"}
    config = {
        "image_size": [16, 24],
        "data": {
            "root": str(tmp_path),
            "train_manifest": "train.jsonl",
            "val_manifest": "",
            "test_manifest": "train.jsonl",
        },
    }

    def save(updated=frame, filename="train.jsonl"):
        (tmp_path / filename).write_text(json.dumps(updated) + "\n")

    save()
    return tmp_path, config, frame, sample, occupancy, save


def test_manifest_files_four_camera_batch_and_targets(manifest_scene):
    _, config, _, original, _, _ = manifest_scene
    sample = FourCameraDataset(config, "train")[0]
    batch = collate([sample])
    assert batch["inputs"]["rgb"].shape == (1, 4, 3, 16, 24)
    assert batch["inputs"]["K"].shape == (1, 4, 3, 3)
    assert batch["inputs"]["T_cam_from_rig"].shape == (1, 4, 4, 4)
    torch.testing.assert_close(sample["targets"]["depth"], original["targets"]["depth"])
    assert sample["metadata"]["sample_id"] == "TownTest/route_01/sequence_01/42"
    assert "occupancy" not in batch["inputs"] and "depth" not in batch["inputs"]
    assert isinstance(batch["targets"]["occupancy"], list)


@pytest.mark.parametrize("damaged", ["missing", "corrupt"])
def test_inputs_only_never_opens_label_files(manifest_scene, monkeypatch, damaged):
    root, config, frame, _, _, save = manifest_scene
    for camera in frame["cameras"]:
        camera["targets"] = {
            "frame_id": "wrong-label-frame",
            "depth": "label_bad.npy",
            "semantic": "label_bad_semantics.npy",
            "sky": "label_bad_sky.npy",
            "depth_type": "invalid-label-type",
        }
    frame["occupancy_targets"] = {"frame_id": "wrong-label-frame", "path": "label_bad.npz"}
    if damaged == "corrupt":
        (root / "label_bad.npy").write_bytes(b"not a numpy file")
        (root / "label_bad.npz").write_bytes(b"not a numpy archive")
    save()

    def refuse_labels(*args, **kwargs):
        raise AssertionError(f"inputs-only unexpectedly opened numpy labels: {args[0]}")

    monkeypatch.setattr(np, "load", refuse_labels)
    sample = FourCameraDataset(config, "test", load_targets=False)[0]
    assert sample["targets"] == {}
    assert sample["inputs"]["rgb"].shape == (4, 3, 16, 24)


def test_occupancy_unknown_semantics_and_geometry_preserved(manifest_scene):
    _, config, _, _, original, _ = manifest_scene
    occ = FourCameraDataset(config, "train")[0]["targets"]["occupancy"]
    for key, value in original.items():
        torch.testing.assert_close(occ[key], torch.from_numpy(value))
    assert (occ["semantics"][occ["state"] == 2] == -1).all()
    assert int((occ["state"] == 0).sum()) == 2
    assert int((occ["state"] == 1).sum()) == 2
    assert int((occ["state"] == 2).sum()) == 2


@pytest.mark.parametrize(
    "fault", ["fractional_state", "bad_semantic_shape", "negative_voxel", "nan_origin", "inf_voxel"]
)
def test_invalid_occupancy_file_rejected(manifest_scene, fault):
    root, config, _, _, occ, _ = manifest_scene
    if fault == "fractional_state":
        occ["state"] = occ["state"].astype(np.float32)
        occ["state"][0, 0, 0] = 1.5
    elif fault == "bad_semantic_shape":
        occ["semantics"] = np.zeros((2, 2, 2), dtype=np.int64)
    elif fault == "negative_voxel":
        occ["voxel_size"][0] = -0.25
    elif fault == "nan_origin":
        occ["grid_origin"][0] = np.nan
    elif fault == "inf_voxel":
        occ["voxel_size"][0] = np.inf
    np.savez(root / "label_occupancy.npz", **occ)
    with pytest.raises(ValueError):
        FourCameraDataset(config, "train")[0]


@pytest.mark.parametrize(
    "fault", ["camera_timestamp", "camera_frame", "depth_frame", "occupancy_frame"]
)
def test_temporal_synchronization_rejected(manifest_scene, fault):
    _, config, frame, _, _, save = manifest_scene
    if fault == "camera_timestamp":
        frame["cameras"][2]["timestamp"] += 0.05
    elif fault == "camera_frame":
        frame["cameras"][2]["frame_id"] = "41"
    elif fault == "depth_frame":
        frame["cameras"][2]["targets"]["frame_id"] = "41"
    else:
        frame["occupancy_targets"]["frame_id"] = "41"
    save()
    with pytest.raises(ValueError):
        FourCameraDataset(config, "train")[0]
    if fault.startswith("camera_"):
        with pytest.raises(ValueError):
            FourCameraDataset(config, "test", load_targets=False)[0]


def test_sequence_split_overlap_rejected(manifest_scene):
    _, config, frame, _, _, save = manifest_scene
    config["data"]["test_manifest"] = ""
    config["data"]["val_manifest"] = "val.jsonl"
    different_frame = copy.deepcopy(frame)
    different_frame["frame_id"] = "999"
    save(different_frame, "val.jsonl")
    with pytest.raises(ValueError, match="同时出现在"):
        audit_split_sequences(config)
    different_frame["sequence"] = "TownOther/route_02/sequence_02"
    save(different_frame, "val.jsonl")
    assert len(audit_split_sequences(config)) == 2


def test_explicit_temporal_gap_preserves_sequence_identity(manifest_scene):
    _, config, frame, _, _, save = manifest_scene
    config["data"].update(
        test_manifest="", val_manifest="val.jsonl", split_protocol="temporal_gap", min_frame_gap=40
    )
    later = dict(frame, frame_id="83")
    save(later, "val.jsonl")
    assert len(audit_split_sequences(config)) == 1
    save(dict(later, frame_id="82"), "val.jsonl")
    with pytest.raises(ValueError, match="间隔小于"):
        audit_split_sequences(config)
    save(dict(later, frame_id="42"), "val.jsonl")
    with pytest.raises(ValueError, match="时间区间重叠"):
        audit_split_sequences(config)
    save(later, "val.jsonl")
    config["data"]["test_manifest"] = "val.jsonl"
    with pytest.raises(ValueError, match="测试序列"):
        audit_split_sequences(config)


def test_explicit_supervision_range_only_masks_labels(manifest_scene):
    _, config, _, _, _, _ = manifest_scene
    original = FourCameraDataset(config, "train")[0]
    config["data"]["supervision_depth_range"] = [4.0, 8.0]
    filtered = FourCameraDataset(config, "train")[0]
    for key, value in original["inputs"].items():
        torch.testing.assert_close(value, filtered["inputs"][key])
    depth = original["targets"]["depth"]
    torch.testing.assert_close(depth, filtered["targets"]["depth"])
    expected = original["targets"]["depth_valid"] & (depth >= 4) & (depth <= 8)
    torch.testing.assert_close(filtered["targets"]["depth_valid"], expected)
    assert expected.any() and (original["targets"]["depth_valid"] & ~expected).any()
    assert FourCameraDataset(config, "train", load_targets=False)[0]["targets"] == {}
    config["data"]["supervision_depth_range"] = [8.0, 4.0]
    with pytest.raises(ValueError, match="supervision_depth_range"):
        FourCameraDataset(config, "train")


def test_valid_input_mask_crop_and_resize_geometry(manifest_scene):
    root, config, frame, original, _, save = manifest_scene
    mask = np.ones((16, 24), dtype=bool)
    mask[:, :4] = False
    np.save(root / "input_valid.npy", mask)
    for camera in frame["cameras"]:
        camera["valid_mask"] = "input_valid.npy"
    save()
    config["image_crop"] = [2, 2, 12, 20]
    config["image_size"] = [24, 40]
    sample = FourCameraDataset(config, "train")[0]
    assert not sample["inputs"]["image_mask"][..., :4].any()
    assert sample["inputs"]["image_mask"][..., 4:].all()
    crop = torch.tensor([[1.0, 0.0, -2.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]])
    resize = torch.tensor([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])
    torch.testing.assert_close(sample["inputs"]["K"], resize @ crop @ original["inputs"]["K"])
    assert sample["targets"]["depth"].shape == (4, 24, 40)


def test_ray_distance_converted_to_optical_meters(manifest_scene):
    root, config, frame, original, _, save = manifest_scene
    y, x = torch.meshgrid(torch.arange(16), torch.arange(24), indexing="ij")
    pixels = torch.stack([x, y, torch.ones_like(x)], -1).float()
    for index, camera in enumerate(frame["cameras"]):
        k = original["inputs"]["K"][index]
        rays = pixels @ torch.linalg.inv(k).T
        distance = original["targets"]["depth"][index] * torch.linalg.vector_norm(rays, dim=-1)
        np.save(root / camera["targets"]["depth"], distance.numpy())
        camera["targets"]["depth_type"] = "ray_range"
    save()
    depth = FourCameraDataset(config, "train")[0]["targets"]["depth"]
    torch.testing.assert_close(depth, original["targets"]["depth"], atol=2e-6, rtol=1e-6)


def test_optional_sky_labels_resize_and_mixed_batch(manifest_scene):
    root, config, frame, _, _, save = manifest_scene
    sky = np.zeros((16, 24), dtype=np.float32)
    sky[:8] = 1
    sky[:, :2] = np.nan
    np.save(root / "sky.npy", sky)
    frame["cameras"][0]["targets"]["sky"] = "sky.npy"
    save()
    config["image_crop"] = [2, 2, 12, 20]
    config["image_size"] = [24, 40]
    sample = FourCameraDataset(config, "train")[0]
    assert sample["targets"]["sky"].shape == (4, 24, 40)
    assert sample["targets"]["sky"][0, :12].eq(1).all()
    assert sample["targets"]["sky"][0, 12:].eq(0).all()
    assert sample["targets"]["sky"][1:].isnan().all()
    unlabeled = copy.deepcopy(sample)
    del unlabeled["targets"]["sky"]
    batch = collate([unlabeled, sample])
    assert batch["targets"]["sky"].shape == (2, 4, 24, 40)
    assert batch["targets"]["sky"][0].isnan().all()
    assert "sky" not in batch["inputs"]
    np.save(root / "sky.npy", np.full((16, 24), 2, dtype=np.float32))
    with pytest.raises(ValueError, match="sky 标签"):
        FourCameraDataset(config, "train")[0]
