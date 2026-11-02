"""公共数据不能隐式变成米制，也不能复制双目冒充四目。"""
import json

import numpy as np
from PIL import Image
import pytest
import torch

from jar.data import CarlaFrameDataset, build_dataset, collate_frames
from official_methods.data import FourCameraDataset, collate


def write_sample(root, units):
    Image.new("RGB", (12, 8), color=(40, 80, 120)).save(root / "rgb.png")
    np.save(root / "depth.npy", np.full((8, 12), 7.25, np.float32))
    cameras = []
    for index in range(2):
        transform = np.eye(4)
        transform[0, 3] = index * 0.5
        cameras.append({
            "camera_id": f"view_{index}", "frame_id": 1, "timestamp": 0.1,
            "rgb": "rgb.png", "K": [[10, 0, 5.5], [0, 10, 3.5], [0, 0, 1]],
            "T_cam_from_rig": transform.tolist(),
            "targets": {"frame_id": 1, "depth": "depth.npy", "depth_type": "optical_z",
                        "depth_units": units,
                        "depth_encoding": "metric_npy" if units == "meters" else "scene_unit_npy"},
        })
    frame = {"sequence": "unit", "frame_id": 1, "timestamp": 0.1,
             "geometry_units": units, "sample_type": "synchronized_stereo",
             "synchronized_capture": True, "cameras": cameras}
    (root / "frames.jsonl").write_text(json.dumps(frame) + "\n")


@pytest.mark.parametrize("load_targets", [True, False])
def test_scene_units_require_explicit_configuration(tmp_path, load_targets):
    write_sample(tmp_path, "scene_unit")
    with pytest.raises(ValueError, match="几何单位"):
        CarlaFrameDataset(tmp_path, "frames.jsonl", load_targets=load_targets)
    sample = CarlaFrameDataset(tmp_path, "frames.jsonl", load_targets=load_targets,
                               geometry_units="scene_unit")[0]
    assert sample["metadata"]["geometry_units"] == "scene_unit"
    assert sample["metadata"]["synchronized_capture"] is True
    assert "depth" not in sample["inputs"]
    if load_targets:
        torch.testing.assert_close(sample["targets"]["depth"], torch.full((2, 8, 12), 7.25))
    else:
        assert not sample["targets"]


def test_stereo_is_explicit_and_four_camera_default_stays_strict(tmp_path):
    write_sample(tmp_path, "meters")
    config = {"data": {"root": str(tmp_path), "test_manifest": "frames.jsonl"},
              "image_size": [8, 12]}
    with pytest.raises(ValueError, match="4个真实视角"):
        FourCameraDataset(config, "test")[0]
    config["data"]["num_views"] = 2
    sample = FourCameraDataset(config, "test")[0]
    batch = collate([sample], expected_views=2)
    assert batch["inputs"]["rgb"].shape == (1, 2, 3, 8, 12)
    with pytest.raises(ValueError, match="四个真实视角"):
        collate([sample])


def test_geometry_and_target_units_cannot_disagree(tmp_path):
    write_sample(tmp_path, "scene_unit")
    path = tmp_path / "frames.jsonl"
    frame = json.loads(path.read_text())
    frame["cameras"][0]["targets"]["depth_units"] = "meters"
    path.write_text(json.dumps(frame))
    with pytest.raises(ValueError, match="深度与相机平移"):
        CarlaFrameDataset(tmp_path, path, geometry_units="scene_unit")


def test_public_training_loader_preserves_stereo_and_matching_resolution(tmp_path):
    write_sample(tmp_path, "meters")
    config = {"data": {"protocol": "calibrated_multiview", "num_views": 2,
                        "root": str(tmp_path), "train_manifest": "frames.jsonl",
                        "image_size": [4, 6], "matching_image_size": [8, 12]}}
    sample = build_dataset(config)[0]
    batch = collate_frames([sample])
    assert batch["inputs"]["rgb"].shape == (1, 2, 3, 4, 6)
    assert batch["inputs"]["matching_rgb"].shape == (1, 2, 3, 8, 12)
    assert batch["targets"]["depth"].shape == (1, 2, 4, 6)
    image_only = build_dataset(config, load_targets=False)[0]
    assert not image_only["targets"]
    for key, value in sample["inputs"].items():
        torch.testing.assert_close(value, image_only["inputs"][key], rtol=0, atol=0)


def test_mixed_training_carla_missing_camera_keeps_ids_and_masks(tmp_path):
    from jar.pretraining.data import MixedDataset, collate_mixed, dataset_config

    write_sample(tmp_path, "meters")
    path = tmp_path / "frames.jsonl"
    frame = json.loads(path.read_text())
    frame["cameras"][0]["camera_id"] = "stereo_left"
    frame["cameras"][1]["camera_id"] = "stereo_up"
    path.write_text(json.dumps(frame) + "\n")
    camera = {"mode": "wide4", "allow_missing": True, "disabled_ids": [],
              "wide_ids": ["stereo_left", "stereo_right", "stereo_up", "stereo_down"],
              "tele_ids": ["tele_left", "tele_right"]}
    domain = {"root": str(tmp_path), "num_views": 4, "geometry_units": "meters",
              "reference_index": 2, "camera": camera,
              "manifests": {"train": str(path), "val": str(path), "test": str(path)}}
    config = {"datasets": [domain], "image_size": [4, 6], "matching_image_size": [8, 12],
              "augmentation": False}
    dataset = MixedDataset(config, "val")
    item = dataset[(0, 0, -1)]
    assert item["inputs"]["camera_ids"] == ["stereo_up", "stereo_left", "stereo_right", "stereo_down"]
    assert item["inputs"]["view_mask"].tolist() == [True, True, False, False]
    assert torch.equal(item["inputs"]["view_mask"], item["inputs"]["camera_valid"])
    assert not item["targets"]["depth_valid"][2:].any()
    assert not item["inputs"]["matching_image_mask"][2:].any()
    batch = collate_mixed([item])
    assert batch["inputs"]["rgb"].shape == (1, 4, 3, 4, 6)
    image_only = build_dataset(dataset_config(domain, config), "val", load_targets=False)[0]
    assert not image_only["targets"]
    assert image_only["inputs"]["view_mask"].tolist() == [True, False, True, False]


def test_input_valid_mask_is_shared_without_loading_targets(tmp_path):
    write_sample(tmp_path, "meters")
    mask = np.ones((8, 12), dtype=bool)
    mask[:, :2] = False
    np.save(tmp_path / "mask.npy", mask)
    path = tmp_path / "frames.jsonl"
    frame = json.loads(path.read_text())
    frame["cameras"][0]["valid_mask"] = "mask.npy"
    frame["cameras"][0]["targets"]["depth"] = "absent_label.npy"
    path.write_text(json.dumps(frame))
    native = CarlaFrameDataset(tmp_path, path, load_targets=False)[0]
    cfg = {"data": {"root": str(tmp_path), "test_manifest": str(path), "num_views": 2},
           "image_size": [8, 12]}
    baseline = FourCameraDataset(cfg, "test", load_targets=False)[0]
    torch.testing.assert_close(native["inputs"]["image_mask"], baseline["inputs"]["image_mask"])
    assert not native["inputs"]["image_mask"][0, :, :2].any()
    cfg["image_size"] = [4, 6]
    official = FourCameraDataset(cfg, "test", load_targets=False)[0]
    public = {"data": {**cfg["data"], "protocol": "calibrated_multiview", "image_size": [4, 6]}}
    own = build_dataset(public, "test", load_targets=False)[0]
    torch.testing.assert_close(own["inputs"]["image_mask"], official["inputs"]["image_mask"])
