"""真实渲染图像 + 轻量固定描述子的接口测试，不宣称预训练网络精度。"""

import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from jar.data import build_dataset, collate_frames
from jar.models.ruler_direct_model import RulerDirectModel
from jar.models.ruler_direct_training import select_reference_targets
from jar.multifocal.view_order import INPUT_CAMERA_FIELDS
from jar.synthetic import make_synthetic_batch
from tests.test_ruler_direct import tiny_settings
from tools.train_ruler_direct import direct_loss


class ImageFoundation(nn.Module):
    """只测试共享编码调度，固定从几何一致RGB产生四层图像特征。"""

    def __init__(self, settings, state=None):
        super().__init__()
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        self.references = inputs["_reference_indices"]
        rgb = inputs["rgb"]
        features = F.avg_pool2d(rgb.flatten(0, 1), 16).repeat(1, 256, 1, 1)
        return [features * (index + 1) / 4 for index in range(4)]


class ImageMatching(nn.Module):
    def __init__(self, state=None):
        super().__init__()
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        rgb = inputs["rgb"]
        batch, views = rgb.shape[:2]
        features = F.avg_pool2d(rgb.flatten(0, 1), 4).repeat(1, 6, 1, 1)[:, :16]
        return features.unflatten(0, (batch, views))


@pytest.fixture
def model_factory(monkeypatch):
    import jar.models.ruler_direct_model as module

    torch.set_num_threads(2)
    monkeypatch.setattr(module, "FrozenFoundation", ImageFoundation)
    monkeypatch.setattr(module, "FrozenMVSMatching", ImageMatching)

    def build(references=None):
        settings = tiny_settings(use_scene=False, use_geometry=False, local_kind="volume")
        settings["reference_index"] = 2
        if references is not None:
            settings["reference_indices"] = references
        config = {
            "direct": settings,
            "camera": {"mode": "wide4", "wide_ids": ["left", "right", "up", "down"]},
            "model": {"dense_decoder": {"enabled": True}, "sparse_decoder": {"enabled": False},
                      "execution": {"reference_indices": [2]}},
        }
        return RulerDirectModel(config, original_state={}).eval()

    return build


def batch_inputs(batch=2):
    sample = make_synthetic_batch(B=batch, N=4, H=32, W=48)
    sample["inputs"]["camera_ids"] = [["left", "right", "up", "down"] for _ in range(batch)]
    return sample


def test_multiple_references_share_encoders_and_match_single(model_factory):
    model = model_factory([0, 1, 2, 3])
    batch = batch_inputs()
    multi = model(batch["inputs"])
    assert model.matching.calls == model.foundation.calls == 1
    assert model.foundation.references == (0, 1, 2, 3)
    assert multi["dense_depth"].shape == (2, 4, 32, 48)
    assert multi["direct"] is multi["direct_by_reference"][2]
    assert multi["dense_valid"].all()
    assert multi["dense_support"].max() <= 3
    assert multi["dense_support"].count_nonzero() > 0
    assert multi["dense_state"].eq(2).all()
    assert not multi["dense_observed"].any()
    for reference in range(4):
        torch.testing.assert_close(
            multi["dense_support"][:, reference],
            multi["direct_by_reference"][reference]["support"][:, 0].long(),
            atol=0, rtol=0,
        )
        model.settings["reference_indices"] = [reference]
        single = model(batch["inputs"])
        torch.testing.assert_close(single["dense_depth"][:, reference], multi["dense_depth"][:, reference], atol=0, rtol=0)
        assert not single["dense_valid"][:, [index for index in range(4) if index != reference]].any()


def test_default_single_reference_checkpoint_and_label_isolation(model_factory, tmp_path):
    model = model_factory()
    batch = batch_inputs(batch=1)
    before = model(batch["inputs"])
    assert before["reference_indices"] == (2,)
    assert model.foundation.references == (2,)
    checkpoint = tmp_path / "multi.pt"
    torch.save(model.checkpoint_record(), checkpoint)
    restored = RulerDirectModel.from_checkpoint(checkpoint)
    batch["targets"]["depth"].fill_(12345)
    after = restored(batch["inputs"])
    torch.testing.assert_close(after["dense_depth"], before["dense_depth"], atol=0, rtol=0)


def test_lightweight_encoder_cache_entry_keeps_single_reference_contract(model_factory):
    from types import SimpleNamespace

    model = model_factory()
    encoder_only = SimpleNamespace(
        foundation=model.foundation, matching=model.matching, cfg=model.cfg,
        reference_index=model.reference_index, settings=model.settings,
    )
    inputs = batch_inputs(batch=1)["inputs"]
    expected = model.encode(inputs)
    actual = RulerDirectModel.encode(encoder_only, inputs)
    assert actual["reference_index"] == expected["reference_index"] == 2
    for first, second in zip(actual["raw_maps"], expected["raw_maps"]):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    torch.testing.assert_close(actual["matching_features"], expected["matching_features"], atol=0, rtol=0)


def test_missing_rgb_and_partial_depth_are_independent(model_factory):
    model = model_factory([0, 1, 2, 3])
    batch = batch_inputs()
    inputs, targets = batch["inputs"], batch["targets"]
    inputs["view_mask"][0, 0] = False
    inputs["image_mask"][0, 0] = False
    inputs["rgb"][0, 0] = float("nan")
    targets["depth_valid"][1, 3] = False
    result = model(inputs)
    assert torch.isfinite(result["dense_depth"]).all()
    assert not result["dense_valid"][0, 0].any()
    assert result["dense_depth"][0, 0].count_nonzero() == 0
    assert result["dense_valid"][1, 3].all(), "缺少标签不能删除推理RGB"
    for reference in (0, 3):
        selected = select_reference_targets(inputs, targets, reference)
        invalid_batch = 0 if reference == 0 else 1
        assert not selected["valid"][invalid_batch].any()
        assert selected["depth"][invalid_batch].count_nonzero() == 0
    missing = select_reference_targets(inputs, {}, 2)
    assert not missing["valid"].any()


def test_per_batch_camera_permutation_restores_depth_and_supervision(model_factory):
    model = model_factory([0, 2, 3])
    batch = batch_inputs()
    inputs, targets = batch["inputs"], batch["targets"]
    expected = model(inputs)
    order = torch.tensor([[3, 2, 1, 0], [1, 0, 3, 2]])
    permuted = dict(inputs)
    rows = torch.arange(2)[:, None]
    for name in INPUT_CAMERA_FIELDS:
        if name in inputs:
            permuted[name] = inputs[name][rows, order]
    permuted["camera_ids"] = [[inputs["camera_ids"][b][index] for index in order[b]] for b in range(2)]
    permuted_targets = {key: value[rows, order] for key, value in targets.items()}
    actual = model(permuted)
    torch.testing.assert_close(actual["dense_depth"], expected["dense_depth"][rows, order], atol=0, rtol=0)
    torch.testing.assert_close(actual["dense_support"], expected["dense_support"][rows, order], atol=0, rtol=0)
    for position, reference in enumerate(actual["reference_indices"]):
        indices = actual["direct_camera_indices"][:, position]
        torch.testing.assert_close(indices, order.argsort(1)[:, reference])
        selected = select_reference_targets(permuted, permuted_targets, indices)
        original = select_reference_targets(inputs, targets, reference)
        torch.testing.assert_close(selected["depth"], original["depth"], atol=0, rtol=0)
        torch.testing.assert_close(selected["valid"], original["valid"], atol=0, rtol=0)


def test_multireference_supervision_backward_and_optimizer_step(model_factory):
    model = model_factory([0, 2, 3]).train()
    batch = batch_inputs()
    batch["targets"]["depth_valid"][:, 0] = False
    output = model(batch["inputs"])
    losses = []
    settings = {"metric_normalizer_m": 20., "log_weight": .2, "gradient_weight": .05}
    for position, reference in enumerate(output["reference_indices"]):
        prediction = output["direct_by_reference"][reference]
        targets = select_reference_targets(
            batch["inputs"], batch["targets"], output["direct_camera_indices"][:, position],
            depth_range=(2, 80), prediction_valid=prediction["valid"],
        )
        if targets["valid"].any():
            losses.append(direct_loss(prediction, targets, settings)["total"])
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=1e-4)
    loss = torch.stack(losses).mean()
    loss.backward()
    for name, parameter in model.decoder.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    parameter = next(model.decoder.score.parameters())
    before = parameter.detach().clone()
    optimizer.step()
    assert not torch.equal(before, parameter)


@pytest.mark.parametrize("references", [[], [1, 1], [-1], [4], [2.5]])
def test_invalid_reference_configuration_fails(model_factory, references):
    with pytest.raises(ValueError, match="reference_indices"):
        model_factory(references)(batch_inputs(batch=1)["inputs"])


def test_real_manifest_keeps_independent_camera_depth_and_missing_labels(tmp_path):
    cameras = []
    height, width = 16, 24
    for index, name in enumerate(("left", "right", "up", "down")):
        if name == "right":
            continue
        Image.fromarray(np.full((height, width, 3), 30 + index, np.uint8)).save(tmp_path / f"{name}.png")
        camera = {
            "camera_id": name, "frame_id": 1, "timestamp": 0.1, "rgb": f"{name}.png",
            "K": [[12., 0., 11.5], [0., 12., 7.5], [0., 0., 1.]],
            "T_cam_from_rig": np.eye(4).tolist(),
        }
        if name != "down":
            np.save(tmp_path / f"{name}.npy", np.full((height, width), 10 + index, np.float32))
            camera["targets"] = {"frame_id": 1, "depth": f"{name}.npy", "depth_type": "optical_z"}
        cameras.append(camera)
    frame = {"sequence": "new_data", "frame_id": 1, "timestamp": 0.1, "cameras": cameras}
    (tmp_path / "frames.jsonl").write_text(json.dumps(frame) + "\n")
    config = {
        "camera": {"mode": "wide4", "wide_ids": ["left", "right", "up", "down"],
                   "tele_ids": [], "allow_missing": True},
        "data": {"root": str(tmp_path), "train_manifest": "frames.jsonl"},
    }
    item = build_dataset(config)[0]
    batch = collate_frames([item])
    assert batch["targets"]["depth"].shape == (1, 4, height, width)
    assert batch["targets"]["depth"][0, 0].eq(10).all()
    assert batch["targets"]["depth"][0, 2].eq(12).all()
    assert not batch["targets"]["depth_valid"][0, [1, 3]].any()
    assert batch["inputs"]["view_mask"].tolist() == [[True, False, True, True]]
    assert "depth" not in batch["inputs"]
    for camera in cameras:
        if "targets" in camera:
            camera["targets"]["depth"] = "nonexistent.npy"
    (tmp_path / "frames.jsonl").write_text(json.dumps(frame) + "\n")
    assert build_dataset(config, load_targets=False)[0]["targets"] == {}


def test_depth_convention_and_intrinsics_are_per_camera(tmp_path):
    from jar.data import CarlaFrameDataset

    height, width = 4, 6
    yy, xx = np.mgrid[:height, :width]
    cameras = []
    for index, kind in enumerate(("optical_z", "ray_range")):
        focal = 4.0 + index * 3
        intrinsic = [[focal, 0., 2.5], [0., focal, 1.5], [0., 0., 1.]]
        distance = np.full((height, width), 5.0 + index * 7, dtype=np.float32)
        if kind == "ray_range":
            norm = np.sqrt(1 + ((xx - 2.5) / focal) ** 2 + ((yy - 1.5) / focal) ** 2)
            distance = (distance * norm).astype(np.float32)
        np.save(tmp_path / f"{index}.npy", distance)
        Image.fromarray(np.zeros((height, width, 3), np.uint8)).save(tmp_path / f"{index}.png")
        cameras.append({
            "frame_id": 1, "timestamp": 0.1, "rgb": f"{index}.png", "K": intrinsic,
            "T_cam_from_rig": np.eye(4).tolist(),
            "targets": {"frame_id": 1, "depth": f"{index}.npy", "depth_type": kind},
        })
    frame = {"sequence": "mixed_depth", "frame_id": 1, "timestamp": 0.1, "cameras": cameras}
    (tmp_path / "frames.jsonl").write_text(json.dumps(frame) + "\n")
    item = CarlaFrameDataset(tmp_path, "frames.jsonl")[0]
    expected = torch.tensor([5., 12.])[:, None, None].expand(2, height, width)
    torch.testing.assert_close(item["targets"]["depth"], expected, atol=2e-6, rtol=0)
    assert item["targets"]["depth_valid"].all()


def test_reference_targets_rejects_single_label_broadcast_and_masks_bad_values():
    batch = batch_inputs()
    with pytest.raises(ValueError, match="不可广播"):
        select_reference_targets(batch["inputs"], {"depth": batch["targets"]["depth"][:, :1]}, 2)
    batch["targets"]["depth"][0, 2, 0, 0] = float("nan")
    batch["targets"]["depth"][1, 2, 0, 0] = 1000
    selected = select_reference_targets(batch["inputs"], batch["targets"], 2, depth_range=(2, 80))
    assert selected["depth"][:, :, 0, 0].count_nonzero() == 0
    assert not selected["valid"][:, :, 0, 0].any()
    assert torch.isfinite(selected["depth"]).all()


def test_common_training_wrapper_counts_only_selected_supervision(model_factory):
    from jar.training import ModelAndLoss

    model = model_factory()
    batch = batch_inputs()
    batch["targets"]["depth_valid"][:, 2] = False
    wrapper = ModelAndLoss(model, None, model.cfg)
    loss = wrapper(batch["inputs"], batch["targets"])
    assert wrapper.last_supervised_samples == 0
    assert loss.requires_grad and float(loss.detach()) == 0
    loss.backward()
    model.settings["reference_indices"] = [0, 2]
    loss = wrapper(batch["inputs"], batch["targets"])
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    assert wrapper.last_supervised_samples == 2
    assert int(wrapper.last_metrics["supervised_views"]) == 2


def test_config_template_keeps_new_data_paths_empty():
    from pathlib import Path
    from jar.config import load_config

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/ruler_direct_multicamera.json")
    assert config["model"]["name"] == "ruler_direct"
    assert config["direct"]["reference_indices"] == [0, 1, 2, 3]
    for key in ("root", "train_manifest", "val_manifest", "test_manifest"):
        assert config["data"][key] == ""
    with pytest.raises(ValueError, match="requires nonempty"):
        build_dataset(config)
