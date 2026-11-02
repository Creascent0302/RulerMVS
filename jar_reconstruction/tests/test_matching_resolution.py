"""独立高分辨率匹配分支：原图来源、像素格点、掩码和实际模型路径。"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from jar.data import add_matching_images, collate_frames
from jar.data_transforms import resize_frame
from jar.geometry import resize_crop_intrinsics
from jar.multifocal.data import MultifocalFrameDataset
from jar.multifocal.synthetic import camera_calibration
from jar.pretrained_matching import FrozenMVSMatching, sample_pretrained_matching


WEIGHTS = Path("runs/carla_research_20261011/cache/mvsanywhere_hero.ckpt")
CAMERA_IDS = ["wide_left", "wide_right", "wide_up", "wide_down"]


@pytest.mark.parametrize("scale", [(1, 1), (2, 2), (2, 1.5)])
def test_scaled_matching_lattice_values_and_scene_uv_gradients(scale):
    y, x = torch.meshgrid(torch.arange(12), torch.arange(16), indexing="ij")
    field = (2 * (4 * x + 1.5) + 3 * (4 * y + 1.5)).float()[None, None]
    uv = torch.tensor([[[4.0, 5.0], [8.0, 7.0]]], requires_grad=True)
    sampled = sample_pretrained_matching(field, uv, image_scale=scale)
    mapped = (uv + .5) * uv.new_tensor(scale) - .5
    expected = (2 * mapped[..., :1] + 3 * mapped[..., 1:])
    torch.testing.assert_close(sampled, expected, atol=2e-5, rtol=0)
    sampled.sum().backward()
    expected_gradient = uv.new_tensor([2 * scale[0], 3 * scale[1]]).expand_as(uv)
    torch.testing.assert_close(uv.grad, expected_gradient, atol=2e-5, rtol=0)
    if scale == (1, 1):
        torch.testing.assert_close(sampled, sample_pretrained_matching(field, uv), atol=0, rtol=0)


def write_fixture(root):
    height, width = 64, 96
    y, x = np.mgrid[:height, :width]
    rgb = np.stack(((x % 3 == 0) * 255, ((x + y) % 5) * 60, (y % 4) * 80), axis=-1).astype(np.uint8)
    valid = np.ones((height, width), dtype=bool)
    valid[8:24, 16:64] = False
    cameras = []
    for index in (0, 2):
        cid = CAMERA_IDS[index]
        Image.fromarray(rgb).save(root / f"{cid}.png")
        np.save(root / f"{cid}_mask.npy", valid)
        k, transform = camera_calibration((height, width), [0, index * .2, 0])
        cameras.append({
            "camera_id": cid, "role": "wide", "frame_id": 3, "timestamp": 1.0,
            "rgb": f"{cid}.png", "K": k.tolist(), "T_cam_from_rig": transform.tolist(),
            "image_valid_mask": f"{cid}_mask.npy",
            "targets": {"frame_id": 3, "depth": "must_not_be_opened.npy", "depth_type": "optical_z"},
        })
    frame = {"sequence": "matching_resolution", "frame_id": 3, "timestamp": 1.0, "cameras": cameras[::-1]}
    (root / "frames.jsonl").write_text(json.dumps(frame) + "\n")
    cfg = {
        "data": {"root": str(root), "test_manifest": "frames.jsonl", "image_size": [16, 24], "matching_image_size": [32, 48]},
        "camera": {"mode": "wide4", "wide_ids": CAMERA_IDS, "tele_ids": ["tele_left", "tele_right"], "allow_missing": True, "disabled_ids": []},
    }
    return cfg, rgb, valid, cameras


def test_disk_reader_uses_original_twice_keeps_calibration_and_missing_masks(tmp_path):
    cfg, raw, mask, cameras = write_fixture(tmp_path)
    dataset = MultifocalFrameDataset(cfg, "test", load_targets=False)
    sample = dataset[0]
    inputs = sample["inputs"]
    original = torch.from_numpy(raw).permute(2, 0, 1).float()[None] / 255
    expected_low = F.interpolate(original, (16, 24), mode="bilinear", align_corners=False, antialias=True)
    expected_high = F.interpolate(original, (32, 48), mode="bilinear", align_corners=False, antialias=True)
    torch.testing.assert_close(inputs["rgb"][0], expected_low[0], atol=0, rtol=0)
    torch.testing.assert_close(inputs["matching_rgb"][0], expected_high[0], atol=0, rtol=0)
    upsampled_low = F.interpolate(expected_low, (32, 48), mode="bilinear", align_corners=False, antialias=True)
    assert (expected_high - upsampled_low).abs().mean() > .01
    expected_mask = F.interpolate(torch.from_numpy(mask)[None, None].float(), (32, 48), mode="area")[0, 0] > .999
    assert torch.equal(inputs["matching_image_mask"][0], expected_mask)
    for index in (1, 3):
        assert not inputs["view_mask"][index]
        assert not inputs["rgb"][index].any()
        assert not inputs["matching_rgb"][index].any()
        assert not inputs["matching_image_mask"][index].any()
    for camera in cameras:
        index = CAMERA_IDS.index(camera["camera_id"])
        expected_k = resize_crop_intrinsics(torch.tensor(camera["K"]), (.25, .25))
        torch.testing.assert_close(inputs["K"][index], expected_k, atol=0, rtol=0)
        torch.testing.assert_close(inputs["T_cam_from_rig"][index], torch.tensor(camera["T_cam_from_rig"]), atol=0, rtol=0)
    assert sample["targets"] == {}
    assert "depth" not in inputs
    assert all(item["from_hw"] == [64, 96] for item in sample["metadata"]["resize_per_present_camera"])
    batch = collate_frames([sample, sample])
    assert batch["inputs"]["rgb"].shape == (2, 4, 3, 16, 24)
    assert batch["inputs"]["matching_rgb"].shape == (2, 4, 3, 32, 48)
    assert not batch["inputs"]["matching_image_mask"][:, [1, 3]].any()
    # 若真的读取标签，磁盘中不存在的深度文件必须使监督读取失败。
    with pytest.raises(FileNotFoundError):
        MultifocalFrameDataset(cfg, "test", load_targets=True)[0]


def test_reader_without_new_attribute_keeps_existing_worker_compatibility(tmp_path):
    cfg, _, _, _ = write_fixture(tmp_path)
    dataset = MultifocalFrameDataset(cfg, "test", load_targets=False)
    expected = dataset[0]
    del dataset.matching_image_size
    legacy = dataset[0]
    assert "matching_rgb" not in legacy["inputs"]
    assert "matching_image_mask" not in legacy["inputs"]
    for field in ("rgb", "K", "T_cam_from_rig", "image_mask", "view_mask"):
        torch.testing.assert_close(legacy["inputs"][field], expected["inputs"][field], atol=0, rtol=0)


@pytest.mark.skipif(not WEIGHTS.exists(), reason="没有本地Hero缓存；测试不下载权重")
def test_frozen_encoder_reads_matching_rgb_and_masks_missing_views():
    torch.set_num_threads(2)
    model = FrozenMVSMatching(str(WEIGHTS)).eval()
    high = torch.rand(1, 3, 3, 32, 48)
    high[:, 1] = torch.nan
    high[:, 0, :, :4, :4] = torch.nan
    high_mask = torch.ones(1, 3, 32, 48, dtype=torch.bool)
    high_mask[:, 0, :4, :4] = False
    inputs = {
        "rgb": torch.full((1, 3, 3, 16, 24), torch.nan),
        "view_mask": torch.tensor([[True, False, True]]),
        "matching_rgb": high, "matching_image_mask": high_mask,
    }
    features = model(inputs)
    assert features.shape == (1, 3, 16, 8, 12)
    assert torch.isfinite(features).all()
    assert not features[:, 1].any()
    uv = torch.tensor([[[5.2, 4.3], [8.1, 7.2]]]).expand(3, -1, -1).clone().requires_grad_(True)
    sample_pretrained_matching(features[0], uv, [2, 2]).square().sum().backward()
    assert torch.isfinite(uv.grad).all() and uv.grad[[0, 2]].abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    del inputs["matching_image_mask"]
    with pytest.raises(ValueError, match="matching_image_mask"):
        model(inputs)


@pytest.mark.skipif(not WEIGHTS.exists(), reason="没有本地Hero缓存；测试不下载权重")
def test_full_dense_model_passes_matching_scale_to_coarse_and_refiner(monkeypatch):
    from jar.config import load_config
    from jar.model import ReconstructionModel
    from jar.models import dense_decoder, matching_evidence
    from jar.multifocal.synthetic import MultifocalSyntheticDataset

    torch.set_num_threads(2)
    cfg = load_config("configs/mf_et_jar_tiny.json")
    cfg["camera"].update(mode="wide4", allow_missing=True)
    cfg["data"]["matching_image_size"] = [64, 96]
    cfg["model"]["sparse_decoder"] = {"enabled": False}
    cfg["model"]["feature_exchange"] = {"enabled": False}
    cfg["model"]["matching_evidence"] = {"enabled": True, "hidden_dim": 8}
    cfg["model"]["pretrained_matching"] = {"weight": 0, "weights": str(WEIGHTS)}
    cfg["model"]["dense_evidence"] = {"enabled": False}
    cfg["model"]["dense_decoder"] = {
        "enabled": True, "hidden_dim": 8, "blocks_per_stage": 1,
        "context_layers": 0, "full_resolution_dim": 8,
        "calibrated_refinement": {"enabled": True, "feature_source": "pretrained", "hidden_dim": 8, "chunk_size": 128},
    }
    cfg["synthetic"].update(height=64, width=96, scene={"wall_depth": 8.0, "ground_height": -2.0, "max_depth": 24.0})
    original = MultifocalSyntheticDataset(cfg, load_targets=True)[0]
    sample = resize_frame(add_matching_images(original, [64, 96]), [32, 48])
    batch = collate_frames([sample])
    model = ReconstructionModel(cfg).train()
    calls = {"coarse": [], "refiner": []}

    def trace(name):
        def wrapped(features, uv, image_scale=None):
            calls[name].append(torch.as_tensor(image_scale).clone() if image_scale is not None else None)
            return sample_pretrained_matching(features, uv, image_scale)
        return wrapped

    monkeypatch.setattr(matching_evidence, "sample_pretrained_matching", trace("coarse"))
    monkeypatch.setattr(dense_decoder, "sample_pretrained_matching", trace("refiner"))
    output = model(batch["inputs"])
    assert output["dense_depth"].shape == (1, 4, 32, 48)
    assert torch.isfinite(output["dense_depth"]).all()
    for name in calls:
        assert calls[name], name
        for scale in calls[name]:
            torch.testing.assert_close(scale, torch.tensor([2., 2.]), atol=0, rtol=0)
    loss = output["dense_depth"].mean() + output["coarse"]["logits"].square().mean() * 1e-5
    loss.backward()
    gradient = model.dense_decoder.heads["1"].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.pretrained_matching.parameters())
