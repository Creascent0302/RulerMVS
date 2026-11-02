"""固定权重的评测边界；使用微型CPU接口，不读取任何真实TEST帧。"""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from tools import confirm_ruler_direct, evaluate_ruler_direct
from tools.carla_compare import file_sha256


def protocol():
    return {
        "data": {"camera_ids": ["stereo_left", "stereo_right", "stereo_up", "stereo_down"],
                 "image_size": [2, 3]},
        "evaluation": {"reference_index": 2, "depth_range": [2, 80],
                       "roi": [[-100, 100]] * 3, "absolute_thresholds_m": [.2, 1, 2],
                       "distance_bands_m": [[2, 80]]},
    }


def inputs():
    return {
        "rgb": torch.zeros(1, 4, 3, 2, 3),
        "K": torch.eye(3).repeat(1, 4, 1, 1),
        "T_cam_from_rig": torch.eye(4).repeat(1, 4, 1, 1),
        "view_mask": torch.ones(1, 4, dtype=torch.bool),
        "image_mask": torch.ones(1, 4, 2, 3, dtype=torch.bool),
        "camera_ids": [protocol()["data"]["camera_ids"]],
    }


def test_protocol_rejects_mismatched_baseline_reference():
    with pytest.raises(ValueError, match="不一致"):
        evaluate_ruler_direct.validate_protocol(
            protocol(), {"direct": SimpleNamespace(reference_index=2), "baseline": SimpleNamespace(reference_index=3)}
        )


def test_complete_four_view_contract_rejects_missing_or_reordered_camera():
    frame = inputs()
    evaluate_ruler_direct.validate_complete_inputs(frame, protocol())
    frame["view_mask"][0, 1] = False
    with pytest.raises(ValueError, match="缺失"):
        evaluate_ruler_direct.validate_complete_inputs(frame, protocol())
    frame = inputs()
    frame["camera_ids"][0][0], frame["camera_ids"][0][1] = frame["camera_ids"][0][1], frame["camera_ids"][0][0]
    with pytest.raises(ValueError, match="槽位"):
        evaluate_ruler_direct.validate_complete_inputs(frame, protocol())


def test_dev_entry_rejects_test_before_loading_data():
    with pytest.raises(ValueError, match="不读取 TEST"):
        evaluate_ruler_direct.run({"split": "test_complete"}, torch.device("cpu"))


def test_confirmation_rejects_changed_checkpoint_before_loading_data(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256不一致"):
        confirm_ruler_direct.run({"checkpoint": str(checkpoint), "selected_checkpoint_sha256": "wrong"}, torch.device("cpu"))


@pytest.mark.parametrize("invalid_prediction", [False, True])
def test_confirmation_keeps_unknown_pixels_and_fails_on_nonfinite(monkeypatch, tmp_path, invalid_prediction):
    """同一标签/分母评价双方；拒绝标签泄漏，非法预测必须报错而非被过滤。"""
    frame = inputs()
    target = torch.full((1, 4, 2, 3), 4.0)
    target[:, :, 0, 0] = 0
    batch = {"inputs": frame, "targets": {"depth": target, "depth_valid": target > 0}}

    class Dataset:
        frames = [{"sequence": "synthetic", "frame_id": index} for index in range(2)]

        def __len__(self):
            return len(self.frames)

        def __getitem__(self, index):
            return copy.deepcopy(batch)

    class Model(torch.nn.Module):
        reference_index = 2

        def __init__(self, direct):
            super().__init__()
            self.direct = direct

        def forward(self, value):
            assert "targets" not in value and "depth" not in value
            if self.direct:
                depth = value["rgb"].new_full((1, 4, 2, 3), 3.0)
                if invalid_prediction:
                    depth[0, 2, 0, 1] = float("nan")
                return {"dense_depth": depth, "dense_state": torch.full_like(depth, 2),
                        "dense_valid": torch.zeros_like(depth, dtype=torch.bool)}
            return {"depth": value["rgb"].new_full((1, 1, 2, 3), 5.0)}

    def dataset(settings, split, load_targets):
        assert split == "test_complete" and load_targets
        return Dataset()

    monkeypatch.setattr(confirm_ruler_direct.RulerDirectModel, "from_checkpoint", lambda path: Model(True))
    monkeypatch.setattr(confirm_ruler_direct, "SharedPretrainedInitializer", lambda settings: Model(False))
    monkeypatch.setattr(confirm_ruler_direct, "position_caches", lambda model: [])
    monkeypatch.setattr(confirm_ruler_direct, "comparison_dataset", dataset)
    monkeypatch.setattr(confirm_ruler_direct, "collate_frames", lambda rows: rows[0])
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"fixed")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("synthetic fixtures, no real TEST data\n")
    settings = protocol()
    settings["data"]["manifests"] = {"test_complete": str(manifest)}
    configuration = tmp_path / "protocol.json"
    configuration.write_text(json.dumps(settings))
    plan = {
        "checkpoint": str(checkpoint), "selected_checkpoint_sha256": file_sha256(checkpoint),
        "initializer": {"weights": str(checkpoint)}, "protocol": str(configuration),
        "output": str(tmp_path / "results"),
    }
    if invalid_prediction:
        with pytest.raises(FloatingPointError, match="不能过滤"):
            confirm_ruler_direct.run(plan, torch.device("cpu"))
    else:
        result = confirm_ruler_direct.run(plan, torch.device("cpu"))
        for metrics in result["models"].values():
            assert metrics["all"]["counts"]["gt_pixels"] == 10
            assert metrics["all"]["counts"]["accepted_pixels"] == 10
            assert metrics["all"]["MAE_m"] == 1.0
            assert metrics["all"]["coverage"] == 1.0
        with pytest.raises(FileExistsError):
            confirm_ruler_direct.run(plan, torch.device("cpu"))
