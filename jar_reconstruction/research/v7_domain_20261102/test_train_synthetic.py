"""独立dataset包装保持参考、VAL、相机身份和随机流。"""

import copy
import json
from unittest.mock import patch

import torch

from jar.pretraining.data import MixedDataset
from research.v7_domain_20261102.test_synthetic_pair import scene
from research.v7_domain_20261102.train_synthetic import SyntheticDataset, exposure_audit


def item():
    inputs, targets = scene(batch=1)
    inputs = {name: value[0] for name, value in inputs.items()}
    inputs["camera_ids"] = ["stereo_up", "stereo_left", "stereo_right", "stereo_down"]
    inputs["camera_roles"] = torch.zeros(4, dtype=torch.long)
    return {"inputs": inputs, "targets": {name: value[0] for name, value in targets.items()},
            "metadata": {"frame_id": "synthetic-test"}, "domain_index": 0}


def wrapper(tmp_path, split="train", probability=1.):
    dataset = object.__new__(SyntheticDataset)
    dataset.config = {"seed": 11}
    dataset.split = split
    dataset.domains = [{"name": "tiny", "depth_scale": 1.}]
    dataset.synthetic_options = {"probability": probability}
    dataset.audit_folder = tmp_path / "synthetic_exposures"
    return dataset


def test_train_wrapper_preserves_reference_and_ids(tmp_path):
    original = item()
    dataset = wrapper(tmp_path)
    random_state = torch.get_rng_state().clone()
    with patch.object(MixedDataset, "__getitem__", side_effect=lambda _: copy.deepcopy(original)):
        result = dataset[(0, 3, 0)]
    assert torch.equal(torch.get_rng_state(), random_state)
    assert result["inputs"]["camera_ids"] == original["inputs"]["camera_ids"]
    assert result["inputs"]["rgb"].shape == original["inputs"]["rgb"].shape
    assert result["inputs"]["view_mask"].sum() == 2
    assert torch.equal(result["targets"]["depth"][0], original["targets"]["depth"][0])
    assert torch.equal(result["targets"]["depth_valid"][0], original["targets"]["depth_valid"][0])
    assert result["metadata"] == original["metadata"]
    assert result["domain_index"] == 0
    (tmp_path / "plan.json").write_text(json.dumps({"schedule": {"batches": [[[0, 3, 0]]]}}))
    audit = exposure_audit(tmp_path, {"seed": 11, "synthetic_options": {"probability": 1.}})
    assert audit["synthetic"] == 1 and audit["original"] == 0


def test_validation_and_negative_epoch_never_synthesize(tmp_path):
    original = item()
    for split, epoch in (("val", 0), ("train", -1)):
        dataset = wrapper(tmp_path, split)
        with patch.object(MixedDataset, "__getitem__", return_value=original):
            assert dataset[(0, 0, epoch)] is original
    assert not (tmp_path / "synthetic_exposures").exists()


def test_zero_probability_does_not_change_sample(tmp_path):
    original = item()
    dataset = wrapper(tmp_path, probability=0.)
    with patch.object(MixedDataset, "__getitem__", return_value=copy.deepcopy(original)):
        result = dataset[(0, 0, 0)]
    for group in ("inputs", "targets"):
        for key, value in original[group].items():
            if torch.is_tensor(value):
                assert torch.equal(result[group][key], value)
            else:
                assert result[group][key] == value
