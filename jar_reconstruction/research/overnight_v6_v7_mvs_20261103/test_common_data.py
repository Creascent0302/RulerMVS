"""共同曝光、增强重现性与仅 TRAIN 使用 GT 的范围变换测试。"""

import copy
import itertools
from unittest.mock import patch

import pytest
import torch

from research.overnight_v6_v7_mvs_20261103.common_data import (
    SharedDataset, make_sampler, own_training_domain,
)


def config(world_size=2):
    return {
        "seed": 20261103,
        "augmentation": True,
        "image_size": [4, 6],
        "matching_image_size": [8, 12],
        "train": {"batch_size": 4, "accumulation": 8 // world_size, "global_batch_size": 32},
        "datasets": [dict(name=f"domain{i}", root="", num_views=4, reference_index=2,
                          geometry_units="meters", train_multiplier=m, manifests={})
                     for i, m in enumerate((1., 1.5))],
        "range_augmentation": {"enabled": True, "nominal_probability": .5,
                               "unit_jitter_probability": .5},
    }


def sample():
    return {"inputs": {
        "rgb": torch.full((4, 3, 4, 6), .5),
        "matching_rgb": torch.full((4, 3, 8, 12), .5),
        "K": torch.eye(3).expand(4, -1, -1).clone(),
        "T_cam_from_rig": torch.eye(4).expand(4, -1, -1).clone(),
        "view_mask": torch.ones(4, dtype=torch.bool),
        "image_mask": torch.ones(4, 4, 6, dtype=torch.bool),
        "matching_image_mask": torch.ones(4, 8, 12, dtype=torch.bool),
        "camera_ids": ["a", "b", "c", "d"],
    }, "targets": {
        "depth": torch.full((4, 4, 6), 3.),
        "depth_valid": torch.ones(4, 4, 6, dtype=torch.bool),
    }}


class FakeDataset:
    def __len__(self):
        return 51

    def __getitem__(self, index):
        return sample()


def global_keys(config, lengths, world_size, epoch=2):
    ranks = [iter(make_sampler(config, lengths, epoch, rank, world_size))
             for rank in range(world_size)]
    return list(itertools.chain.from_iterable(
        batch for micro in zip(*ranks) for batch in micro))


def test_two_and_four_cards_have_identical_global_sample_and_augmentation_order():
    lengths = [53, 79]
    two = config(2)
    four = config(4)
    assert global_keys(two, lengths, 2) == global_keys(four, lengths, 4)
    larger_microbatch = config(2)
    larger_microbatch["train"].update(batch_size=8, accumulation=2)
    assert global_keys(larger_microbatch, lengths, 2) == global_keys(four, lengths, 4)
    sampler = make_sampler(two, lengths, 2, 0, 2)
    assert sampler.domain_counts[0]["requested_exposures"] == 53
    assert sampler.domain_counts[1]["requested_exposures"] == 119
    expected = list(sampler)[sampler.accumulation:]
    assert expected == list(make_sampler(two, lengths, 2, 0, 2, start_update=1))
    four["train"]["accumulation"] = 1
    with pytest.raises(ValueError, match="共同协议"):
        make_sampler(four, lengths, 2, 0, 4)


def test_brightness_is_reproducible_without_source_dropout_and_val_is_unaugmented():
    with patch("jar.pretraining.data.build_dataset", return_value=FakeDataset()):
        train = SharedDataset(config(), "train")
        validation = SharedDataset(config(), "val")
    for index in range(51):
        item = train[0, index, 3]
        repeated = train[0, index, 3]
        assert torch.equal(item["inputs"]["rgb"], repeated["inputs"]["rgb"])
        assert item["inputs"]["view_mask"].all()
        assert item["inputs"]["image_mask"].all()
        assert torch.equal(item["targets"]["depth"], sample()["targets"]["depth"])
        assert torch.equal(item["inputs"]["rgb"][:, :, 0, 0],
                           item["inputs"]["matching_rgb"][:, :, 0, 0])
    assert train[0, 0, 3]["inputs"]["camera_ids"][0] == "d"
    assert validation[0, 0, -1]["inputs"]["camera_ids"][0] == "c"
    assert torch.equal(validation[0, 0, -1]["inputs"]["rgb"], sample()["inputs"]["rgb"])


def batch():
    result = sample()
    for collection in result.values():
        for name, value in collection.items():
            if torch.is_tensor(value):
                collection[name] = value.unsqueeze(0)
    result["targets"]["depth"][0, 0, 0, 0] = 2.
    result["targets"]["depth"][0, 0, 0, 1] = 8.
    # 此值被有效性掩码排除，不能扩大训练范围。
    result["targets"]["depth"][0, 0, 0, 2] = 9999.
    result["targets"]["depth_valid"][0, 0, 0, 2] = False
    return result


def test_training_domain_expands_only_valid_reference_gt_and_preserves_batch():
    cfg = config()
    cfg["range_augmentation"].update(nominal_probability=0., unit_jitter_probability=1.)
    original = batch()
    before = copy.deepcopy(original)
    domain = {"near": .1, "far": 100., "depth_scale": 2.}
    result = own_training_domain(original, domain, cfg, update=12, seed=9)
    assert 1 <= result["near"] <= 2
    assert 8 <= result["far"] <= 16
    assert 1 <= result["depth_scale"] <= 4
    assert result == own_training_domain(original, domain, cfg, update=12, seed=9)
    assert domain == {"near": .1, "far": 100., "depth_scale": 2.}
    for group in original:
        for name, value in original[group].items():
            if torch.is_tensor(value):
                assert torch.equal(value, before[group][name])


def test_disabled_policy_needs_no_targets_and_empty_train_truth_fails():
    domain = {"near": .1, "far": 100., "depth_scale": 1.}
    assert own_training_domain({"inputs": {}}, domain, {}, 0, 9) == domain
    cfg = config()
    cfg["range_augmentation"].update(nominal_probability=1., unit_jitter_probability=0.)
    result = own_training_domain(batch(), domain, cfg, 0, 9)
    assert (result["near"], result["far"], result["depth_scale"]) == (.1, 100., 1.)
    invalid = batch()
    invalid["inputs"]["view_mask"][0, 0] = False
    with pytest.raises(ValueError, match="没有有效 GT"):
        own_training_domain(invalid, domain, cfg, 0, 9)
    cfg["range_augmentation"]["near_multiplier"] = [1., 2.]
    with pytest.raises(ValueError, match="向外扩展"):
        own_training_domain(batch(), domain, cfg, 0, 9)
