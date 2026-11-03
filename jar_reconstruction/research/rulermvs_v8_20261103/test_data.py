"""V8数据契约CPU测试：确定性、干净曝光、无几何改动与范围覆盖。"""

import copy
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from jar.pretraining.data import collate_mixed
from research.overnight_v6_v7_mvs_20261103.common_data import SharedDataset as LegacyDataset
from research.rulermvs_v8_20261103.data import SharedDataset, compose_training_domain


def sample(views=4):
    rgb = torch.linspace(.05, .95, views * 3 * 12 * 16).reshape(views, 3, 12, 16)
    return {"inputs": {
        "rgb": rgb, "matching_rgb": F.interpolate(rgb, (24, 32), mode="bilinear", align_corners=False),
        "K": torch.eye(3).repeat(views, 1, 1), "T_cam_from_rig": torch.eye(4).repeat(views, 1, 1),
        "view_mask": torch.ones(views, dtype=torch.bool),
        "image_mask": torch.ones(views, 12, 16, dtype=torch.bool),
        "matching_image_mask": torch.ones(views, 24, 32, dtype=torch.bool),
    }, "targets": {"depth": torch.linspace(2., 8., views * 12 * 16).reshape(views, 12, 16),
                   "depth_valid": torch.ones(views, 12, 16, dtype=torch.bool)}}


class FakeDataset:
    def __init__(self, views=4):
        self.views = views

    def __len__(self):
        return 32

    def __getitem__(self, index):
        return sample(self.views)


def config():
    return {"seed": 19, "augmentation": True, "image_size": [12, 16], "matching_image_size": [24, 32],
            "datasets": [{"name": "toy", "root": "", "num_views": 4, "reference_index": 2,
                          "geometry_units": "meters", "manifests": {}}],
            "photometric_augmentation": {"enabled": True, "clean_probability": .4},
            "range_augmentation": {"enabled": True, "nominal_probability": .4, "broad_probability": .2}}


def datasets(cfg=None, views=4, split="train"):
    cfg = config() if cfg is None else cfg
    with patch("jar.pretraining.data.build_dataset", return_value=FakeDataset(views)):
        return SharedDataset(cfg, split), LegacyDataset(dict(cfg, augmentation=False), split)


def assert_inputs_equal(left, right):
    for name, value in left.items():
        torch.testing.assert_close(value, right[name], rtol=0, atol=0)


def test_clean_and_validation_exact_and_metadata_survives_collation():
    cfg = config()
    cfg["photometric_augmentation"]["clean_probability"] = 1.
    dataset, baseline = datasets(cfg)
    item, original = dataset[0, 3, 2], baseline[0, 3, 2]
    assert_inputs_equal(item["inputs"], original["inputs"])
    assert_inputs_equal(item["targets"], original["targets"])
    assert item["metadata"]["v8_augmentation"]["clean"]
    batch = collate_mixed([item, dataset[0, 4, 2]])
    assert batch["metadata"][0]["v8_augmentation"]["key"] == [0, 3, 2]
    validation, baseline = datasets(config(), split="val")
    assert_inputs_equal(validation[0, 3, -1]["inputs"], baseline[0, 3, -1]["inputs"])
    assert "depth" not in validation[0, 3, -1]["inputs"]


def test_augmented_deterministic_consistent_rgb_and_geometry_labels_untouched():
    cfg = config()
    cfg["photometric_augmentation"].update(clean_probability=0., source_noise_probability=1.,
        source_blur_probability=1., source_occlusion_probability=1.)
    dataset, baseline = datasets(cfg)
    original = baseline[0, 3, 2]
    before_rng = torch.get_rng_state().clone()
    first, second = dataset[0, 3, 2], dataset[0, 3, 2]
    assert torch.equal(before_rng, torch.get_rng_state())
    assert_inputs_equal(first["inputs"], second["inputs"])
    for name in ("K", "T_cam_from_rig", "view_mask", "image_mask", "matching_image_mask"):
        assert torch.equal(first["inputs"][name], original["inputs"][name])
    assert_inputs_equal(first["targets"], original["targets"])
    assert not torch.equal(first["inputs"]["rgb"], original["inputs"]["rgb"])
    expected = F.interpolate(first["inputs"]["rgb"], (24, 32), mode="bilinear", align_corners=False)
    assert torch.equal(first["inputs"]["matching_rgb"], expected)
    assert not torch.equal(first["inputs"]["rgb"], dataset[0, 3, 3]["inputs"]["rgb"])


def test_source_corruptions_never_corrupt_reference_without_shared_style():
    cfg = config()
    cfg["photometric_augmentation"].update(clean_probability=0., gamma=[1., 1.], contrast=[1., 1.],
        saturation=[1., 1.], color_balance=[1., 1.], source_noise_probability=1.,
        source_blur_probability=1., source_occlusion_probability=1.)
    dataset, baseline = datasets(cfg)
    # 浮点gray + (rgb-gray)容许一ulp，不把源增强误施于reference。
    torch.testing.assert_close(dataset[0, 3, 2]["inputs"]["rgb"][0],
                               baseline[0, 3, 2]["inputs"]["rgb"][0], rtol=0, atol=1e-7)


@pytest.mark.parametrize("views", [2, 3, 4])
def test_source_dropout_preserves_reference_and_one_source_and_gt(views):
    cfg = config()
    cfg["photometric_augmentation"].update(clean_probability=0., source_dropout_probability=1.)
    dataset, baseline = datasets(cfg, views)
    result = dataset[0, 1, 0]
    assert result["inputs"]["view_mask"][0]
    assert result["inputs"]["view_mask"][1:].sum() >= 1
    assert result["inputs"]["view_mask"].sum() == (2 if views == 2 else views - 1)
    assert_inputs_equal(result["targets"], baseline[0, 1, 0]["targets"])
    dropped = ~result["inputs"]["view_mask"]
    for name in ("rgb", "matching_rgb", "image_mask", "matching_image_mask"):
        assert not result["inputs"][name][dropped].any()


def test_clean_probability_retains_substantial_unmodified_exposures():
    dataset, _ = datasets()
    flags = [dataset[0, i, 0]["metadata"]["v8_augmentation"]["clean"] for i in range(256)]
    assert .3 <= sum(flags) / len(flags) <= .5


def training_batch():
    dataset, _ = datasets()
    result = collate_mixed([dataset[0, 0, 0], dataset[0, 1, 0]])
    result["targets"]["depth"][0, 0, 0, 0] = 1e20
    result["targets"]["depth_valid"][0, 0, 0, 0] = False
    return result


DOMAIN = {"name": "toy", "near": 1., "far": 10., "depth_scale": 2.}


@pytest.mark.parametrize("mode,probabilities", [("nominal", (1., 0.)), ("broad", (0., 1.)),
                                               ("gt_padding", (0., 0.))])
def test_range_all_branches_cover_truth_reproduce_and_preserve_inputs(mode, probabilities):
    cfg = config()
    cfg["range_augmentation"].update(nominal_probability=probabilities[0], broad_probability=probabilities[1],
                                     unit_jitter_probability=1., unit_scale_bounds=[.25, 4.])
    batch = training_batch()
    before = copy.deepcopy(batch)
    output = compose_training_domain(batch, DOMAIN, cfg, 12, 9)
    assert output["range_augmentation_mode"] == mode
    assert output == compose_training_domain(batch, DOMAIN, cfg, 12, 9999)
    assert .5 <= output["depth_scale"] <= 8.
    assert output["nominal_depth_scale"] == 2.
    truth = batch["targets"]["depth"][:, 0][batch["targets"]["depth_valid"][:, 0]]
    assert output["near"] <= truth.min() and output["far"] >= truth.max()
    assert_inputs_equal(batch["inputs"], before["inputs"])
    assert_inputs_equal(batch["targets"], before["targets"])
    assert DOMAIN == {"name": "toy", "near": 1., "far": 10., "depth_scale": 2.}


def test_range_coverage_repair_outliers_empty_gt_and_safety_fail_loudly():
    cfg = config()
    cfg["range_augmentation"].update(nominal_probability=0., broad_probability=1., broad_range=[3., 4.])
    batch = training_batch()
    result = compose_training_domain(batch, DOMAIN, cfg, 0, 0)
    assert result["range_was_repaired"]
    assert result["near"] < 3 and result["far"] > 4
    batch["targets"]["depth_valid"][0, 0, 0, 0] = True
    with pytest.raises(ValueError, match="有效GT超出"):
        compose_training_domain(batch, DOMAIN, cfg, 0, 0)
    batch["targets"]["depth_valid"].zero_()
    with pytest.raises(ValueError, match="没有有效参考GT"):
        compose_training_domain(batch, DOMAIN, cfg, 0, 0)


def test_val_or_disabled_range_never_reads_targets_and_inference_has_no_label_dependency():
    cfg = config()
    no_labels = {"inputs": {}, "metadata": [{"v8_augmentation": {"training": False}}]}
    assert compose_training_domain(no_labels, DOMAIN, cfg, 0, 0) == DOMAIN
    assert compose_training_domain({"split": "inference"}, DOMAIN, cfg, 0, 0) == DOMAIN
    assert compose_training_domain({}, DOMAIN, {}, 0, 0) == DOMAIN


def test_duplicate_batch_size_does_not_change_range_random_distribution():
    cfg = config()
    batch = training_batch()
    doubled = {"metadata": batch["metadata"] * 2,
               **{name: {k: torch.cat([v, v]) for k, v in batch[name].items()}
                  for name in ("inputs", "targets")}}
    for update in range(20):
        assert compose_training_domain(batch, DOMAIN, cfg, update, 0) == compose_training_domain(
            doubled, DOMAIN, cfg, update, 123)


def test_invalid_probabilities_and_ranges_rejected():
    cfg = config()
    cfg["range_augmentation"].update(nominal_probability=.8, broad_probability=.8)
    with pytest.raises(ValueError, match="不能超过1"):
        compose_training_domain(training_batch(), DOMAIN, cfg, 0, 0)
    cfg["photometric_augmentation"]["clean_probability"] = -1
    with pytest.raises(ValueError, match="clean_probability"):
        datasets(cfg)
