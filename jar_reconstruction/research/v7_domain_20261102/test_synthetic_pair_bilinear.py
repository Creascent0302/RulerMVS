"""亚像素光度、遮挡不混合、明确监督映射的解析测试。"""

import copy

import pytest
import torch

from research.v7_domain_20261102.synthetic_pair import forward_splat as nearest_splat
from research.v7_domain_20261102.synthetic_pair_bilinear import (
    augment_training_batch, correspondence_targets, forward_splat, synthesize_pair,
)
from research.v7_domain_20261102.test_synthetic_pair import scene


@pytest.mark.parametrize("displacement", [.25, .5, .75])
def test_subpixel_plane_photometry(displacement):
    inputs, targets = scene()
    depth = torch.full_like(targets["depth"][:, 0], 8.)
    valid = torch.ones_like(depth, dtype=torch.bool)
    reference = inputs["rgb"][:, 0]
    tx = torch.full((2,), displacement * 8. / 32.)
    bilinear = forward_splat(reference, depth, valid, inputs["K"][:, 0], tx)
    nearest = nearest_splat(reference, depth, valid, inputs["K"][:, 0], tx)
    # 同一fronto-parallel平面，平移后对应分段线性图像的解析值。
    expected = (1. - displacement) * reference[..., 1:] + displacement * reference[..., :-1]
    selected = bilinear["valid"][..., 1:] & nearest["valid"][..., 1:]
    error_new = (bilinear["rgb"][..., 1:] - expected).abs().permute(0, 2, 3, 1)[selected].mean()
    error_old = (nearest["rgb"][..., 1:] - expected).abs().permute(0, 2, 3, 1)[selected].mean()
    assert float(error_new) < 1e-6
    assert float(error_new) < float(error_old) * .01
    assert float(error_old) > .05
    torch.testing.assert_close(bilinear["depth"][bilinear["valid"]],
                               torch.full_like(bilinear["depth"][bilinear["valid"]], 8.))


def test_foreground_and_background_never_mix():
    inputs, targets = scene()
    depth = targets["depth"][:, 0]
    rgb = torch.zeros_like(inputs["rgb"][:, 0])
    rgb[:, 0] = (depth == 4).float()
    rgb[:, 2] = (depth == 8).float()
    result = forward_splat(rgb, depth, torch.ones_like(depth, dtype=torch.bool),
                            inputs["K"][:, 0], torch.full((2,), -.13))
    valid = result["valid"]
    error = torch.minimum((result["depth"] - 4.).abs(), (result["depth"] - 8.).abs())
    assert error[valid].max() < 1e-6
    red = result["rgb"][:, 0]
    assert torch.minimum(red.abs(), (red - 1.).abs())[valid].max() < 1e-6
    assert result["occluded_contributions"].min() > 0
    assert (~valid).any()
    assert not result["rgb"].permute(0, 2, 3, 1)[~valid].any()
    assert not result["depth"][~valid].any()
    assert result["centroid_error"][valid].max() <= .25


def test_mapping_reconstructs_labels_and_colour():
    inputs, targets = scene()
    result = forward_splat(inputs["rgb"][:, 0], targets["depth"][:, 0],
                            targets["depth_valid"][:, 0], inputs["K"][:, 0],
                            torch.full((2,), -.13))
    mapping = result["mapping"]
    index, weight = mapping["source_index"], mapping["weight"]
    sums = torch.zeros(2, 16 * 24).scatter_add_(1, index, weight)
    torch.testing.assert_close(sums, result["valid"].flatten(1).float())
    original = inputs["rgb"][:, 0].flatten(2)
    colours = original.gather(2, mapping["reference_index"][:, None].expand(-1, 3, -1))
    reconstruction = torch.zeros_like(original).scatter_add_(
        2, index[:, None].expand(-1, 3, -1), colours * weight[:, None])
    torch.testing.assert_close(reconstruction.reshape_as(result["rgb"]), result["rgb"])
    inverse = torch.zeros_like(sums).scatter_add_(1, index, weight / mapping["source_depth"])
    reconstructed_depth = inverse.clamp_min(1e-12).reciprocal().reshape_as(result["depth"])
    torch.testing.assert_close(reconstructed_depth[result["valid"]], result["depth"][result["valid"]])


def test_input_contract_units_and_original_preservation():
    inputs, targets = scene()
    original_inputs, original_targets = copy.deepcopy(inputs), copy.deepcopy(targets)
    first, supervision, record = synthesize_pair(
        inputs, targets, {"depth_scale": 1.}, generator=torch.Generator().manual_seed(2))
    updated = correspondence_targets(targets, supervision)
    assert torch.equal(record["retained_views"], torch.tensor([2, 2]))
    assert torch.equal(updated["depth"][:, 0], targets["depth"][:, 0])
    assert torch.equal(updated["depth"][:, 1], supervision["source_depth"])
    for key, value in original_inputs.items():
        assert torch.equal(value, inputs[key])
    for key, value in original_targets.items():
        assert torch.equal(value, targets[key])
    scaled_targets = copy.deepcopy(targets)
    scaled_targets["depth"] *= 100.
    scaled, scaled_supervision, _ = synthesize_pair(
        inputs, scaled_targets, {"depth_scale": 100.}, generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(first["rgb"], scaled["rgb"], atol=1e-5, rtol=1e-5)
    assert torch.equal(first["image_mask"], scaled["image_mask"])
    torch.testing.assert_close(supervision["source_depth"] * 100., scaled_supervision["source_depth"])


def test_subpixel_hole_unsupported_fringe_is_not_filled():
    inputs, targets = scene()
    valid = torch.zeros_like(targets["depth"][:, 0], dtype=torch.bool)
    valid[:, 6, 9] = True
    result = forward_splat(inputs["rgb"][:, 0], targets["depth"][:, 0], valid,
                            inputs["K"][:, 0], torch.full((2,), .0625))
    # 单点正好落在半像素，左右均没有足够连续表面支持，保持未知。
    assert not result["valid"].any()
    assert not result["mapping"]["weight"].any()
    assert torch.isfinite(result["rgb"]).all()


def test_training_wrapper_mixed_batch_and_independent_rng():
    inputs, targets = scene(batch=8)
    before_rng = torch.get_rng_state().clone()
    augmented, updated, record = augment_training_batch(
        inputs, targets, {"depth_scale": 1.}, generator=torch.Generator().manual_seed(8))
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert record["synthetic_count"] + record["original_count"] == 8
    assert 0 < record["synthetic_count"] < 8
    original = ~record["active"]
    for key, value in inputs.items():
        assert torch.equal(value[original], augmented[key][original])
    for key, value in targets.items():
        assert torch.equal(value[original], updated[key][original])
    assert torch.equal(updated["depth"][:, 0], targets["depth"][:, 0])
