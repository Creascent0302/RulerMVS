"""固定候选预算下的粗峰编号排斥；不读取真值。"""

from types import SimpleNamespace

import pytest
import torch

from jar.models.ruler_direct import RulerDirectDecoder
from jar.models.ruler_proposals import transport_proposals
from tests.test_ruler_direct import assert_finite, fixture_inputs, tiny_settings


def proposals(mode, exclusion=None, count=8, peaks=4):
    settings = {"peak_separation_m": .1, "fine_min_step_m": .005,
                "fine_max_step_m": 100., "fine_step_pixels": .5}
    if exclusion is not None:
        settings["peak_exclusion_bins"] = exclusion
    model = SimpleNamespace(settings=settings, peak_count=peaks, proposal_transport=mode,
                            depth_lattice=torch.arange(1, count + 1).float(), near=1., far=float(count))
    scores = torch.arange(count, 0, -1).float()[None, :, None, None].expand(1, -1, 2, 2)
    depth = model.depth_lattice[None, :, None, None].expand_as(scores).clone().requires_grad_()
    coarse = {"scores": scores, "probability": scores.softmax(1), "candidate_depth": depth,
              "valid": torch.ones_like(scores, dtype=torch.bool), "selected_index": scores.argmax(1, keepdim=True)}
    features = torch.randn(1, 4, count, 2, 2, requires_grad=True)
    observation = {"ray_derivative": torch.ones(1, 2, 2, count, 2, 2),
                   "source_valid": torch.ones(1, 2, count, 2, 2, dtype=torch.bool)}
    if mode == "nearest":
        output = RulerDirectDecoder._fine_proposals(model, coarse, features, observation)
    else:
        output = transport_proposals(model, coarse, features, observation, {})
    return output, depth, features


@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
def test_default_zero_keeps_original_peaks(mode):
    torch.manual_seed(13)
    legacy, _, _ = proposals(mode)
    torch.manual_seed(13)
    zero, _, _ = proposals(mode, exclusion=0)
    for before, after in zip(legacy, zero):
        assert torch.equal(before, after)
    torch.testing.assert_close(zero[0][0, 1::3, 0, 0], torch.tensor([1., 2., 3., 4.]))


@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
def test_radius_one_selects_nonadjacent_bins_and_keeps_gradients(mode):
    output, depth, features = proposals(mode, exclusion=1)
    fine, half_width, inherited = output
    torch.testing.assert_close(fine[0, 1::3, 0, 0], torch.tensor([1., 3., 5., 7.]))
    assert fine.shape == (1, 12, 4, 4)
    assert torch.isfinite(half_width).all()
    (fine.sum() + inherited.sum()).backward()
    for value in (depth, features):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
def test_exhausted_small_lattice_falls_back_safely(mode):
    output, _, _ = proposals(mode, exclusion=8, count=3, peaks=3)
    fine, half_width, inherited = output
    assert fine.shape[1] == 9
    for value in output:
        assert torch.isfinite(value).all()
    torch.testing.assert_close(fine[0, 1::3, 0, 0], torch.ones(3))


@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
def test_full_decoder_with_exclusion_forward_backward(mode):
    model = RulerDirectDecoder(tiny_settings(use_scene=False, local_kind="volume",
                                            proposal_transport=mode, peak_exclusion_bins=1))
    inputs, _ = fixture_inputs(batch=2, views=3)
    inputs["view_mask"][1] = False
    result = model(inputs)
    assert_finite(result)
    loss = result["depth"].mean() + result["fine"]["scores"][0].mean()
    loss.backward()
    assert model.score.weight.grad.abs().sum() > 0
    assert model.offset.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
