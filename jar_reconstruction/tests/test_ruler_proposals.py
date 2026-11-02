"""子射线候选传递的覆盖、相位及梯度。"""

import pytest
import torch

from jar.models.ruler_direct import RulerDirectDecoder
from jar.models.ruler_proposals import SubrayTransport
from tests.test_ruler_direct import fixture_inputs, tiny_settings, assert_finite


def test_transport_preserves_two_surfaces_without_averaging_their_depths():
    probability = torch.tensor([[[[0.9, 0.1]], [[0.1, 0.9]]]])
    depth = torch.tensor([3., 30.]).reshape(1, 2, 1, 1).expand_as(probability)
    coarse = dict(probability=probability, candidate_depth=depth,
                  valid=torch.ones_like(depth, dtype=torch.bool))
    transport = SubrayTransport(coarse, {}, "bilinear")
    torch.testing.assert_close(transport.probability[..., ::2, ::2], probability)
    torch.testing.assert_close(transport.probability[0, :, 0, 1], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(transport.depth[0, :, 0, 1], torch.tensor([3., 30.]))
    selected = torch.tensor([0, 1, 0, 1]).reshape(1, 1, 1, 4).expand(-1, -1, 2, -1)
    result = transport.selected(depth[:, None], selected)
    torch.testing.assert_close(result[:, 0], transport.depth.gather(1, selected))


@pytest.mark.parametrize("mode", ["bilinear", "appearance"])
@pytest.mark.parametrize("views", [2, 4])
def test_transport_forward_backward_masks(mode, views):
    model = RulerDirectDecoder(tiny_settings(use_scene=False, local_kind="volume",
                                            proposal_transport=mode, use_fine_offset=False))
    inputs, _ = fixture_inputs(batch=2, views=views)
    inputs["view_mask"][1] = False
    inputs["matching_features"][1] = float("nan")
    result = model(inputs)
    assert_finite(result)
    assert result["fine"]["scores"].shape == (2, 12, 8, 12)
    assert not result["valid"][1].any()
    loss = result["depth"].mean() + result["fine"]["scores"][0].mean()
    loss.backward()
    assert model.offset.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_default_stays_nearest():
    model = RulerDirectDecoder(tiny_settings())
    assert model.proposal_transport == "nearest"


def test_extremely_unlikely_candidates_keep_metric_units():
    probability = torch.tensor([1.0, 1e-20, 1e-40, 0.0]).reshape(1, 4, 1, 1).expand(-1, -1, 2, 2)
    depth = torch.tensor([3., 30., 50., 60.]).reshape(1, 4, 1, 1).expand_as(probability)
    transport = SubrayTransport(dict(probability=probability, candidate_depth=depth,
                                    valid=torch.ones_like(depth, dtype=torch.bool)), {}, "bilinear")
    torch.testing.assert_close(transport.depth, depth.repeat_interleave(2, -2).repeat_interleave(2, -1))
