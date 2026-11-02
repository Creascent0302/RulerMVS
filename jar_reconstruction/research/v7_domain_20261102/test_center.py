"""投影中心的解析性质和可拆卸模型契约。"""

import copy
from unittest.mock import patch

import torch

from research.early_geometry_20261030.test_encoding import TinyViT, inputs
from research.rulermvs_v5_20261031.test_bridge import tiny_pair
from research.rulermvs_v6_20261101.model import V6Model
from research.rulermvs_v6_20261101.objective import objective
from .center import projective_group_center
from .model import build_model


def geometry(baseline=.06, forward=0.):
    K = torch.tensor([[100., 0., 0.], [0., 100., 0.], [0., 0., 1.]])
    T = torch.eye(4).repeat(1, 2, 1, 1)
    T[0, 1, 0, 3] = baseline
    T[0, 1, 2, 3] = forward
    return {"reference": 0, "K": K.repeat(1, 2, 1, 1), "T": T,
            "image_size": (32, 32), "view_mask": torch.ones(1, 2, dtype=torch.bool),
            "image_mask": torch.ones(1, 2, 32, 32, dtype=torch.bool)}


def test_parallel_center_is_harmonic():
    z = torch.tensor([2., 4., 8.]).reshape(1, 1, 3, 1, 1).requires_grad_()
    p = torch.tensor([.2, .3, .5]).reshape_as(z)
    center, valid, sensitivity = projective_group_center(z, p, geometry())
    torch.testing.assert_close(center, (p / z).sum(2).reciprocal())
    assert valid.all() and (sensitivity > 0).all()
    center.sum().backward()
    assert torch.isfinite(z.grad).all() and (z.grad > 0).all()


def test_forward_translation_exact_projection_and_units():
    z = torch.tensor([2., 4., 8.]).reshape(1, 1, 3, 1, 1)
    p = torch.full_like(z, 1 / 3)
    data = geometry(forward=.4)
    center, _, _ = projective_group_center(z, p, data)
    torch.testing.assert_close(6 / (center + .4), (p * 6 / (z + .4)).sum(2))
    data["T"][..., :3, 3] *= 7
    scaled, _, _ = projective_group_center(z * 7, p, data)
    torch.testing.assert_close(scaled / 7, center)


def test_no_baseline_or_no_source_falls_back_without_nan():
    z = torch.tensor([2., 4., 8.]).reshape(1, 1, 3, 1, 1)
    p = torch.full_like(z, 1 / 3)
    for data in (geometry(0.), geometry()):
        if data["T"][0, 1, 0, 3] != 0:
            data["view_mask"][0, 1] = False
        center, valid, _ = projective_group_center(z, p, data)
        torch.testing.assert_close(center, (z * p).sum(2))
        assert not valid.any() and torch.isfinite(center).all()


def test_zero_init_matches_v6_and_receives_gradient():
    torch.set_num_threads(2)
    torch.manual_seed(61)
    _, old = tiny_pair()
    cfg = copy.deepcopy(old.config)
    cfg.update(architecture="rulermvs_v6", handoff={"mode": "parent"})
    with patch("jar.pretraining.encoders._build_native_foundation", side_effect=TinyViT):
        v6 = V6Model(cfg, initialize=False).eval()
        v6.load_v5_state(old.state_dict())
        v7, _ = build_model({"architecture": "rulermvs_v6", "config": cfg,
                             "model": v6.state_dict()}, "projective")
    v7.eval()
    domain = {"near": 1., "far": 10., "depth_scale": 1.}
    data = inputs()
    with torch.no_grad():
        a, b = v6(data, domain), v7(data, domain)
    assert torch.equal(a["depth"], b["depth"])
    assert torch.equal(a["fine"]["candidate_depth"], b["fine"]["candidate_depth"])
    targets = {"depth": torch.full_like(data["image_mask"], 3., dtype=torch.float32),
               "depth_valid": data["image_mask"].clone()}
    prediction = v7(data, domain)
    objective(prediction, data, targets, domain, {})["total"].backward()
    grads = [p.grad for p in v7.decoder.center.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any((g != 0).any() for g in grads)
