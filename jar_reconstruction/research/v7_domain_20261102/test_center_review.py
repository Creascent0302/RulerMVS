"""独立审查投影中心：旋转、部分出界、无效观测与单位。"""

import math

import torch

from research.v7_domain_20261102.center import projective_group_center
from research.v7_domain_20261102.test_center import geometry


def test_rotated_source_exact_mean_projection():
    z = torch.tensor([2., 4., 8.]).reshape(1, 1, 3, 1, 1).requires_grad_()
    p = torch.tensor([.2, .3, .5]).reshape_as(z).requires_grad_()
    data = geometry(forward=.2)
    angle = .03
    c, s = math.cos(angle), math.sin(angle)
    data['T'][0, 1, :3, :3] = torch.tensor([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    data['T'][0, 1, 1, 3] = .04
    center, supported, _ = projective_group_center(z, p, data)
    assert supported.all()
    # 参考格点(0,0)，源投影u/v同时满足加权投影均值。
    denominator = c * z + .2
    u = 100 * (s * z + .06) / denominator
    v = 100 * .04 / denominator
    torch.testing.assert_close(100 * (s * center + .06) / (c * center + .2), (p * u).sum(2))
    torch.testing.assert_close(100 * .04 / (c * center + .2), (p * v).sum(2))
    center.sum().backward()
    assert z.grad.isfinite().all() and p.grad.isfinite().all()


def test_partial_out_of_view_renormalizes_only_visible_samples():
    z = torch.tensor([.1, 4., 8.]).reshape(1, 1, 3, 1, 1)
    p = torch.tensor([.5, .25, .25]).reshape_as(z)
    center, supported, _ = projective_group_center(z, p, geometry())
    expected = torch.full_like(center, 2 / (1 / 4 + 1 / 8))
    torch.testing.assert_close(center, expected)
    assert supported.all()


def test_pure_rotation_behind_camera_and_masked_views_fall_back():
    z = torch.tensor([2., 4., 8.]).reshape(1, 1, 3, 1, 1)
    p = torch.tensor([.2, .3, .5]).reshape_as(z)
    expected = (p * z).sum(2)
    rotated = geometry(0.)
    c, s = math.cos(.03), math.sin(.03)
    rotated['T'][0, 1, :3, :3] = torch.tensor([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    behind = geometry(forward=-20.)
    masked = geometry()
    masked['image_mask'][0, 1] = False
    for data in (rotated, behind, masked):
        center, supported, sensitivity = projective_group_center(z, p, data)
        torch.testing.assert_close(center, expected)
        assert not supported.any()
        assert center.isfinite().all() and sensitivity.isfinite().all()


def test_units_preserve_sensitivity_and_probability_mass():
    z = torch.tensor([2., 4., 8.]).reshape(1, 1, 3, 1, 1)
    p = torch.tensor([.2, .3, .5]).reshape_as(z)
    data = geometry(forward=.4)
    center, supported, sensitivity = projective_group_center(z, p, data)
    data['T'][..., :3, 3] *= .125
    scaled, scaled_supported, scaled_sensitivity = projective_group_center(z * .125, p, data)
    torch.testing.assert_close(scaled / .125, center)
    torch.testing.assert_close(scaled_sensitivity, sensitivity)
    assert torch.equal(scaled_supported, supported)
