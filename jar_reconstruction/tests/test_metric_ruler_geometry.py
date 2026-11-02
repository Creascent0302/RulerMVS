"""解析光滑表面、公制导数与必要条件反例；不证明真实重建精度。"""

import pytest
import torch
import torch.nn.functional as F

from jar.geometry import project, unproject
from jar.models.metric_ruler_geometry import (
    feature_spatial_gradients,
    metric_jet_statistics,
    pullback_feature_jets,
)


def cameras(scale=1.0):
    intrinsic = torch.tensor([[120.0, 0, 64], [0, 100.0, 48], [0, 0, 1.0]], dtype=torch.float64)
    centers = scale * torch.tensor(
        [[0.0, 0, 0], [0.6, 0, 0], [0, 0.5, 0], [-0.5, 0.2, 0]], dtype=torch.float64
    )
    transform = torch.eye(4, dtype=torch.float64).repeat(4, 1, 1)
    transform[:, :3, 3] = -centers
    return intrinsic.repeat(4, 1, 1), transform, centers


def texture_and_derivative(points, affine=False):
    x, y = points[..., 0], points[..., 1]
    zero = torch.zeros_like(x)
    one = torch.ones_like(x)
    if affine:
        values = torch.stack((x + 0.2 * y, 0.3 * x - y), -1)
        derivative = torch.stack(
            (torch.stack((one, 0.2 * one, zero), -1), torch.stack((0.3 * one, -one, zero), -1)), -2
        )
        return values, derivative
    values = torch.stack((x + 2, y + 3, x * y + 1, x.square() + 1, y.square() + 2, (x + 0.7 * y).sin()), -1)
    derivative = torch.stack(
        (
            torch.stack((one, zero, zero), -1),
            torch.stack((zero, one, zero), -1),
            torch.stack((y, x, zero), -1),
            torch.stack((2 * x, zero, zero), -1),
            torch.stack((zero, 2 * y, zero), -1),
            torch.stack(((x + 0.7 * y).cos(), 0.7 * (x + 0.7 * y).cos(), zero), -1),
        ),
        -2,
    )
    length = values.norm(dim=-1, keepdim=True)
    normalized = values / length
    # 对通道单位化后的真实纹理求导；不是对未归一化梯度再独立归一化。
    projection = (normalized[..., :, None] * derivative).sum(-2, keepdim=True)
    derivative = (derivative - normalized[..., :, None] * projection) / length[..., None]
    return normalized, derivative


def plane_observations(candidates, affine=False):
    """所有图像来自 z=5 的同一纹理平面，候选可以不位于该表面。"""
    intrinsic, transform, centers = cameras()
    uv, _, valid, jacobian = project(candidates, intrinsic, transform, (96, 128))
    surface = centers[:, None] + (candidates[None] - centers[:, None]) * (5 / candidates[None, :, 2:])
    appearance, derivative = texture_and_derivative(surface, affine)
    image_to_surface = candidates.new_tensor([[5 / 120, 0], [0, 5 / 100], [0, 0]])
    pixel_gradient = derivative @ image_to_surface
    jets = pullback_feature_jets(pixel_gradient, jacobian)
    _, ray = unproject(uv[0], candidates[:, 2], intrinsic[0], transform[0])
    parallax = (jacobian[1:] * ray[None, :, None]).sum(-1)
    return appearance, pixel_gradient, jacobian, jets, valid, parallax


def statistics(candidates, affine=False, **kwargs):
    appearance, _, _, jets, valid, parallax = plane_observations(candidates, affine)
    result = metric_jet_statistics(jets[0], jets[1:], valid[1:], parallax, **kwargs)
    return appearance, result


def test_gradients_have_original_pixel_units():
    y, x = torch.meshgrid(torch.arange(7.0), torch.arange(9.0), indexing="ij")
    features = torch.stack((3 * x + 2 * y, x - 4 * y))[None]
    gradient = feature_spatial_gradients(features, image_scale=(2, 3), feature_stride=4, normalize=False)
    assert gradient.shape == (1, 2, 2, 7, 9)
    expected = features.new_tensor([[1.5, 1.5], [0.5, -3]])
    torch.testing.assert_close(gradient[0, :, :, 1:-1, 1:-1], expected[:, :, None, None].expand(2, 2, 5, 7))


def test_feature_normalization_precedes_finite_differences():
    torch.manual_seed(3)
    features = torch.rand(2, 5, 6, 8, dtype=torch.float64)
    gradients = feature_spatial_gradients(features)
    normalized = F.normalize(features, dim=1)
    torch.testing.assert_close(gradients[:, :, 0, :, 1:-1], (normalized[..., 2:] - normalized[..., :-2]) / 2)
    torch.testing.assert_close(gradients[:, :, 1, 1:-1], (normalized[..., 2:, :] - normalized[..., :-2, :]) / 2)


def test_pullback_matches_finite_difference_of_real_plane_images():
    candidates = torch.tensor([[0.2, -0.3, 5.0], [-0.4, 0.1, 4.0]], dtype=torch.float64)
    _, _, _, jets, _, _ = plane_observations(candidates)
    intrinsic, transform, centers = cameras()
    step = 1e-5
    for axis in range(3):
        displacement = torch.zeros_like(candidates)
        displacement[:, axis] = step
        values = []
        for trial in (candidates - displacement, candidates + displacement):
            uv, _, _, _ = project(trial, intrinsic, transform, (96, 128))
            plane = centers[:, None].clone().expand(-1, len(trial), -1).clone()
            plane[..., 0] += (uv[..., 0] - 64) * (5 / 120)
            plane[..., 1] += (uv[..., 1] - 48) * (5 / 100)
            plane[..., 2] = 5
            values.append(texture_and_derivative(plane)[0])
        numerical = (values[1] - values[0]) / (2 * step)
        torch.testing.assert_close(jets[..., axis], numerical, rtol=1e-7, atol=1e-8)


def test_true_plane_has_rank_one_difference_and_normal():
    points = torch.tensor([[0.2, -0.3, 5.0], [-0.4, 0.1, 5.0]], dtype=torch.float64)
    _, result = statistics(points, compute_normal=True)
    assert result["usable"].all()
    assert result["normal_valid"].all()
    assert result["rank_defect"].max() < 1e-10
    assert (result["normal"][:, 2].abs() > 1 - 1e-8).all()


def test_wrong_candidates_can_violate_tangent_consistency():
    points = torch.tensor([[0.2, -0.3, 3.0], [-0.4, 0.1, 7.0]], dtype=torch.float64)
    _, result = statistics(points)
    assert result["usable"].all()
    assert result["rank_defect"].min() > 1e-4


def test_affine_texture_counterexample_is_not_a_match_certificate():
    points = torch.tensor([[0.2, -0.3, 3.0], [-0.4, 0.1, 7.0]], dtype=torch.float64)
    appearance, result = statistics(points, affine=True)
    assert result["usable"].all()
    assert result["rank_defect"].max() < 1e-10
    # 错误深度也满足低秩！中心纹理不一致，不能把低 defect 解释为表面。
    assert (appearance[1:] - appearance[0]).abs().max() > 0.1


def test_scale_covariance_and_metric_resolution():
    points = torch.tensor([[0.2, -0.3, 5.0], [-0.4, 0.1, 4.0]], dtype=torch.float64)
    _, _, _, jets, valid, parallax = plane_observations(points)
    original = metric_jet_statistics(jets[0], jets[1:], valid[1:], parallax, ruler_scale=0.5)
    factor = 3.7
    rescaled = metric_jet_statistics(jets[0] / factor, jets[1:] / factor, valid[1:], parallax / factor, ruler_scale=0.5 * factor)
    for key in ("rank_defect", "gram", "jet_energy", "ruler_parallax", "support", "usable"):
        torch.testing.assert_close(rescaled[key], original[key])
    original_width = parallax.norm(dim=-1).reciprocal()
    scaled_width = (parallax / factor).norm(dim=-1).reciprocal()
    torch.testing.assert_close(scaled_width, factor * original_width)
    intrinsic, transform, _ = cameras()
    _, scaled_transform, _ = cameras(factor)
    uv, _, _, jacobian = project(points, intrinsic, transform, (96, 128))
    scaled_uv, _, _, scaled_jacobian = project(points * factor, intrinsic, scaled_transform, (96, 128))
    torch.testing.assert_close(uv, scaled_uv)
    torch.testing.assert_close(jacobian, scaled_jacobian * factor)


def test_rig_rotation_does_not_change_compatibility():
    points = torch.tensor([[0.2, -0.3, 5.0], [-0.4, 0.1, 4.0]], dtype=torch.float64)
    _, _, _, jets, valid, parallax = plane_observations(points)
    rotation = torch.tensor([[0.0, 0, 1], [1.0, 0, 0], [0.0, 1, 0]], dtype=torch.float64)
    rotated = jets @ rotation.T
    original = metric_jet_statistics(jets[0], jets[1:], valid[1:], parallax)
    result = metric_jet_statistics(rotated[0], rotated[1:], valid[1:], parallax)
    torch.testing.assert_close(result["rank_defect"], original["rank_defect"])
    torch.testing.assert_close(result["jet_energy"], original["jet_energy"])


@pytest.mark.parametrize("reason", ["constant", "rotation", "missing", "one_source", "nonfinite"])
def test_no_evidence_never_creates_support(reason):
    torch.manual_seed(5)
    reference = torch.randn(4, 8, 3)
    source = torch.randn(3, 4, 8, 3)
    valid = torch.ones(3, 4, dtype=torch.bool)
    parallax = torch.ones(3, 4, 2)
    if reason == "constant":
        reference.zero_()
        source.zero_()
    elif reason == "rotation":
        parallax.zero_()
    elif reason == "missing":
        valid.zero_()
    elif reason == "one_source":
        valid[1:] = False
    else:
        source.fill_(float("nan"))
    result = metric_jet_statistics(reference, source, valid, parallax)
    assert not result["usable"].any()
    assert not result["normal_valid"].any()
    for value in result.values():
        assert torch.isfinite(value).all()


def test_rank_statistics_have_finite_gradients_and_no_default_eigensolve(monkeypatch):
    torch.manual_seed(17)
    reference = torch.randn(6, 8, 3, requires_grad=True)
    source = torch.randn(3, 6, 8, 3, requires_grad=True)
    monkeypatch.setattr(torch.linalg, "eigh", lambda *args, **kwargs: pytest.fail("默认不能调用 eigh"))
    monkeypatch.setattr(torch.linalg, "eigvalsh", lambda *args, **kwargs: pytest.fail("默认不能调用 eigvalsh"))
    result = metric_jet_statistics(reference, source, torch.ones(3, 6, dtype=torch.bool), torch.ones(3, 6, 2))
    (result["rank_defect"].mean() + 0.1 * result["jet_energy"].mean()).backward()
    for values in (reference, source):
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()
        assert values.grad.abs().sum() > 0


def test_constant_extra_view_cannot_pollute_other_compatible_views():
    points = torch.tensor([[0.2, -0.3, 5.0]], dtype=torch.float64)
    _, _, _, jets, valid, parallax = plane_observations(points)
    original = metric_jet_statistics(jets[0], jets[1:3], valid[1:3], parallax[:2])
    with_bad_view = metric_jet_statistics(jets[0], torch.cat((jets[1:3], torch.zeros_like(jets[3:]))), valid[1:], parallax)
    torch.testing.assert_close(with_bad_view["gram"], original["gram"])
    torch.testing.assert_close(with_bad_view["rank_defect"], original["rank_defect"])
