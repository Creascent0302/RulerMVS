"""候选空间融合的掩码、邻接和标定测试。"""

import torch

from jar.models.ruler_spatial import CandidatePlaneMessage, CandidateSpatialConv, candidate_plane_grid


def test_spatial_conv_observes_neighbor_pixels():
    model = CandidateSpatialConv(1)
    with torch.no_grad():
        model.depthwise.weight.zero_()
        model.depthwise.weight[0, 0, 1, 1, 2] = 1
        model.depthwise.bias.zero_()
        model.pointwise.weight.fill_(1)
        model.pointwise.bias.zero_()
    value = torch.zeros(1, 1, 2, 3, 4)
    value[0, 0, 0, 1, 2] = 2
    output = model(value, torch.ones_like(value[:, 0], dtype=torch.bool))
    assert output[0, 0, 0, 1, 1] > 1
    assert output[0, 0, 0, 1, -1] == 0
    assert output[0, 0, 1].eq(0).all()


def test_spatial_conv_missing_values_do_not_leak():
    torch.manual_seed(4)
    model = CandidateSpatialConv(4)
    value = torch.randn(2, 4, 3, 5, 7)
    valid = torch.rand(2, 3, 5, 7) > 0.3
    clean = torch.where(valid[:, None], value, 0)
    corrupt = torch.where(valid[:, None], value, torch.nan)
    assert torch.equal(model(clean, valid), model(corrupt, valid))
    assert model(clean, valid).masked_select(~valid[:, None]).eq(0).all()


def test_spatial_conv_all_invalid_and_gradients():
    model = CandidateSpatialConv(4)
    value = torch.randn(2, 4, 3, 5, 7, requires_grad=True)
    valid = torch.zeros(2, 3, 5, 7, dtype=torch.bool)
    output = model(value, valid)
    assert output.eq(0).all()
    output.sum().backward()
    assert torch.isfinite(value.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    model.zero_grad()
    model(value, ~valid).square().mean().backward()
    assert all(parameter.grad.abs().sum() > 0 for parameter in model.parameters())


def test_spatial_conv_parameter_budget():
    channels = 32
    model = CandidateSpatialConv(channels)
    previous = 3 * channels + channels + channels * channels + channels
    assert sum(value.numel() for value in model.parameters()) == previous + 24 * channels


def test_spatial_conv_initially_preserves_old_depth_local():
    torch.manual_seed(9)
    old = torch.nn.Sequential(
        torch.nn.Conv3d(4, 4, (3, 1, 1), padding=(1, 0, 0), groups=4),
        torch.nn.GELU(), torch.nn.Conv3d(4, 4, 1),
    )
    new = CandidateSpatialConv(4).initialize_from_depth(old)
    valid = torch.rand(2, 5, 3, 7) > 0.3
    value = torch.randn(2, 4, 5, 3, 7, requires_grad=True)
    clean = torch.where(valid[:, None], value, 0)
    reference = torch.where(valid[:, None], old(clean), 0)
    actual = new(value, valid)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    expected_gradient = torch.autograd.grad(reference.square().sum(), value, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual.square().sum(), value)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def _geometry(batch=2, height=5, width=7, stride=2):
    depth = torch.tensor([2., 2.5, 4., 8., 20.])
    intrinsic = torch.tensor([[8., 0., 6.], [0., 8., 2.], [0., 0., 1.]])
    transform = torch.eye(4)
    transform[:3, :3] = torch.tensor([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    center = torch.tensor([0., 0., 2.])
    transform[:3, 3] = -transform[:3, :3] @ center
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    pixels = torch.stack((x * stride, y * stride, torch.ones_like(x)), 0).float().flatten(1)
    rays = (transform[:3, :3].T @ torch.linalg.solve(intrinsic, pixels)).reshape(3, height, width)
    points = center[:, None, None, None] + rays[:, None] * depth[None, :, None, None]
    return depth, points[None].expand(batch, -1, -1, -1, -1), intrinsic[None].expand(batch, -1, -1), transform[None].expand(batch, -1, -1), stride


def test_plane_grid_uses_nonuniform_depth_interpolation_exactly():
    geometry = _geometry()
    depth, points = geometry[:2]
    grid, legal, projected_depth = candidate_plane_grid(*geometry)
    batch, _, count, height, width = points.shape
    source = depth[None, None, :, None, None].expand(batch, 1, -1, height, width)
    sampled = torch.nn.functional.grid_sample(source, grid.reshape(batch, -1, height, width, 3),
                                             align_corners=False)
    sampled = sampled.reshape_as(projected_depth)
    torch.testing.assert_close(sampled[legal], projected_depth[legal], atol=2e-5, rtol=1e-5)
    # 前向平面恒定 Z；水平面沿竖直邻居真正跨越 depth 候选。
    torch.testing.assert_close(projected_depth[:, 0], depth[None, None, :, None, None].expand_as(projected_depth[:, 0]))
    assert (projected_depth[:, 1] - projected_depth[:, 0]).abs()[legal[:, 1]].max() > 1


def test_plane_grid_masks_horizon_behind_outside_and_no_wrapping():
    grid, legal, depth = candidate_plane_grid(*_geometry())
    assert torch.isfinite(grid).all()
    assert not legal[:, :, 0, :, 0].any()  # 上邻居不越过图像边界。
    assert not legal[:, :, 1, :, -1].any()
    assert not legal[:, :, 2, :, :, 0].any()
    assert not legal[:, :, 3, :, :, -1].any()
    # 图像 y=1 对应水平地平线；相邻射线落到此处不能求可靠平面交点。
    assert not legal[:, 1, 0, :, 2].any()
    assert not legal[:, 1, 1, :, 0].any()
    assert (depth[legal] >= 2).all() and (depth[legal] <= 20).all()


def test_plane_grid_flat_control_keeps_neighbor_count_and_shapes():
    geometry = _geometry()
    metric, metric_valid, _ = candidate_plane_grid(*geometry)
    flat, flat_valid, depth = candidate_plane_grid(*geometry, use_metric=False)
    assert flat.shape == metric.shape
    assert flat_valid.shape == metric_valid.shape
    torch.testing.assert_close(depth[:, 0], depth[:, 1])
    assert flat_valid.sum() > metric_valid.sum()


def test_plane_message_matches_old_initial_output_and_gradient():
    torch.manual_seed(2)
    old = torch.nn.Sequential(
        torch.nn.Conv3d(4, 4, (3, 1, 1), padding=(1, 0, 0), groups=4),
        torch.nn.GELU(), torch.nn.Conv3d(4, 4, 1),
    )
    new = CandidatePlaneMessage(4).initialize_from_depth(old)
    geometry = _geometry()
    valid = torch.rand(2, 5, 5, 7) > 0.2
    value = torch.randn(2, 4, 5, 5, 7, requires_grad=True)
    reference = torch.where(valid[:, None], old(torch.where(valid[:, None], value, 0)), 0)
    output = new(value, valid, *geometry)
    torch.testing.assert_close(output, reference, atol=0, rtol=0)
    expected_gradient = torch.autograd.grad(reference.square().sum(), value, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(output.square().sum(), value)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=0, rtol=0)


def test_plane_message_all_invalid_and_real_gradients():
    model = CandidatePlaneMessage(4)
    geometry = _geometry()
    value = torch.randn(2, 4, 5, 5, 7, requires_grad=True)
    valid = torch.zeros(2, 5, 5, 7, dtype=torch.bool)
    output = model(value, valid, *geometry)
    assert torch.isfinite(output).all() and output.eq(0).all()
    output.sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    model.zero_grad()
    with torch.no_grad():
        model.neighbor_weight.fill_(0.1)
    model(value, ~valid, *geometry).square().mean().backward()
    assert all(torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
               for parameter in model.parameters())


def test_plane_message_rejects_missing_corrupted_candidates():
    model = CandidatePlaneMessage(4)
    with torch.no_grad():
        model.neighbor_weight.fill_(1)
    valid = torch.rand(2, 5, 5, 7) > 0.3
    value = torch.randn(2, 4, 5, 5, 7)
    geometry = _geometry()
    clean = model(torch.where(valid[:, None], value, 0), valid, *geometry)
    corrupt = model(torch.where(valid[:, None], value, torch.nan), valid, *geometry)
    assert torch.isfinite(corrupt).all()
    torch.testing.assert_close(clean, corrupt, atol=0, rtol=0)


def test_plane_message_parameter_budget_matches_small_control():
    metric = CandidatePlaneMessage(32)
    flat = CandidatePlaneMessage(32, use_metric=False)
    control = CandidateSpatialConv(32)
    assert sum(parameter.numel() for parameter in metric.parameters()) == 1540
    assert sum(parameter.numel() for parameter in flat.parameters()) == 1540
    assert sum(parameter.numel() for parameter in control.parameters()) == 1952


def test_plane_neighbors_preserve_resize_and_rig_translation():
    depth, points, intrinsic, transform, stride = _geometry()
    original, valid, _ = candidate_plane_grid(depth, points, intrinsic, transform, stride)
    resized_intrinsic = intrinsic.clone()
    resized_intrinsic[:, :2] *= 2
    resized, resized_valid, _ = candidate_plane_grid(depth, points, resized_intrinsic, transform, stride * 2)
    torch.testing.assert_close(original, resized, atol=0, rtol=0)
    assert torch.equal(valid, resized_valid)
    translation = torch.tensor([3., -2., 4.])
    moved_points = points + translation[None, :, None, None, None]
    moved_transform = transform.clone()
    moved_transform[:, :3, 3] -= (transform[:, :3, :3] @ translation[:, None])[..., 0]
    moved, moved_valid, _ = candidate_plane_grid(depth, moved_points, intrinsic, moved_transform, stride)
    torch.testing.assert_close(original, moved, atol=0, rtol=0)
    assert torch.equal(valid, moved_valid)


def test_tilted_camera_front_plane_does_not_drop_depth_endpoints():
    depth, _, intrinsic, transform, stride = _geometry()
    angle = torch.tensor(0.2)
    rotation = torch.tensor([[1., 0., 0.], [0., angle.cos(), -angle.sin()],
                             [0., angle.sin(), angle.cos()]])
    transform = transform.clone()
    transform[:, :3, :3] = rotation @ transform[:, :3, :3]
    transform[:, :3, 3] = 0
    y, x = torch.meshgrid(torch.arange(5), torch.arange(7), indexing="ij")
    pixels = torch.stack((x * stride, y * stride, torch.ones_like(x)), 0).float().flatten(1)
    rays = transform[:, :3, :3].transpose(1, 2) @ torch.linalg.solve(intrinsic, pixels[None].expand(2, -1, -1))
    points = rays.reshape(2, 3, 5, 7)[:, :, None] * depth[None, None, :, None, None]
    _, legal, mapped_depth = candidate_plane_grid(depth, points, intrinsic, transform, stride)
    _, flat_legal, _ = candidate_plane_grid(depth, points, intrinsic, transform, stride, use_metric=False)
    assert torch.equal(legal[:, 0], flat_legal[:, 0])
    torch.testing.assert_close(mapped_depth[:, 0], depth[None, None, :, None, None].expand_as(mapped_depth[:, 0]),
                               atol=0, rtol=0)
