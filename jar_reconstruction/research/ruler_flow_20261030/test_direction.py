"""方向证据的 CPU 小测试：真实投影平面、链式导数、退化及可训练性。"""

import unittest

import torch

from research.ruler_flow_20261030.direction import (
    CHANNELS, _directional_samples, _sample_feature_and_gradient,
    directional_evidence, prepare_direction,
)


def fixture(views=3, scale=2., scene_scale=1.):
    height, width = 16, 32
    K = torch.tensor([[24., 0., 15.5], [0., 24., 7.5], [0., 0., 1.]])[None, None].repeat(1, views, 1, 1)
    T = torch.eye(4)[None, None].repeat(1, views, 1, 1)
    centers = torch.linspace(0., .6, views) if views > 1 else torch.zeros(1)
    T[0, :, 0, 3] = -centers / scene_scale
    yy, xx = torch.meshgrid(torch.arange(int(height * scale) // 4), torch.arange(int(width * scale) // 4), indexing="ij")
    u, v = (xx * 4 + .5) / scale - .5, (yy * 4 + .5) / scale - .5
    # 正面纹理平面 Z=4；不同相机描述子由同一世界表面坐标解析生成。
    features = []
    for center in centers:
        world_x, world_y = (u - 15.5) * 4 / 24 + center, (v - 7.5) * 4 / 24
        features.append(torch.stack((torch.ones_like(u), world_x * .3, world_y * .2, .1 + world_x * .1 + world_y * .2)))
    maps = torch.stack(features)[None].float()
    return {
        "pyramid": {4: maps}, "K": K, "T": T,
        "view_mask": torch.ones(1, views, dtype=torch.bool),
        "image_mask": torch.ones(1, views, height, width, dtype=torch.bool),
        "scale": torch.tensor([[scale, scale]]), "reference": 0,
        "image_size": (height, width),
    }


class DirectionTest(unittest.TestCase):
    def test_bilinear_exact_derivative(self):
        torch.manual_seed(7)
        maps = torch.randn(1, 4, 6, 8)
        cache = prepare_direction({"pyramid": {4: maps[:, None]}, "view_mask": torch.ones(1, 1, dtype=torch.bool)})
        uv = torch.tensor([[[1.31, 2.47], [4.17, 3.63], [-.2, 1.27], [7.3, 4.21]]])
        value, gx, gy = _sample_feature_and_gradient(maps, cache["dx"][:, 0], cache["dy"][:, 0], uv)
        for axis, gradient in enumerate((gx, gy)):
            change = torch.zeros_like(uv)
            change[..., axis] = 1e-3
            plus = _sample_feature_and_gradient(maps, cache["dx"][:, 0], cache["dy"][:, 0], uv + change)[0]
            minus = _sample_feature_and_gradient(maps, cache["dx"][:, 0], cache["dy"][:, 0], uv - change)[0]
            torch.testing.assert_close((plus - minus) / .002, gradient, atol=4e-4, rtol=4e-4)
        self.assertTrue(torch.isfinite(value).all())

    def test_cosine_chain_rule_real_projection(self):
        prepared = fixture()
        cache = prepare_direction(prepared)
        depth = torch.full((1, 4, 4, 8), 3.3)
        radius = torch.full_like(depth, .23)
        observed = _directional_samples(prepared, cache, depth, radius)
        epsilon = 2e-3
        plus = _directional_samples(prepared, cache, depth * (epsilon * radius).exp(), radius)
        minus = _directional_samples(prepared, cache, depth * (-epsilon * radius).exp(), radius)
        mask = observed["textured"] & plus["valid"] & minus["valid"]
        finite_difference = (plus["cosine"] - minus["cosine"]) / (2 * epsilon)
        self.assertGreater(mask.sum().item(), 40)
        torch.testing.assert_close(finite_difference[mask], observed["slope"][mask], atol=7e-5, rtol=.03)

    def test_plane_direction_sign(self):
        prepared = fixture()
        cache = prepare_direction(prepared)
        for z, sign in ((3., 1), (5., -1)):
            depth = torch.full((1, 4, 4, 8), z)
            evidence = directional_evidence(prepared, cache, depth, torch.full_like(depth, .25))
            reliable = evidence[:, 5] > 0
            self.assertGreater((evidence[:, 2][reliable] * sign).min().item(), 0)
            self.assertEqual(evidence.shape, (1, len(CHANNELS), 4, 4, 8))

    def test_renormalized_units_and_matching_resolution(self):
        depth = torch.full((1, 4, 4, 8), 3.3)
        radius = torch.full_like(depth, .2)
        original = fixture()
        scaled = fixture(scene_scale=11.)
        first = directional_evidence(original, prepare_direction(original), depth, radius)
        second = directional_evidence(scaled, prepare_direction(scaled), depth / 11., radius)
        # 恰在图像边界的合法性会受 FP32 投影舍入影响，比较严格内部点。
        torch.testing.assert_close(first[:, :, :, 1:3, 2:6], second[:, :, :, 1:3, 2:6], atol=1e-5, rtol=1e-4)
        # 两种 matching 分辨率采样同一解析平面，内部像素的方向应一致。
        other = fixture(scale=1.)
        third = directional_evidence(other, prepare_direction(other), depth, radius)
        torch.testing.assert_close(first[:, :6, :, 1:3, 2:6], third[:, :6, :, 1:3, 2:6], atol=2e-5, rtol=1e-4)

    def test_missing_views_and_single_camera(self):
        for views in (1, 2, 3, 4):
            prepared = fixture(views=views)
            prepared["view_mask"][:, 1:] = False
            prepared["image_mask"][:, 1:] = False
            depth = torch.full((1, 4, 4, 8), 3.3, requires_grad=True)
            evidence = directional_evidence(prepared, prepare_direction(prepared), depth, torch.full_like(depth, .2))
            self.assertEqual(torch.count_nonzero(evidence).item(), 0)
            evidence.sum().backward()
            self.assertTrue(torch.isfinite(depth.grad).all())

    def test_pure_rotation_and_constant_texture(self):
        depth = torch.full((1, 4, 4, 8), 3.3, requires_grad=True)
        prepared = fixture()
        prepared["T"][..., :3, 3] = 0
        angle = torch.tensor(.1)
        rotation = torch.tensor([[angle.cos(), 0, angle.sin()], [0, 1, 0], [-angle.sin(), 0, angle.cos()]])
        prepared["T"][0, 1, :3, :3] = rotation
        evidence = directional_evidence(prepared, prepare_direction(prepared), depth, torch.full_like(depth, .2))
        self.assertEqual(torch.count_nonzero(evidence).item(), 0)
        evidence.sum().backward()
        self.assertTrue(torch.isfinite(depth.grad).all())
        prepared = fixture()
        prepared["pyramid"][4] = torch.ones_like(prepared["pyramid"][4], requires_grad=True)
        evidence = directional_evidence(prepared, prepare_direction(prepared), depth.detach(), torch.full_like(depth, .2))
        for channel in (1, 2, 3, 5, 7, 9):
            self.assertEqual(torch.count_nonzero(evidence[:, channel]).item(), 0)
        evidence.sum().backward()
        self.assertTrue(torch.isfinite(prepared["pyramid"][4].grad).all())

    def test_feature_depth_and_weight_gradients(self):
        prepared = fixture()
        prepared["pyramid"][4].requires_grad_()
        depth = torch.full((1, 4, 4, 8), 3.3, requires_grad=True)
        weights = torch.full((1, 3, 4, 4, 8), .3, requires_grad=True)
        evidence = directional_evidence(prepared, prepare_direction(prepared), depth, torch.full_like(depth, .2), weights)
        evidence.square().mean().backward()
        for tensor in (prepared["pyramid"][4], depth, weights):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        compact = weights.detach()[:, 1:]
        again = directional_evidence(prepared, prepare_direction(prepared), depth.detach(), torch.full_like(depth, .2), compact)
        torch.testing.assert_close(evidence.detach(), again.detach())

    def test_batch_nonzero_reference_and_absent_reference(self):
        prepared = fixture(views=4)
        for key in ("K", "T", "view_mask", "image_mask", "scale"):
            prepared[key] = prepared[key].repeat((2,) + (1,) * (prepared[key].ndim - 1))
        prepared["pyramid"][4] = prepared["pyramid"][4].repeat(2, 1, 1, 1, 1)
        prepared["reference"] = 2
        prepared["view_mask"][1, 2] = False
        prepared["image_mask"][1, 2] = False
        depths = torch.full((2, 4, 4, 8), 3.3)
        evidence = directional_evidence(prepared, prepare_direction(prepared), depths, torch.full_like(depths, .2))
        self.assertGreater(evidence[0, 5].sum().item(), 0)
        self.assertEqual(evidence[1].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(evidence).all())

    def test_invalid_depth_and_zero_radius(self):
        prepared = fixture()
        for value, radius in ((float("nan"), .2), (-1., .2), (3., 0.), (3., float("inf"))):
            depths = torch.full((1, 4, 4, 8), value)
            evidence = directional_evidence(prepared, prepare_direction(prepared), depths, torch.full_like(depths, radius))
            self.assertTrue(torch.isfinite(evidence).all())
            self.assertEqual(evidence.abs().sum().item(), 0)

    def test_fp32_inside_autocast(self):
        prepared = fixture()
        depths = torch.full((1, 4, 4, 8), 3.3)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            evidence = directional_evidence(prepared, prepare_direction(prepared), depths, torch.full_like(depths, .2))
        self.assertEqual(evidence.dtype, torch.float32)
        self.assertTrue(torch.isfinite(evidence).all())

    def test_zero_reference_descriptor(self):
        prepared = fixture()
        prepared["pyramid"][4][:, 0] = 0
        depths = torch.full((1, 4, 4, 8), 3.3)
        evidence = directional_evidence(prepared, prepare_direction(prepared), depths, torch.full_like(depths, .2))
        self.assertEqual(evidence.abs().sum().item(), 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
