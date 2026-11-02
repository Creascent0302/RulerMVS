"""早期条件的CPU几何/恒等/反向测试；小ViT替身避免加载大权重。"""

import copy
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from jar.geometry import pixel_grid
from jar.pretraining.encoders import MatchingPyramid, TrainableFoundation
from jar.pretraining.model import PretrainingModel
from research.early_geometry_20261030.encoding import (
    CameraAdapter, camera_descriptor, foundation_camera, install_encoding, matching_camera,
)


def inputs(dtype=torch.float32):
    K = torch.tensor([[30., 2., 23.5], [0., 28., 15.5], [0., 0., 1.]], dtype=dtype)
    K = K[None, None].repeat(1, 3, 1, 1)
    T = torch.eye(4, dtype=dtype)[None, None].repeat(1, 3, 1, 1)
    T[0, 1, :3, 3] = torch.tensor([-.3, -.1, -.2], dtype=dtype)
    T[0, 2, :3, 3] = torch.tensor([-.6, 0., -.4], dtype=dtype)
    return {
        "rgb": torch.rand(1, 3, 3, 32, 48), "matching_rgb": torch.rand(1, 3, 3, 64, 96),
        "K": K, "T_cam_from_rig": T, "view_mask": torch.ones(1, 3, dtype=torch.bool),
        "image_mask": torch.ones(1, 3, 32, 48, dtype=torch.bool),
        "matching_image_mask": torch.ones(1, 3, 64, 96, dtype=torch.bool),
    }


class TinyPatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 768, 14, stride=14)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 16), nn.GELU(), nn.Linear(16, 768))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + self.net(x)


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = TinyPatch()
        self.blocks = nn.ModuleList(TinyBlock() for _ in range(12))
        self.norm = nn.LayerNorm(768)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        self.position = nn.Parameter(torch.randn(1, 7, 768) * .01)
        self.last_token_count = None

    def prepare_tokens_with_masks(self, x):
        tokens = self.patch_embed(x)
        tokens = torch.cat((self.cls_token.expand(len(x), -1, -1), tokens), 1)
        self.last_token_count = tokens.shape[1]
        return tokens + self.position


def tiny_model():
    foundation = TrainableFoundation.__new__(TrainableFoundation)
    nn.Module.__init__(foundation)
    foundation.model = TinyViT()
    foundation.indices = (2, 5, 8, 11)
    foundation.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
    foundation.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])
    model = PretrainingModel.__new__(PretrainingModel)
    nn.Module.__init__(model)
    model.foundation = foundation
    model.matching = MatchingPyramid("", channels=8, initialize=False)
    model.config = {}
    return model.eval()


class EncodingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.domain = {"depth_scale": .7}
        self.grid = pixel_grid(3, 4, dtype=torch.float64, stride=4) + 3.5

    def descriptor(self, item):
        return camera_descriptor(item, self.domain, self.grid, (4., 4.))

    def test_rig_gauge_and_camera_permutation(self):
        item = inputs(torch.float64)
        expected = self.descriptor(item)
        change = torch.eye(4, dtype=torch.float64)
        c, s = math.cos(.4), math.sin(.4)
        change[:3, :3] = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], dtype=torch.float64)
        change[:3, 3] = torch.tensor([13., -4., 2.])
        moved = dict(item, T_cam_from_rig=item["T_cam_from_rig"] @ torch.linalg.inv(change))
        torch.testing.assert_close(self.descriptor(moved), expected, rtol=1e-10, atol=1e-10)
        order = [2, 0, 1]
        reordered = {key: value[:, order] for key, value in item.items()}
        torch.testing.assert_close(self.descriptor(reordered), expected[:, order], rtol=0, atol=1e-15)

    def test_missing_calibration_is_excluded_before_inverse(self):
        item = inputs(torch.float64)
        item["view_mask"][0, 2] = False
        expected = self.descriptor(item)
        item["K"][0, 2] = 0
        item["T_cam_from_rig"][0, 2] = float("nan")
        actual = self.descriptor(item)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(int(torch.count_nonzero(actual[:, 2])), 0)
        subset = {key: value[:, :2] for key, value in item.items()}
        torch.testing.assert_close(self.descriptor(subset), actual[:, :2], rtol=0, atol=0)

    def test_scene_unit_scaling_and_pure_rotation(self):
        item = inputs(torch.float64)
        expected = self.descriptor(item)
        larger = copy.deepcopy(item)
        larger["T_cam_from_rig"][..., :3, 3] *= 17
        actual = camera_descriptor(larger, {"depth_scale": .7 * 17}, self.grid, (4., 4.))
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        item["T_cam_from_rig"][..., :3, 3] = 0
        item["T_cam_from_rig"][0, 1, :3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.assertEqual(int(torch.count_nonzero(self.descriptor(item)[:, :, -1])), 0)

    def test_resize_crop_keeps_same_physical_ray_and_footprint(self):
        item = inputs(torch.float64)
        expected = self.descriptor(item)
        resized = copy.deepcopy(item)
        scale = torch.tensor([1.5, 2.], dtype=torch.float64)
        crop = torch.tensor([3., 5.], dtype=torch.float64)
        transform = torch.eye(3, dtype=torch.float64)
        transform[0, 0], transform[1, 1] = scale
        transform[:2, 2] = .5 * scale - .5 - crop
        resized["K"] = transform @ item["K"]
        grid = (self.grid + .5) * scale - .5 - crop
        actual = camera_descriptor(resized, self.domain, grid, 4 * scale)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_cnn_and_vit_centers_are_distinct_and_correct(self):
        item = inputs()
        cnn = matching_camera(item, self.domain, (16, 24))
        vit = foundation_camera(item, self.domain, (28, 42))
        for descriptor, uv in ((cnn, (-.25, -.25)), (vit, (7.5, 7.5))):
            q = torch.linalg.solve(item["K"][0, 0], torch.tensor([*uv, 1.]))
            torch.testing.assert_close(descriptor[0, 0, :3, 0, 0], F.normalize(q, dim=0))

    def test_channel_modulation_survives_l2_and_zero_features_stay_zero(self):
        adapter = CameraAdapter(8)
        feature = torch.randn(2, 8, 3, 4)
        condition = torch.randn(2, 7, 3, 4)
        torch.testing.assert_close(adapter(feature, condition), feature, rtol=0, atol=0)
        torch.testing.assert_close(F.normalize(feature, dim=1), F.normalize(2 * feature, dim=1), rtol=0, atol=0)
        with torch.no_grad():
            adapter.net[-1].weight.normal_(std=.2)
        result = adapter(feature, condition)
        self.assertGreater(float((F.normalize(result, dim=1) - F.normalize(feature, dim=1)).abs().max()), 1e-4)
        self.assertEqual(int(torch.count_nonzero(adapter(torch.zeros_like(feature), condition))), 0)

    def test_zero_initialization_preserves_encoding_and_rng(self):
        item, model = inputs(), tiny_model()
        model.requires_grad_(False)
        with torch.no_grad():
            expected = model.encode(item, self.domain)
        old = {name: (id(value), value.clone()) for name, value in model.named_parameters()}
        rng = torch.random.get_rng_state().clone()
        install_encoding(model, {"matching": True, "foundation": True})
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        with torch.no_grad():
            actual = model.encode(item, self.domain)
        for name, parameter in model.named_parameters():
            if name in old:
                self.assertEqual(id(parameter), old[name][0])
                torch.testing.assert_close(parameter, old[name][1], rtol=0, atol=0)
            else:
                self.assertIn("camera_adapter", name)
        for stride in expected["pyramid"]:
            torch.testing.assert_close(actual["pyramid"][stride], expected["pyramid"][stride], rtol=0, atol=0)
        for left, right in zip(actual["raw_maps"], expected["raw_maps"]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(model.foundation.model.last_token_count, 7)
        self.assertEqual(len(model.foundation.model.patch_embed._forward_hooks), 0)

    def test_frozen_backbones_propagate_adapter_gradient_in_eval(self):
        item, model = inputs(), tiny_model().requires_grad_(False)
        install_encoding(model, {"matching": True, "foundation": True})
        model.eval()
        # 输入中故意不提供depth/targets；推理接口只持有RGB、标定与掩码。
        encoded = model.encode(item, self.domain)
        weights = torch.arange(768, dtype=torch.float32)[None, :, None, None] / 768
        loss = (encoded["raw_maps"][-1] * weights).mean()
        loss = loss + (encoded["pyramid"][4] * torch.arange(8)[None, None, :, None, None]).mean()
        loss.backward()
        for branch in (model.foundation, model.matching):
            gradient = branch.camera_adapter.net[-1].weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().max()), 0.)
        for name, parameter in model.named_parameters():
            if "camera_adapter" not in name:
                self.assertIsNone(parameter.grad)
        self.assertTrue(all(block.calls >= 2 for block in model.foundation.model.blocks))
        self.assertEqual(len(model.foundation.model.patch_embed._forward_hooks), 0)

    def test_constant_and_geometry_have_equal_parameters(self):
        source = tiny_model()
        geometry, constant = copy.deepcopy(source), copy.deepcopy(source)
        install_encoding(geometry, {"matching": True, "camera_mode": "geometry"})
        install_encoding(constant, {"matching": True, "camera_mode": "constant"})
        self.assertEqual(geometry.state_dict().keys(), constant.state_dict().keys())
        for key in geometry.state_dict():
            torch.testing.assert_close(geometry.state_dict()[key], constant.state_dict()[key], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
