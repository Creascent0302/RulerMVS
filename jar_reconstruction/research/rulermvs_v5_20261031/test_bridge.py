"""V5 CPU验证：真实ResNet、轻量ViT替身；完整ViT另由GPU冒烟验证。"""

import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn

from research.early_geometry_20261030.test_encoding import TinyViT, inputs
from research.v2_training_20261028.training import V2TrainingModel
from research.rulermvs_v5_20261031.bridge import CNNTokenBridge, aligned_subcells
from research.rulermvs_v5_20261031.model import V5Model


ROOT = Path(__file__).resolve().parents[2]


def tiny_pair():
    config = json.loads((ROOT / "research/v2_restart_20261029/configs/train_public.json").read_text())
    config["model"].update(channels=8, raw_channels=8, depth_bins=8, peak_count=2)
    config["bridge"] = {"bottleneck": 4, "hidden": 8}
    with patch("jar.pretraining.encoders._build_native_foundation", side_effect=TinyViT):
        original = V2TrainingModel(config, initialize=False).eval()
        updated = V5Model(copy.deepcopy(config), initialize=False).eval()
    updated.load_v2_state(original.state_dict())
    return original, updated


class BridgeGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(45)
        torch.set_num_threads(2)

    def test_linear_coordinates_and_subcell_order(self):
        for stride in (4, 8):
            yy, xx = torch.meshgrid(torch.arange(512 // stride, dtype=torch.float64),
                                    torch.arange(896 // stride, dtype=torch.float64), indexing="ij")
            feature = torch.stack((xx * stride, yy * stride))[None]
            packed, coverage = aligned_subcells(
                feature, torch.ones(1, 512, 896, dtype=torch.bool), (256, 448), (16, 28), stride)
            actual = packed.reshape(1, 16, 28, 2, 2, 2)
            for py in range(2):
                for px in range(2):
                    yy, xx = torch.meshgrid(torch.arange(15, dtype=torch.float64),
                                            torch.arange(27, dtype=torch.float64), indexing="ij")
                    expected = torch.stack((32 * xx + 7.5 + 16 * px,
                                            32 * yy + 7.5 + 16 * py), -1)
                    torch.testing.assert_close(actual[0, :15, :27, py, px], expected, rtol=0, atol=1e-10)
            self.assertTrue(bool((coverage > 0).all()))

    def test_odd_matching_shape_constant_and_masked_nan(self):
        mask = torch.ones(1, 65, 99, dtype=torch.bool)
        mask[:, :, :25] = False
        feature = torch.full((1, 3, 17, 25), 7.)
        feature[:, :, :, :7] = float("nan")
        packed, coverage = aligned_subcells(feature, mask, (32, 48), (2, 3), 4)
        self.assertTrue(bool(torch.isfinite(packed).all()))
        nonzero = packed != 0
        torch.testing.assert_close(packed[nonzero], torch.full_like(packed[nonzero], 7.))
        self.assertTrue(bool(((coverage >= 0) & (coverage <= 1)).all()))

    def test_all_invalid_does_not_create_token_from_bias(self):
        bridge = CNNTokenBridge({"bottleneck": 4, "hidden": 8})
        nn.init.normal_(bridge.output.weight)
        nn.init.ones_(bridge.fusion[0].bias)
        tokens = torch.randn(2, 6, 768)
        features = {"layer1": torch.full((2, 64, 16, 24), float("nan")),
                    "layer2": torch.full((2, 128, 8, 12), float("nan"))}
        output = bridge(tokens, features, torch.zeros(2, 64, 96, dtype=torch.bool),
                        torch.zeros(2, 32, 48, dtype=torch.bool), (2, 3))
        self.assertTrue(torch.equal(output, tokens))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is None or bool(p.grad.isfinite().all()) for p in bridge.parameters()))

    def test_partial_nan_mask_is_applied_before_projection(self):
        bridge = CNNTokenBridge({"layers": ["layer1"], "bottleneck": 4, "hidden": 8})
        nn.init.normal_(bridge.output.weight, std=.01)
        feature = torch.randn(1, 64, 16, 24)
        feature[:, :, :, :4] = float("nan")
        mask = torch.ones(1, 64, 96, dtype=torch.bool)
        mask[:, :, :16] = False
        result = bridge(torch.randn(1, 6, 768), {"layer1": feature}, mask,
                        torch.ones(1, 32, 48, dtype=torch.bool), (2, 3))
        self.assertTrue(bool(result.isfinite().all()))
        result.square().mean().backward()
        self.assertTrue(all(p.grad is None or bool(p.grad.isfinite().all()) for p in bridge.parameters()))


class V5ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(14)
        self.domain = {"depth_scale": .7, "near": 1., "far": 10.}

    def test_zero_bridge_and_disabled_are_original_v2(self):
        original, updated = tiny_pair()
        data = inputs()
        with torch.no_grad():
            before = original(data, self.domain)
            after = updated(data, self.domain)
            for key in ("depth", "valid", "confidence"):
                self.assertTrue(torch.equal(before[key], after[key]), key)
            for stage in ("coarse", "fine"):
                self.assertTrue(torch.equal(before[stage]["scores"], after[stage]["scores"]))
            nn.init.normal_(updated.bridge.output.weight, std=.1)
            updated.bridge.enabled = False
            disabled = updated(data, self.domain)
            self.assertTrue(torch.equal(disabled["depth"], before["depth"]))
        self.assertEqual(set(original.state_dict()), set(updated.v2_state_dict()))

    def test_gradient_crosses_frozen_vit_and_two_steps_train_earlier_bridge(self):
        _, model = tiny_pair()
        model.requires_grad_(False)
        model.bridge.requires_grad_(True)
        model.train()
        optimizer = torch.optim.Adam(model.bridge.parameters(), lr=.001)
        data = inputs()
        target = torch.randn(1, 768, 2, 3)
        for step in range(2):
            encoded = model.encode(data, self.domain)
            loss = (encoded["raw_maps"][-1] * target).mean()
            loss.backward()
            self.assertGreater(float(model.bridge.output.weight.grad.abs().max()), 0)
            if step == 1:
                self.assertGreater(float(model.bridge.projections["layer1"].weight.grad.abs().max()), 0)
            self.assertTrue(all(p.grad is None for p in model.foundation.parameters()))
            self.assertTrue(all(p.grad is None for p in model.matching.parameters()))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.assertGreaterEqual(model.foundation.model.blocks[0].calls, 4)
        self.assertFalse(model.foundation.model.patch_embed._forward_hooks)
        self.assertFalse(model.matching.backbone.layer1._forward_hooks)

    def test_optional_gradient_to_shared_backbone_and_checkpoint_reload(self):
        original, model = tiny_pair()
        model.requires_grad_(False)
        model.bridge.requires_grad_(True)
        model.matching.backbone.layer1.requires_grad_(True)
        model.bridge.detach_backbone = False
        nn.init.normal_(model.bridge.output.weight, std=.01)
        model.train()
        raw = model.encode(inputs(), self.domain)["raw_maps"][-1]
        (raw * torch.randn_like(raw)).mean().backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().max() > 0)
                            for p in model.matching.backbone.layer1.parameters()))
        self.assertTrue(all(p.grad is None or bool(p.grad.isfinite().all()) for p in model.parameters()))
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        state = torch.load(buffer, weights_only=True)
        model.load_state_dict(state, strict=True)
        with self.assertRaises(ValueError):
            model.load_v2_state(state)
        original.load_state_dict(model.v2_state_dict(), strict=True)

    def test_multireference_shared_matching_and_missing_view(self):
        _, model = tiny_pair()
        data = inputs()
        data["view_mask"][:, 1] = False
        data["rgb"][:, 1] = float("nan")
        data["matching_rgb"][:, 1] = float("nan")
        nn.init.normal_(model.bridge.output.weight, std=.01)
        calls = []
        handle = model.matching.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            result = model.forward_references(data, self.domain)
            self.assertEqual(len(calls), 1)
            handle.remove()
            for ref in range(3):
                separate = model(data, self.domain, ref)
                self.assertTrue(torch.equal(result[ref]["depth"], separate["depth"]))
                self.assertTrue(bool(result[ref]["depth"].isfinite().all()))
            self.assertFalse(bool(result[1]["valid"].any()))
            data["targets"] = {"depth": torch.full((1, 3, 32, 48), 999.)}
            repeated = model(data, self.domain, 0)
            self.assertTrue(torch.equal(result[0]["depth"], repeated["depth"]))


if __name__ == "__main__":
    unittest.main()
