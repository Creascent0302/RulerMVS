"""V7独立支路的恒等起点、掩码、两步梯度与相机数测试。"""

import copy
import unittest
from unittest.mock import patch

import torch

from research.early_geometry_20261030.test_encoding import TinyViT, inputs
from research.rulermvs_v5_20261031.test_bridge import tiny_pair
from research.rulermvs_v6_20261101.model import V6Model
from research.rulermvs_v6_20261101.objective import objective
from research.overnight_v6_v7_mvs_20261103.v7.model import V7Model
from research.overnight_v6_v7_mvs_20261103.v7.refusion import CandidateRefusion


class RefusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(73)

    def test_all_invalid_and_nondivisible_candidates(self):
        block = CandidateRefusion(8, 8, {"hidden": 8, "heads": 2, "bands": 4})
        torch.nn.init.normal_(block.readout.weight, std=.02)
        value = torch.randn(1, 8, 10, 5, 7, requires_grad=True)
        raw = [torch.full((1, 8, 5, 7), float("nan")) for _ in range(4)]
        valid = torch.zeros(1, 10, 5, 7, dtype=torch.bool)
        quality = torch.zeros(1, 1, 10, 5, 7)
        output = block(value, raw, valid, quality, quality)
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        output.sum().backward()
        self.assertTrue(all(p.grad is None or bool(p.grad.isfinite().all()) for p in block.parameters()))

    def test_first_and_second_step_train_the_whole_branch(self):
        block = CandidateRefusion(8, 8, {"hidden": 8, "heads": 2, "bands": 4})
        value = torch.randn(1, 8, 8, 4, 6)
        raw = [torch.randn(1, 8, 4, 6) for _ in range(4)]
        valid = torch.ones(1, 8, 4, 6, dtype=torch.bool)
        quality = torch.rand(1, 1, 8, 4, 6)
        optimizer = torch.optim.SGD(block.parameters(), lr=.05)
        target = torch.randn(1, 8, 4, 6)
        output = block(value, raw, valid, quality, quality)
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        (output - target).square().mean().backward()
        self.assertGreater(float(block.readout.weight.grad.abs().sum()), 0.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        output = block(value, raw, valid, quality, quality)
        (output - target).square().mean().backward()
        for name, parameter in block.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(parameter.grad.isfinite().all()), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0., name)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(74)
        _, v5 = tiny_pair()
        config = copy.deepcopy(v5.config)
        config.update(architecture="rulermvs_v6", handoff={"mode": "parent"})
        with patch("jar.pretraining.encoders._build_native_foundation", side_effect=TinyViT):
            self.v6 = V6Model(config, initialize=False).eval()
            config = copy.deepcopy(config)
            config.update(architecture="rulermvs_v7_refusion", refusion={"hidden": 8, "heads": 2, "bands": 4})
            self.v7 = V7Model(config, initialize=False).eval()
        self.v7.load_v6_state(self.v6.state_dict())
        self.domain = {"near": 1., "far": 10., "depth_scale": 1.}

    def test_exact_identity_and_disabled(self):
        data = inputs()
        with torch.no_grad():
            original = self.v6(data, self.domain)
            updated = self.v7(data, self.domain)
            for key in ("depth", "valid", "confidence"):
                self.assertTrue(torch.equal(original[key], updated[key]), key)
            torch.nn.init.normal_(self.v7.decoder.refusion.readout.weight, std=.1)
            self.v7.decoder.refusion.enabled = False
            disabled = self.v7(data, self.domain)
            self.assertTrue(torch.equal(original["depth"], disabled["depth"]))

    def test_missing_view_real_objective_and_reload(self):
        data = inputs()
        data["view_mask"][:, -1] = False
        data["rgb"][:, -1] = float("nan")
        data["matching_rgb"][:, -1] = float("nan")
        targets = {"depth": torch.full_like(data["image_mask"], 3., dtype=torch.float32),
                   "depth_valid": data["image_mask"].clone()}
        self.v7.requires_grad_(False)
        self.v7.decoder.requires_grad_(True)
        self.v7.decoder.train()
        prediction = self.v7(data, self.domain)
        terms = objective(prediction, data, targets, self.domain, {})
        terms["total"].backward()
        self.assertGreater(float(self.v7.decoder.refusion.readout.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None or bool(p.grad.isfinite().all()) for p in self.v7.parameters()))
        self.v7.eval()
        state = copy.deepcopy(self.v7.state_dict())
        self.v7.load_state_dict(state, strict=True)
        with torch.no_grad():
            expected = self.v7(data, self.domain)["depth"]
            targets["depth"] *= 10
            actual = self.v7(data, self.domain)["depth"]
        self.assertTrue(torch.equal(expected, actual))

    def test_variable_camera_counts(self):
        original = inputs()
        for count in (2, 3, 4):
            data = {key: value[:, torch.arange(count) % 3].clone() for key, value in original.items()}
            if count == 4:
                data["T_cam_from_rig"][:, 3, 0, 3] -= .9
            with torch.no_grad():
                output = self.v7(data, self.domain)
            self.assertTrue(bool(torch.isfinite(output["depth"]).all()))
            self.assertEqual(output["depth"].shape, (1, 1, 32, 48))


if __name__ == "__main__":
    unittest.main()
