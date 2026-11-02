"""真实共享底座/梯度/恢复验证；CPU用已有轻量ViT替身。"""

import copy
import unittest
from unittest.mock import patch

import torch

from research.early_geometry_20261030.test_encoding import TinyViT, inputs
from research.rulermvs_v5_20261031.test_bridge import tiny_pair
from research.rulermvs_v6_20261101.objective import objective
from research.paper_ablation_20261102.handoff_train import build_ablation
from research.paper_ablation_20261102.handoff_sharedval import frozen_hash, tensor_leaves


class HandoffTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(35)
        _, self.v5 = tiny_pair()
        self.saved = {"architecture": "rulermvs_v5", "config": self.v5.config,
                      "model": self.v5.state_dict()}
        self.domain = {"near": 1.0, "far": 10.0, "depth_scale": 1.0}

    def build(self, saved, variant):
        with patch("jar.pretraining.encoders._build_native_foundation", side_effect=TinyViT):
            return build_ablation(saved, variant)

    def test_constant_and_adaptive_same_initial_outputs_and_shared_weights(self):
        constant, _ = self.build(self.saved, "constant")
        adaptive, _ = self.build(self.saved, "parent")
        for model in (constant, adaptive):
            model.eval()
            for name, tensor in self.saved["model"].items():
                self.assertTrue(torch.equal(model.state_dict()[name], tensor), name)
        data = inputs()
        with torch.no_grad():
            first = constant(data, self.domain)
            second = adaptive(data, self.domain)
        for name in ("depth", "valid"):
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in constant.decoder.handoff.gate.parameters()), 1)
        self.assertEqual(sum(p.numel() for p in adaptive.decoder.handoff.gate.parameters()), 129)

    def test_constant_gradient_and_strict_checkpoint_restore(self):
        model, config = self.build(self.saved, "constant")
        model.requires_grad_(False)
        model.decoder.requires_grad_(True)
        data = inputs()
        targets = {"depth": torch.full_like(data["image_mask"], 3.0, dtype=torch.float32),
                   "depth_valid": data["image_mask"].clone()}
        prediction = model(data, self.domain)
        objective(prediction, data, targets, self.domain, {}, grouped=True)["total"].backward()
        gradient = model.decoder.handoff.gate.logit.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient))
        self.assertGreater(abs(float(gradient)), 0.0)
        saved = {"architecture": model.architecture, "config": config,
                 "model": copy.deepcopy(model.state_dict())}
        restored, _ = self.build(saved, "constant")
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(data, self.domain)["depth"],
                                       restored(data, self.domain)["depth"], rtol=0, atol=0)

    def test_all_five_variants_execute_assigned_objective(self):
        data = inputs()
        targets = {"depth": torch.full_like(data["image_mask"], 3.0, dtype=torch.float32),
                   "depth_valid": data["image_mask"].clone()}
        for variant in ("v5", "marginal", "constant", "parent", "parent_oldpoint"):
            with self.subTest(variant=variant):
                model, _ = self.build(self.saved, variant)
                model.requires_grad_(False)
                model.decoder.requires_grad_(True)
                terms = objective(model(data, self.domain), data, targets, self.domain, {},
                                  grouped=variant not in ("v5", "parent_oldpoint"))
                terms["total"].backward()
                self.assertTrue(torch.isfinite(terms["total"]))
                self.assertTrue(all(p.grad is None or bool(p.grad.isfinite().all())
                                    for p in model.parameters()))

    def test_shared_encoding_is_exact_and_not_mutated(self):
        data = inputs()
        models = [self.build(self.saved, variant)[0].eval()
                  for variant in ("v5", "marginal", "constant", "parent", "parent_oldpoint")]
        self.assertEqual(len({frozen_hash(model) for model in models}), 1)
        with torch.no_grad():
            encoded = models[0].encode(data, self.domain)
            snapshot = {name: value.clone() for name, value in tensor_leaves(encoded).items()}
            for model in models:
                model.decoder.set_range(self.domain["near"], self.domain["far"])
                shared = model.decoder(encoded)
                complete = model(data, self.domain)
                for key in ("depth", "valid"):
                    torch.testing.assert_close(shared[key], complete[key], rtol=0, atol=0)
                for name, value in tensor_leaves(encoded).items():
                    torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
