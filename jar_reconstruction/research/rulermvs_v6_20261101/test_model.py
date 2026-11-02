"""V6计算图、可拆卸性与训练目标测试；真实权重由VAL实验另验。"""

import copy
import unittest
from unittest.mock import patch

import torch

from research.early_geometry_20261030.test_encoding import TinyViT, inputs
from research.rulermvs_v5_20261031.test_bridge import tiny_pair
from research.rulermvs_v6_20261101.model import V6Model
from research.rulermvs_v6_20261101.objective import objective


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(61)
        _, self.v5 = tiny_pair()
        config = copy.deepcopy(self.v5.config)
        config.update(architecture="rulermvs_v6", handoff={"mode": "parent"})
        with patch("jar.pretraining.encoders._build_native_foundation", side_effect=TinyViT):
            self.v6 = V6Model(config, initialize=False).eval()
        self.v6.load_v5_state(self.v5.state_dict())
        self.domain = {"near": 1., "far": 10., "depth_scale": 1.}

    def test_disabled_and_coarse_are_unchanged(self):
        data = inputs()
        with torch.no_grad():
            old = self.v5(data, self.domain)
            self.v6.decoder.handoff_enabled = False
            disabled = self.v6(data, self.domain)
            self.assertTrue(torch.equal(old["depth"], disabled["depth"]))
            self.v6.decoder.handoff_enabled = True
            new = self.v6(data, self.domain)
            self.assertTrue(torch.equal(old["coarse"]["scores"], new["coarse"]["scores"]))
            self.assertTrue(torch.equal(old["fine"]["candidate_depth"], new["fine"]["candidate_depth"]))

    def test_missing_view_loss_gradient_and_reload(self):
        data = inputs()
        data["view_mask"][:, -1] = False
        data["rgb"][:, -1] = float("nan")
        data["matching_rgb"][:, -1] = float("nan")
        targets = {"depth": torch.full_like(data["image_mask"], 3., dtype=torch.float32),
                   "depth_valid": data["image_mask"].clone()}
        self.v6.requires_grad_(False)
        self.v6.decoder.requires_grad_(True)
        self.v6.decoder.train()
        prediction = self.v6(data, self.domain)
        terms = objective(prediction, data, targets, self.domain, {})
        terms["total"].backward()
        gate_grads = [p.grad for p in self.v6.decoder.handoff.parameters() if p.grad is not None]
        self.assertTrue(gate_grads)
        self.assertTrue(any(bool((g != 0).any()) for g in gate_grads))
        self.assertTrue(all(p.grad is None or bool(p.grad.isfinite().all()) for p in self.v6.parameters()))
        self.v6.eval()
        state = copy.deepcopy(self.v6.state_dict())
        self.v6.load_state_dict(state, strict=True)
        with torch.no_grad():
            before = self.v6(data, self.domain)["depth"]
            targets["depth"] *= 7
            after = self.v6(data, self.domain)["depth"]
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(bool(torch.isfinite(before).all()))

    def test_group_winner_preserves_original_point_in_candidate_set(self):
        self.v6.decoder.handoff_options["within_group"] = "winner"
        data = inputs()
        with torch.no_grad():
            old = self.v5(data, self.domain)
            new = self.v6(data, self.domain)
        nearest = (new["fine"]["group_depth"] - old["fine"]["depth"]).abs().amin(1)
        self.assertTrue(bool((nearest == 0).all()))

    def test_empty_supervision_has_zero_finite_loss_and_gradient(self):
        data = inputs()
        prediction = self.v6(data, self.domain)
        targets = {"depth": torch.full_like(data["image_mask"], float("nan"), dtype=torch.float32),
                   "depth_valid": torch.zeros_like(data["image_mask"])}
        terms = objective(prediction, data, targets, self.domain, {})
        self.assertEqual(terms["total"].item(), 0.)
        terms["total"].backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all())
                            for p in self.v6.parameters()))

    def test_explicit_winner_feature_preserves_default_outputs(self):
        data = inputs()
        with torch.no_grad():
            default = self.v6(data, self.domain)
            self.v6.decoder.handoff_options["upsample_feature"] = "winner"
            explicit = self.v6(data, self.domain)
        for key in ("depth", "confidence", "valid", "support", "unknown", "state"):
            self.assertTrue(torch.equal(default[key], explicit[key]), key)
        self.assertNotIn("conditional_probability", explicit["fine"])

    def test_group_feature_leaves_coarse_fine_and_support_unchanged(self):
        data = inputs()
        with torch.no_grad():
            original = self.v6(data, self.domain)
            self.v6.decoder.handoff_options["upsample_feature"] = "group_mean"
            pooled = self.v6(data, self.domain)
        for stage in ("coarse", "fine"):
            for key, value in original[stage].items():
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value, pooled[stage][key]), (stage, key))
        self.assertIn("conditional_probability", pooled["fine"])
        for key in ("valid", "support", "unknown", "state"):
            self.assertTrue(torch.equal(original[key], pooled[key]), key)

    def test_group_feature_is_bypassed_when_handoff_disabled(self):
        data = inputs()
        self.v6.decoder.handoff_options["upsample_feature"] = "group_mean"
        self.v6.decoder.handoff_enabled = False
        with torch.no_grad():
            expected = self.v5(data, self.domain)
            actual = self.v6(data, self.domain)
        for key in ("depth", "confidence", "valid", "support"):
            self.assertTrue(torch.equal(expected[key], actual[key]), key)
        self.assertNotIn("conditional_probability", actual["fine"])

    def test_group_feature_can_train_only_existing_mask_head(self):
        self.v6.decoder.handoff_options["upsample_feature"] = "group_mean"
        self.v6.requires_grad_(False)
        self.v6.decoder.upsample_mask.requires_grad_(True)
        prediction = self.v6(inputs(), self.domain)
        loss = (prediction["depth"] - 3.).square().mean()
        loss.backward()
        gradients = [p.grad for p in self.v6.decoder.upsample_mask.parameters()]
        self.assertTrue(all(g is not None and bool(g.isfinite().all()) for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)
        self.assertTrue(all(p.grad is None for name, p in self.v6.named_parameters()
                            if not name.startswith("decoder.upsample_mask.")))


if __name__ == "__main__":
    unittest.main()
