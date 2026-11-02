"""实际细层读出监督：格点、米制掩码及组内定位梯度。"""

import copy
import unittest
from unittest.mock import patch

import torch

from research.early_geometry_20261030.test_encoding import TinyViT, inputs
from research.rulermvs_v5_20261031.test_bridge import tiny_pair
from .handoff import HierarchicalHandoff
from .model import V6Model
from .objective import fine_readout_loss, objective


def example(estimate=3., truth=2., scale=1.):
    depth = torch.full((1, 1, 2, 3), estimate, requires_grad=True)
    prediction = {"unit_scale": scale, "fine": {
        "depth": depth, "ref_valid": torch.ones_like(depth, dtype=torch.bool)}}
    mask = torch.ones(1, 2, 8, 12, dtype=torch.bool)
    data = {"image_mask": mask, "view_mask": torch.ones(1, 2, dtype=torch.bool)}
    targets = {"depth": torch.full(mask.shape, truth * scale),
               "depth_valid": mask.clone()}
    return prediction, data, targets


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_exact_stride_lattice_and_reference(self):
        prediction, data, targets = example(2., 2.)
        targets["depth"].fill_(99.)
        targets["depth"][:, 0, ::4, ::4] = 2.
        self.assertEqual(fine_readout_loss(prediction, data, targets, {}).item(), 0.)

    def test_depth_unit_and_evaluation_range(self):
        prediction, data, targets = example(6., 5., scale=2.)
        expected = torch.log(torch.tensor(6. / 5.)) + .2 / 5.
        loss = fine_readout_loss(prediction, data, targets, {"eval_range": [9., 11.]})
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(bool((prediction["fine"]["depth"].grad > 0).all()))
        empty = fine_readout_loss(prediction, data, targets, {"eval_range": [1., 9.]})
        self.assertEqual(empty.item(), 0.)

    def test_no_evidence_or_labels_has_zero_finite_gradient(self):
        for reason in ("depth_valid", "image_mask", "view_mask", "ref_valid"):
            prediction, data, targets = example()
            if reason == "depth_valid":
                targets[reason].fill_(False)
                targets["depth"].fill_(float("nan"))
            elif reason == "ref_valid":
                prediction["fine"][reason].fill_(False)
            else:
                data[reason].fill_(False)
            loss = fine_readout_loss(prediction, data, targets, {})
            loss.backward()
            self.assertEqual(loss.item(), 0., reason)
            self.assertTrue(bool((prediction["fine"]["depth"].grad == 0).all()), reason)

    def test_group_mass_invariant_direction_gets_localization_gradient(self):
        # 保持组三点总概率为1，仍能沿组内分配改变实际深度；组间CE不辨别此方向。
        theta = torch.tensor(.2, requires_grad=True)
        q = torch.stack((theta, theta.new_tensor(.3), .7 - theta))
        scores = q.log().reshape(1, 3, 1, 1)
        candidates = torch.tensor([1., 2., 4.]).reshape(1, 3, 1, 1)
        result = HierarchicalHandoff("marginal")(
            scores, torch.ones_like(scores, dtype=torch.bool), candidates)
        prediction, data, targets = example(2., 2.)
        prediction["fine"]["depth"] = result["depth"].expand(1, 1, 2, 3)
        loss = fine_readout_loss(prediction, data, targets, {})
        loss.backward()
        # z=3.4-3theta=2.8 > 2，增加靠近点权重应降低误差。
        self.assertLess(theta.grad.item(), 0.)
        self.assertTrue(bool(torch.isfinite(theta.grad)))

    def test_real_model_addition_matches_weight_and_gradients(self):
        torch.manual_seed(62)
        _, v5 = tiny_pair()
        config = copy.deepcopy(v5.config)
        config.update(architecture="rulermvs_v6", handoff={"mode": "parent"})
        with patch("jar.pretraining.encoders._build_native_foundation", side_effect=TinyViT):
            model = V6Model(config, initialize=False).eval()
        model.load_v5_state(v5.state_dict())
        model.requires_grad_(False)
        model.decoder.requires_grad_(True)
        data = inputs()
        targets = {"depth": torch.full_like(data["image_mask"], 3., dtype=torch.float32),
                   "depth_valid": data["image_mask"].clone()}
        domain = {"near": 1., "far": 10., "depth_scale": 1.}
        prediction = model(data, domain)
        plain = objective(prediction, data, targets, domain, {})
        explicit_zero = objective(prediction, data, targets, domain, {"fine_readout_weight": 0.})
        self.assertTrue(torch.equal(plain["total"], explicit_zero["total"]))
        terms = objective(prediction, data, targets, domain, {"fine_readout_weight": .15})
        torch.testing.assert_close(terms["total"], plain["total"] + .15 * terms["fine_readout"])
        terms["total"].backward()
        gradients = [p.grad for p in model.decoder.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(g.isfinite().all()) for g in gradients))
        self.assertTrue(any(bool((g != 0).any()) for g in gradients))


if __name__ == "__main__":
    unittest.main()
