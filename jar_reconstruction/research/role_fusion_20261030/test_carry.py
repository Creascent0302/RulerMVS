"""粗细组分布池化的数学恒等、掩码边界和梯度隔离。"""

import unittest

import torch

from research.role_fusion_20261030.carry import HandoffGate, pool_group_scores


class CarryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.scores = torch.randn(2, 12, 3, 4)
        self.parent = torch.rand(2, 4, 3, 4)
        self.parent /= self.parent.sum(1, keepdim=True)

    def test_eta_zero_preserves_original_and_one_inherits_group_mass(self):
        torch.testing.assert_close(pool_group_scores(self.scores, self.parent, 0),
                                   self.scores, rtol=0, atol=0)
        actual = pool_group_scores(self.scores, self.parent, 1)
        mass = actual.softmax(1).reshape(2, 4, 3, 3, 4).sum(2)
        torch.testing.assert_close(mass, self.parent, rtol=2e-6, atol=1e-7)

    def test_group_conditional_and_geometric_pool(self):
        original = self.scores.reshape(2, 4, 3, 3, 4)
        actual = pool_group_scores(self.scores, self.parent, .35).reshape_as(original)
        torch.testing.assert_close(actual.softmax(2), original.softmax(2), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(actual.argmax(2), original.argmax(2), rtol=0, atol=0)
        fine_mass = original.logsumexp(2).softmax(1)
        expected = fine_mass.pow(.65) * self.parent.pow(.35)
        expected /= expected.sum(1, keepdim=True)
        actual_mass = actual.flatten(1, 2).softmax(1).reshape_as(actual).sum(2)
        torch.testing.assert_close(actual_mass, expected, rtol=1e-6, atol=1e-7)

    def test_zero_prior_one_group_and_zero_sources_are_finite(self):
        scores = torch.zeros(1, 3, 2, 3, requires_grad=True)
        parent = torch.zeros(1, 1, 2, 3)
        features = torch.zeros(1, 8, 3, 2, 3)
        null = torch.ones(1, 1, 3, 2, 3)
        for sources in (torch.zeros(1, 0, 3, 2, 3, dtype=torch.bool),
                        torch.zeros(1, 2, 3, 2, 3, dtype=torch.bool)):
            corrected, gate = HandoffGate(8)(features, scores, parent, null, sources)
            self.assertTrue(torch.isfinite(corrected).all())
            self.assertTrue(torch.isfinite(gate).all())
            torch.testing.assert_close(corrected, scores, rtol=0, atol=0)
            torch.testing.assert_close(gate, torch.full_like(gate, torch.sigmoid(torch.tensor(-2.)).item()))

    def test_gate_and_logits_train_without_confidence_shortcuts(self):
        model = HandoffGate(8)
        features = torch.randn(2, 8, 12, 3, 4, requires_grad=True)
        scores = self.scores.clone().requires_grad_()
        parent = self.parent.clone().requires_grad_()
        null = torch.rand(2, 1, 12, 3, 4, requires_grad=True)
        sources = torch.rand(2, 2, 12, 3, 4) > .3
        corrected, gate = model(features, scores, parent, null, sources)
        loss = (corrected.softmax(1) * torch.randn_like(corrected)).sum()
        loss.backward()
        self.assertIsNone(features.grad)
        self.assertIsNone(parent.grad)
        self.assertIsNone(null.grad)
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertGreater(float(scores.grad.abs().max()), 0)
        self.assertTrue(torch.isfinite(model.net[-1].weight.grad).all())
        self.assertGreater(float(model.net[-1].weight.grad.abs().max()), 0)
        self.assertTrue(((gate > 0) & (gate < 1)).all())

    def test_extreme_logits_and_zero_mass_groups_remain_finite(self):
        scores = torch.full((1, 12, 1, 1), -1000., requires_grad=True)
        with torch.no_grad():
            scores[:, :3] = 1000.
        parent = torch.tensor([0., 1., 0., 0.]).reshape(1, 4, 1, 1)
        corrected = pool_group_scores(scores, parent, .5)
        self.assertTrue(torch.isfinite(corrected).all())
        probability = corrected.softmax(1)
        torch.testing.assert_close(probability.sum(1), torch.ones(1, 1, 1))
        corrected.sum().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
