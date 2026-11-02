"""极端错面和无效组不应因风险截断而得到错误分类标签。"""

import unittest

import torch

from .objective import group_classification, group_targets


class GroupTargetTests(unittest.TestCase):
    def test_tiny_target_mass_keeps_classification_gradient(self):
        scores = torch.tensor([0., 0., 0., -200., -200., -200.]).reshape(1, 6, 1, 1)
        scores.requires_grad_(True)
        loss = group_classification(scores, torch.ones_like(scores, dtype=torch.bool),
                                    torch.ones(1, 1, 1, dtype=torch.long)).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertLess(scores.grad[:, 3:].sum().item(), -.99)

    def test_ranking_before_risk_clipping(self):
        depth = torch.tensor([1., 2.]).reshape(1, 2, 1, 1)
        target, risk, valid = group_targets(depth, depth.new_full((1, 1, 1, 1), 1000.),
                                             torch.ones_like(depth, dtype=torch.bool))
        self.assertEqual(target.item(), 1)
        self.assertTrue(bool((risk == 4.).all()))
        self.assertTrue(valid.item())

    def test_invalid_group_cannot_win_and_empty_is_masked(self):
        depth = torch.tensor([float('nan'), 1.]).reshape(1, 2, 1, 1)
        truth = depth.new_full((1, 1, 1, 1), 1000.)
        target, risk, valid = group_targets(depth, truth, torch.tensor([False, True]).reshape_as(depth))
        self.assertEqual(target.item(), 1)
        self.assertTrue(bool(torch.isfinite(risk).all()))
        _, _, empty = group_targets(depth, truth, torch.zeros_like(depth, dtype=torch.bool))
        self.assertFalse(empty.item())


if __name__ == '__main__':
    unittest.main()
