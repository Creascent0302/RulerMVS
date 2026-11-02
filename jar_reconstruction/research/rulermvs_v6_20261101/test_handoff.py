"""层次读出的CPU算例：宽峰、遮挡分组、无效输入与训练梯度。"""

import unittest

import torch

from .handoff import HierarchicalHandoff


class HierarchicalHandoffTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(2)
        self.scores = torch.tensor([.21, .20, .19, .35, .03, .02]).log().reshape(1, 6, 1, 1)
        self.depth = torch.tensor([2., 3., 4., 20., 21., 22.]).reshape_as(self.scores)
        self.valid = torch.ones_like(self.scores, dtype=torch.bool)

    def test_wide_high_mass_surface_beats_sharp_wrong_surface(self):
        out = HierarchicalHandoff()(self.scores, self.valid, self.depth)
        self.assertEqual(self.scores.argmax(1).item(), 3)
        self.assertEqual(out['selected_group'].item(), 0)
        self.assertEqual(out['selected_index'].item(), 0)
        torch.testing.assert_close(out['group_probability'].flatten(), torch.tensor([.6, .4]))
        self.assertAlmostEqual(out['depth'].item(), (2 * .21 + 3 * .20 + 4 * .19) / .6, places=5)
        self.assertLess(out['depth'].item(), 4.)
        self.assertGreater((out['joint_probability'] * self.depth).sum().item(), 9.)
        torch.testing.assert_close(out['joint_probability'], self.scores.softmax(1))
        torch.testing.assert_close(out['conditional_probability'].sum(2), torch.ones(1, 2, 1, 1))

    def test_invalid_candidates_and_groups_receive_zero_mass(self):
        valid = self.valid.clone()
        valid[:, 0] = False
        valid[:, 3:] = False
        scores, depth = self.scores.clone(), self.depth.clone()
        scores[~valid] = float('nan')
        depth[~valid] = float('nan')
        out = HierarchicalHandoff('parent')(scores, valid, depth, torch.tensor([.01, .99]).reshape(1, 2, 1, 1))
        torch.testing.assert_close(out['group_probability'].flatten(), torch.tensor([1., 0.]))
        self.assertEqual(out['selected_index'].item(), 1)
        self.assertTrue(bool((out['joint_probability'][~valid] == 0).all()))
        self.assertGreaterEqual(out['depth'].item(), 3.)
        self.assertLessEqual(out['depth'].item(), 4.)
        self.assertTrue(all(bool(value.isfinite().all()) for value in out.values()))

    def test_parent_pool_preserves_within_group_distribution(self):
        model = HierarchicalHandoff('parent')
        torch.nn.init.constant_(model.gate.net[-1].bias, 60.)
        parent = torch.tensor([.1, .9]).reshape(1, 2, 1, 1)
        out = model(self.scores, self.valid, self.depth, parent)
        plain = HierarchicalHandoff()(self.scores, self.valid, self.depth)
        torch.testing.assert_close(out['group_probability'], parent)
        torch.testing.assert_close(out['conditional_probability'], plain['conditional_probability'])
        torch.testing.assert_close(out['scores'].softmax(1), out['joint_probability'])
        self.assertEqual(out['selected_group'].item(), 1)
        self.assertEqual(out['selected_index'].item(), 3)
        self.assertGreaterEqual(out['depth'].item(), 20.)
        self.assertLessEqual(out['depth'].item(), 22.)

    def test_gate_initialization_and_parameter_budget(self):
        model = HierarchicalHandoff('parent')
        parent = torch.tensor([.2, .8]).reshape(1, 2, 1, 1)
        out = model(self.scores, self.valid, self.depth, parent)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 129)
        self.assertEqual(sum(p.numel() for p in HierarchicalHandoff().parameters()), 0)
        self.assertAlmostEqual(out['gate'].item(), torch.sigmoid(torch.tensor(-2.)).item())

    def test_all_invalid_is_finite_and_has_zero_depth_confidence(self):
        for mode in ('marginal', 'parent'):
            invalid = torch.zeros_like(self.valid)
            values = torch.full_like(self.scores, float('nan'))
            out = HierarchicalHandoff(mode)(values, invalid, values, torch.zeros(1, 2, 1, 1))
            self.assertTrue(all(bool(value.isfinite().all()) for value in out.values()))
            for key in ('depth', 'confidence', 'valid', 'joint_probability', 'group_probability'):
                self.assertEqual(out[key].count_nonzero().item(), 0, key)

    def test_missing_source_quality_does_not_silently_reject_candidates(self):
        out = HierarchicalHandoff('parent')(
            self.scores, self.valid, self.depth, torch.ones(1, 2, 1, 1),
            source_null=torch.ones(1, 1, 1), source_valid=torch.zeros(1, 1, 1))
        self.assertTrue(out['valid'].item())
        self.assertGreater(out['depth'].item(), 0.)

    def test_single_group_and_zero_parent_fallback(self):
        for groups in (1, 2):
            size = 3 * groups
            plain = HierarchicalHandoff()(self.scores[:, :size], self.valid[:, :size], self.depth[:, :size])
            out = HierarchicalHandoff('parent')(
                self.scores[:, :size], self.valid[:, :size], self.depth[:, :size], torch.zeros(1, groups, 1, 1))
            torch.testing.assert_close(out['joint_probability'], plain['joint_probability'])
            torch.testing.assert_close(out['depth'], plain['depth'])

    def test_bfloat16_keeps_fp32_probabilities_and_finite_gradients(self):
        model = HierarchicalHandoff('parent')
        scores = self.scores.to(torch.bfloat16).requires_grad_()
        depth = self.depth.to(torch.bfloat16).requires_grad_()
        parent = torch.tensor([.2, .8]).reshape(1, 2, 1, 1).requires_grad_()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = model(scores, self.valid, depth, parent)
            loss = out['depth'].mean() - out['joint_probability'][:, 3].clamp_min(1e-8).log().mean()
        loss.backward()
        self.assertIsNone(parent.grad)
        for value in (scores.grad, depth.grad, model.gate.net[-1].weight.grad):
            self.assertIsNotNone(value)
            self.assertTrue(bool(value.isfinite().all()))
            self.assertGreater(value.abs().sum().item(), 0.)
        for key in ('depth', 'joint_probability', 'group_probability', 'conditional_probability'):
            self.assertEqual(out[key].dtype, torch.float32)

    def test_batch_spatial_shapes_and_group_mass_normalization(self):
        scores = torch.randn(2, 12, 3, 4)
        depth = torch.rand_like(scores) + 1
        valid = torch.rand_like(scores) > .2
        out = HierarchicalHandoff('parent')(scores, valid, depth, torch.rand(2, 4, 3, 4))
        self.assertEqual(out['selected_index'].shape, (2, 1, 3, 4))
        self.assertEqual(out['group_depth'].shape, (2, 4, 3, 4))
        torch.testing.assert_close(out['joint_probability'].sum(1), torch.ones(2, 3, 4))
        torch.testing.assert_close(out['group_probability'], out['joint_probability'].reshape(2, 4, 3, 3, 4).sum(2))


if __name__ == '__main__':
    unittest.main()
