"""组内特征池化的CPU算例；不启动GPU，不改V6已有训练或推理入口。"""

import unittest

import torch

from .group_feature import pool_selected_group_features


class GroupFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(203)

    def test_uses_selected_group_only_and_matches_weighted_value(self):
        features = torch.tensor([100., 200., 300., 2., 4., 8.]).reshape(1, 1, 6, 1, 1)
        probability = torch.tensor([.2, .3, .5, .2, .3, .5]).reshape(1, 2, 3, 1, 1)
        group = torch.ones(1, 1, 1, 1, dtype=torch.long)
        output = pool_selected_group_features(features, group, probability)
        torch.testing.assert_close(output, torch.full((1, 1, 1, 1), 5.6))
        features[:, :, :3] *= 1000
        torch.testing.assert_close(pool_selected_group_features(features, group, probability), output)

    def test_one_hot_is_exact_original_winner_feature(self):
        features = torch.randn(2, 4, 12, 3, 5)
        group = torch.randint(4, (2, 1, 3, 5))
        probability = torch.zeros(2, 4, 3, 3, 5)
        probability[:, :, 1] = 1
        index = (group * 3 + 1)[:, None].expand(-1, 4, -1, -1, -1)
        expected = features.gather(2, index)[:, :, 0]
        actual = pool_selected_group_features(features, group, probability)
        self.assertTrue(torch.equal(actual, expected))

    def test_near_tie_is_continuous_while_winner_feature_jumps(self):
        features = torch.tensor([0., 10., 5.]).reshape(1, 1, 3, 1, 1)
        group = torch.zeros(1, 1, 1, 1, dtype=torch.long)
        outputs, winners = [], []
        for epsilon in (-1e-4, 1e-4):
            logits = torch.tensor([epsilon, -epsilon, -20.]).reshape(1, 1, 3, 1, 1)
            probability = logits.softmax(2)
            outputs.append(pool_selected_group_features(features, group, probability))
            winners.append(features.flatten()[logits.argmax(2).item()])
        self.assertEqual((winners[1] - winners[0]).abs().item(), 10.)
        self.assertLess((outputs[1] - outputs[0]).abs().item(), .0011)

    def test_invalid_or_nonfinite_candidates_are_removed_and_renormalized(self):
        features = torch.tensor([2., float('nan'), 8.]).reshape(1, 1, 3, 1, 1)
        probability = torch.tensor([.2, .3, .5]).reshape(1, 1, 3, 1, 1)
        group = torch.zeros(1, 1, 1, 1, dtype=torch.long)
        output = pool_selected_group_features(features, group, probability)
        self.assertAlmostEqual(output.item(), (2 * .2 + 8 * .5) / .7, places=5)
        valid = torch.tensor([True, True, False]).reshape(1, 3, 1, 1)
        output = pool_selected_group_features(features, group, probability, valid)
        self.assertEqual(output.item(), 2.)

    def test_all_invalid_and_zero_mass_have_finite_zero_gradients(self):
        group = torch.zeros(1, 1, 1, 1, dtype=torch.long)
        for invalid in (False, True):
            features = torch.full((1, 2, 3, 1, 1), float('nan') if invalid else 2., requires_grad=True)
            probability = torch.full((1, 1, 3, 1, 1), float('nan') if invalid else 0., requires_grad=True)
            output = pool_selected_group_features(features, group, probability)
            self.assertEqual(output.count_nonzero().item(), 0)
            output.sum().backward()
            for gradient in (features.grad, probability.grad):
                self.assertTrue(bool(gradient.isfinite().all()))
                self.assertEqual(gradient.count_nonzero().item(), 0)

    def test_valid_feature_and_conditional_logit_gradients_are_finite_nonzero(self):
        features = torch.randn(2, 4, 6, 3, 5, requires_grad=True)
        logits = torch.randn(2, 2, 3, 3, 5, requires_grad=True)
        group = torch.ones(2, 1, 3, 5, dtype=torch.long)
        output = pool_selected_group_features(features, group, logits.softmax(2))
        output.square().mean().backward()
        for gradient in (features.grad, logits.grad):
            self.assertTrue(bool(gradient.isfinite().all()))
            self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertEqual(features.grad[:, :, :3].count_nonzero().item(), 0)
        self.assertEqual(logits.grad[:, :1].count_nonzero().item(), 0)

    def test_bfloat16_output_and_backward(self):
        features = torch.randn(1, 4, 6, 2, 3, dtype=torch.bfloat16, requires_grad=True)
        logits = torch.randn(1, 2, 3, 2, 3, requires_grad=True)
        group = torch.zeros(1, 1, 2, 3, dtype=torch.long)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = pool_selected_group_features(features, group, logits.softmax(2))
            loss = output.float().square().mean()
        self.assertEqual(output.dtype, torch.bfloat16)
        loss.backward()
        self.assertTrue(bool(features.grad.isfinite().all()))
        self.assertTrue(bool(logits.grad.isfinite().all()))

    def test_shape_contract(self):
        features = torch.randn(1, 4, 6, 2, 3)
        probability = torch.rand(1, 2, 3, 2, 3)
        group = torch.zeros(1, 1, 2, 3, dtype=torch.long)
        with self.assertRaises(ValueError):
            pool_selected_group_features(features[:, :, :5], group, probability)
        with self.assertRaises(ValueError):
            pool_selected_group_features(features, group[:, 0], probability)
        with self.assertRaises(ValueError):
            pool_selected_group_features(features, group, probability[:, :1])


if __name__ == '__main__':
    unittest.main()
