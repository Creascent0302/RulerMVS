"""相对证据的数学性质、缺失输入和训练专用干预测试。"""

import unittest

import torch

from research.near_evidence_20261030.variants import ExperimentalAttention
from research.near_evidence_20261030.ray_evidence import corrupt_local_source, relative_response


class EvidenceTests(unittest.TestCase):
    def test_common_response_is_not_depth_evidence(self):
        x = torch.full((2, 7, 3, 4), .6)
        valid = torch.ones_like(x, dtype=torch.bool)
        valid[:, 3:5, 1] = False
        torch.testing.assert_close(relative_response(x, valid), torch.zeros_like(x), atol=1e-6, rtol=0)
        torch.testing.assert_close(relative_response(x + .17, valid), relative_response(x, valid), atol=1e-6, rtol=0)

    def test_peak_order_and_mask(self):
        x = torch.tensor([.1, .7, .3, 100.]).reshape(1, 4, 1, 1).requires_grad_()
        valid = torch.tensor([1, 1, 1, 0], dtype=torch.bool).reshape_as(x)
        y = relative_response(x, valid)
        self.assertGreater(y[0, 1].item(), y[0, 2].item())
        self.assertEqual(y[0, 3].item(), 0.)
        y.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(x.grad[0, 3].item(), 0.)
        self.assertTrue(torch.equal(relative_response(x, torch.zeros_like(valid)), torch.zeros_like(x)))

    def test_attention_backward_missing_and_all_invalid(self):
        for views in (1, 2, 3):
            m = ExperimentalAttention(8, heads=2, groups=2)
            m._variant_relative = True
            ref = torch.randn(1, 8, 2, 3, requires_grad=True)
            src = torch.randn(1, views, 8, 5, 2, 3, requires_grad=True)
            valid = torch.ones(1, views, 5, 2, 3, dtype=torch.bool)
            valid[..., 0, 0] = False
            g = torch.randn(1, views, 2, 5, 2, 3)
            result = m(ref, src, valid, g, torch.ones(5))
            self.assertTrue(all(torch.isfinite(v).all() for v in result.values()))
            self.assertTrue((result['null_mass'][..., 0, 0] == 1).all())
            self.assertTrue((result['relative_similarity'][..., 0, 0] == 0).all())
            loss = result['pooled'].square().mean() + result['relative_similarity'].square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(src.grad).all())
            self.assertGreater(m.pair[0].weight.grad.abs().sum().item(), 0.)

    def test_corruption_is_local_and_has_independent_rng(self):
        source = torch.arange(2 * 3 * 8 * 6 * 4 * 6.).reshape(2, 3, 8, 6, 4, 6).requires_grad_()
        valid = torch.ones(2, 3, 6, 4, 6, dtype=torch.bool)
        before = torch.random.get_rng_state().clone()
        x = corrupt_local_source(source, valid, 1., torch.Generator().manual_seed(9))
        y = corrupt_local_source(source, valid, 1., torch.Generator().manual_seed(9))
        self.assertTrue(torch.equal(x, y))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertFalse(torch.equal(x, source))
        self.assertLessEqual(int((x != source).sum()), source.numel() // 12)
        x.square().mean().backward()
        self.assertTrue(torch.isfinite(source.grad).all())
        self.assertIs(corrupt_local_source(source, valid, 0., torch.Generator()), source)
        self.assertIs(corrupt_local_source(source, torch.zeros_like(valid), 1., torch.Generator()), source)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
