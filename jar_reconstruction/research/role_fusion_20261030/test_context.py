"""上下文仲裁：恒等起点、显式checkpoint上下文、掩码及梯度。"""

import copy
import unittest

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from jar.pretraining.model import CandidateViewAttention
from research.role_fusion_20261030.context import install_context


def inputs(depth=5, height=3, width=4):
    reference = F.normalize(torch.randn(1, 8, height, width), dim=1)
    source = F.normalize(torch.randn(1, 3, 8, depth, height, width), dim=2)
    valid = torch.rand(1, 3, depth, height, width) > .2
    derivative = torch.randn(1, 3, 2, depth, height, width)
    half_width = torch.linspace(.1, 1., depth)
    context = torch.randn(1, 12, height, width)
    return reference, source, valid, derivative, half_width, True, context


def loss(output):
    channels = torch.arange(output["pooled"].shape[1])[None, :, None, None, None]
    return (output["pooled"] * channels).square().mean() + output["null_mass"].square().mean()


class ContextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)

    def build(self):
        original = CandidateViewAttention(8, heads=2, groups=2)
        variant = copy.deepcopy(original)
        old_ids = {name: id(value) for name, value in variant.named_parameters()}
        rng = torch.random.get_rng_state().clone()
        install_context(variant, 12)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for name, value in variant.named_parameters():
            if name in old_ids:
                self.assertEqual(id(value), old_ids[name])
        return original, variant

    def test_zero_init_preserves_all_outputs_and_inputs(self):
        original, variant = self.build()
        item = inputs()
        before = [x.clone() if torch.is_tensor(x) else x for x in item]
        expected, actual = original(*item[:-1]), variant(*item)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        for old, new in zip(before, item):
            if torch.is_tensor(old):
                torch.testing.assert_close(old, new, rtol=0, atol=0)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, variant.state_dict()[key], rtol=0, atol=0)

    def test_new_parameters_have_finite_nonzero_gradients(self):
        _, variant = self.build()
        item = inputs()
        loss(variant(*item)).backward()
        for branch in (variant.context_projection, variant.context_null):
            self.assertTrue(torch.isfinite(branch.weight.grad).all())
            self.assertGreater(float(branch.weight.grad.abs().max()), 0)
        self.assertEqual(len(variant.reliability._forward_hooks), 0)
        self.assertEqual(len(variant.null._forward_hooks), 0)

    def test_all_invalid_sources_stay_null(self):
        _, variant = self.build()
        item = list(inputs())
        item[2] = torch.zeros_like(item[2])
        item[1] = torch.zeros_like(item[1])
        with torch.no_grad():
            variant.context_projection.weight.normal_()
            variant.context_null.weight.normal_()
        output = variant(*item)
        self.assertEqual(int(torch.count_nonzero(output["weights"])), 0)
        self.assertEqual(int(torch.count_nonzero(output["pooled"])), 0)
        torch.testing.assert_close(output["null_mass"], torch.ones_like(output["null_mass"]), rtol=0, atol=0)
        self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))
        loss(output).backward()
        self.assertTrue(torch.isfinite(variant.context_projection.weight.grad).all())

    def test_checkpoint_two_stages_matches_eager_gradients(self):
        _, first = self.build()
        with torch.no_grad():
            first.context_projection.weight.normal_(std=.03)
            first.context_null.weight.normal_(std=.03)
        second = copy.deepcopy(first)
        coarse, fine = inputs(), inputs(depth=3, height=6, width=8)
        (loss(first(*coarse)) + loss(first(*fine))).backward()
        (loss(checkpoint(second, *coarse, use_reentrant=False)) +
         loss(checkpoint(second, *fine, use_reentrant=False))).backward()
        for (name, left), (_, right) in zip(first.named_parameters(), second.named_parameters()):
            self.assertIsNotNone(left.grad, name)
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-6, atol=1e-7)
        self.assertEqual(len(second.reliability._forward_hooks), 0)

    def test_hoisted_reference_and_strict_restore(self):
        _, first = self.build()
        _, second = self.build()
        with torch.no_grad():
            first.context_projection.weight.normal_(std=.03)
            first.context_null.weight.normal_(std=.03)
        second.load_state_dict(first.state_dict(), strict=True)
        first.hoist_reference = second.hoist_reference = True
        item = inputs()
        left, right = first(*item), second(*item)
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
