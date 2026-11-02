"""真实解码流程的职责融合回归，含checkpoint反向和父候选保持。"""

import copy
from types import SimpleNamespace
import unittest

import torch

from jar.pretraining.model import CalibratedDecoder
from research.near_evidence_20261030.test_variants import tiny_inputs, tiny_settings, objective
from research.role_fusion_20261030.variants import install


class RoleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(34)

    def build(self, **options):
        original = CalibratedDecoder(tiny_settings())
        holder = SimpleNamespace(decoder=copy.deepcopy(original))
        state = torch.random.get_rng_state().clone()
        install(holder, options)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, holder.decoder.state_dict()[name], rtol=0, atol=0)
        return original, holder.decoder

    def compare(self, original, actual):
        torch.testing.assert_close(original["depth"], actual["depth"], rtol=0, atol=0)
        for stage in ("coarse", "fine"):
            for name in ("scores", "candidate_depth", "sampling_depth", "probability"):
                torch.testing.assert_close(original[stage][name], actual[stage][name], rtol=0, atol=0)

    def test_context_initial_identity(self):
        original, variant = self.build(context=True)
        self.compare(original.eval()(tiny_inputs()), variant.eval()(tiny_inputs()))

    def test_zero_carry_recovers_exact_original_transport(self):
        original, variant = self.build(carry=True)
        # 纯诊断：sigmoid饱和为精确0，验证手动传递父质量没有改候选位置/特征。
        with torch.no_grad():
            variant.handoff_gate.net[-1].bias.fill_(-1000.)
        self.compare(original.eval()(tiny_inputs()), variant.eval()(tiny_inputs()))

    def test_carry_changes_decision_not_supply_or_continuous_positions(self):
        original, variant = self.build(carry=True)
        before = original.eval()(tiny_inputs())
        after = variant.eval()(tiny_inputs())
        for name in ("sampling_depth", "candidate_depth"):
            torch.testing.assert_close(before["fine"][name], after["fine"][name], rtol=0, atol=0)
        torch.testing.assert_close(before["coarse"]["scores"], after["coarse"]["scores"], rtol=0, atol=0)
        self.assertGreater(float((after["fine"]["scores"] - before["fine"]["scores"]).detach().abs().max()), 0.)

    def test_combined_training_reaches_all_new_modules(self):
        _, variant = self.build(context=True, carry=True)
        output = variant.train()(tiny_inputs())
        objective(output).backward()
        for marker in ("context_projection", "context_null", "handoff_gate"):
            gradients = [p.grad for name, p in variant.named_parameters() if marker in name and p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
        self.assertFalse(variant.source.reliability._forward_hooks)
        self.assertFalse(variant.source.null._forward_hooks)

    def test_missing_sources_and_reference_are_finite(self):
        _, variant = self.build(context=True, carry=True)
        for reference_missing in (False, True):
            item = tiny_inputs()
            item["view_mask"][:, 1:] = False
            if reference_missing:
                item["view_mask"][:, 0] = False
                item["image_mask"][:, 0] = False
            output = variant(item)
            self.assertTrue(torch.isfinite(output["depth"]).all())
            self.assertTrue(torch.isfinite(output["fine"]["handoff_gate"]).all())
            if reference_missing:
                self.assertEqual(int(torch.count_nonzero(output["depth"])), 0)

    def test_strict_restore(self):
        _, first = self.build(context=True, carry=True)
        _, second = self.build(context=True, carry=True)
        second.load_state_dict(first.state_dict(), strict=True)
        self.compare(first.eval()(tiny_inputs()), second.eval()(tiny_inputs()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
