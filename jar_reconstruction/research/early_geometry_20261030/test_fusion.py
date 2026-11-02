"""单目差分读取的恒等起点、实际梯度与粗细流程测试。"""

import copy
from types import SimpleNamespace
import unittest

import torch

from jar.pretraining.model import CalibratedDecoder
from research.early_geometry_20261030.fusion import install_fusion
from research.near_evidence_20261030.test_variants import tiny_inputs, tiny_settings, objective


class FusionTests(unittest.TestCase):
    def build(self):
        original = CalibratedDecoder(tiny_settings())
        model = SimpleNamespace(decoder=copy.deepcopy(original))
        rng = torch.random.get_rng_state().clone()
        install_fusion(model)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        return original, model.decoder

    def test_zero_gain_preserves_outputs_and_original_state(self):
        original, variant = self.build()
        original.eval()
        variant.eval()
        first, second = original(tiny_inputs()), variant(tiny_inputs())
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, variant.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(first["depth"], second["depth"], rtol=0, atol=0)
        for stage in ("coarse", "fine"):
            for key in ("scores", "candidate_depth", "probability"):
                torch.testing.assert_close(first[stage][key], second[stage][key], rtol=0, atol=0)

    def test_gain_has_finite_nonzero_gradient_and_changes_scores(self):
        _, variant = self.build()
        variant.train()
        before = variant(tiny_inputs())
        objective(before).backward()
        gradient = variant.late_gain.grad
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0)
        with torch.no_grad():
            variant.late_gain.add_(0.2)
        after = variant(tiny_inputs())
        self.assertGreater(float((after["coarse"]["scores"] - before["coarse"]["scores"]).detach().abs().max()), 0)
        self.assertEqual(before["coarse"]["sampling_depth"].shape, after["coarse"]["sampling_depth"].shape)
        self.assertEqual(before["fine"]["sampling_depth"].shape, after["fine"]["sampling_depth"].shape)

    def test_no_source_is_finite(self):
        _, variant = self.build()
        inputs = tiny_inputs()
        inputs["view_mask"][:, 1:] = False
        with torch.no_grad():
            variant.late_gain.fill_(0.1)
        output = variant(inputs)
        self.assertTrue(bool(torch.isfinite(output["depth"]).all()))
        objective(output).backward()
        self.assertTrue(bool(torch.isfinite(variant.late_gain.grad).all()))

    def test_checkpoint_roundtrip(self):
        _, first = self.build()
        second = CalibratedDecoder(tiny_settings())
        holder = SimpleNamespace(decoder=second)
        install_fusion(holder)
        second.load_state_dict(first.state_dict(), strict=True)
        torch.testing.assert_close(first(tiny_inputs())["depth"], second(tiny_inputs())["depth"], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
