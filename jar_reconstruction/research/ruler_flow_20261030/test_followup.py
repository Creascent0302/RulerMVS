"""桥起点采样与评分减法：只改变已诊断出的单一条件。"""

import unittest

import torch

from .flow import LocalVelocity, bridge_targets


class TargetedTests(unittest.TestCase):
    def test_anchor_sampling_preserves_noise_assignment_and_rng(self):
        anchor = torch.zeros(128, 2, 2, 2)
        radius = torch.ones_like(anchor)
        lo, hi = -torch.ones_like(anchor), torch.ones_like(anchor)
        truth = torch.full((128, 1, 2, 2), 1.5)
        valid = torch.ones_like(truth, dtype=torch.bool)
        first, second = [torch.Generator().manual_seed(72) for _ in range(2)]
        old = bridge_targets(anchor, radius, lo, hi, truth, valid, "flow", .05, first)
        new = bridge_targets(anchor, radius, lo, hi, truth, valid, "flow", .05, second, .5)
        for key in ("initial", "velocity", "assigned", "covered", "endpoint"):
            torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
        expected = torch.where(old["time"] < .5, 0., 2 * old["time"] - 1)
        torch.testing.assert_close(new["time"], expected, rtol=0, atol=0)
        self.assertGreater(int((new["time"] == 0).sum()), 0)
        self.assertGreater(int((new["time"] > 0).sum()), 0)
        self.assertTrue(torch.equal(first.get_state(), second.get_state()))

    def test_quality_subtraction_keeps_velocity_architecture_and_initialization(self):
        torch.manual_seed(81)
        first = LocalVelocity(8, 10, 8)
        torch.manual_seed(81)
        second = LocalVelocity(8, 10, 8, learn_quality=False)
        self.assertIsNone(second.quality)
        self.assertFalse(any(name.startswith("quality") for name, _ in second.named_parameters()))
        for name, value in second.state_dict().items():
            torch.testing.assert_close(value, first.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
