"""CPU 数学测试：配对路径、分支支持、单位变化与显式 Euler。"""

import unittest

import torch

from research.ruler_flow_20261030.flow import (
    bridge_targets,
    clamp_state,
    local_coordinates,
    velocity_loss,
)


def fixture(scale=1.):
    anchor = torch.tensor([5., 15., 25., 35.], dtype=torch.float64).reshape(1, 4, 1, 1) / scale
    lower = anchor - 2. / scale
    upper = anchor + 2. / scale
    coordinates = local_coordinates(anchor, lower, upper)
    truth = torch.tensor([[[[14.5]]]], dtype=torch.float64) / scale
    valid = torch.ones_like(truth, dtype=torch.bool)
    return coordinates, truth, valid


def bridge(objective, sigma=.05, scale=1., seed=12):
    coordinates, truth, valid = fixture(scale)
    return bridge_targets(*coordinates, truth, valid, objective, sigma,
                          torch.Generator().manual_seed(seed))


class FlowMathTest(unittest.TestCase):
    def test_direct_and_flow_share_noise_assignment_and_target(self):
        coordinates, truth, valid = fixture()
        generators = [torch.Generator().manual_seed(12) for _ in range(2)]
        direct = bridge_targets(*coordinates, truth, valid, "direct", .05, generators[0])
        flow = bridge_targets(*coordinates, truth, valid, "flow", .05, generators[1])
        for name in ("initial", "endpoint", "velocity", "assigned", "covered"):
            torch.testing.assert_close(direct[name], flow[name], rtol=0, atol=0)
        torch.testing.assert_close(direct["state"], direct["initial"], rtol=0, atol=0)
        torch.testing.assert_close(flow["state"],
                                   (1 - flow["time"]) * flow["initial"] + flow["time"] * flow["endpoint"])
        self.assertGreater(flow["time"].item(), 0.)
        self.assertLess(flow["time"].item(), 1.)
        torch.testing.assert_close(torch.rand(3, generator=generators[0]),
                                   torch.rand(3, generator=generators[1]), rtol=0, atol=0)

    def test_unreachable_truth_does_not_expand_interval(self):
        coordinates, truth, valid = fixture()
        original = [value.clone() for value in coordinates]
        result = bridge_targets(*coordinates, truth * 10, valid, "flow", .05,
                                torch.Generator().manual_seed(12))
        self.assertFalse(result["covered"].any())
        self.assertFalse(result["assigned"].any())
        torch.testing.assert_close(result["endpoint"], result["initial"], rtol=0, atol=0)
        self.assertEqual(result["velocity"].count_nonzero().item(), 0)
        for actual, expected in zip(coordinates, original):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue((result["state"] >= coordinates[2]).all())
        self.assertTrue((result["state"] <= coordinates[3]).all())

    def test_competing_peaks_are_not_averaged(self):
        coordinates, truth, _ = fixture()
        result = bridge("flow", sigma=0.)
        self.assertEqual(result["assigned"].flatten().tolist(), [False, True, False, False])
        self.assertEqual(result["assigned"].sum().item(), 1)
        # 可达分支到14.5；另外三峰保持各自起点，而不是共同移动到真实深度。
        depths = (coordinates[0] + coordinates[1] * result["endpoint"]).exp()
        expected = torch.tensor([5., truth.item(), 25., 35.], dtype=torch.float64).reshape_as(depths)
        torch.testing.assert_close(depths, expected)

    def test_overlapping_intervals_assign_only_nearest_anchor(self):
        anchor = torch.tensor([5., 5.2, 15., 25.]).reshape(1, 4, 1, 1)
        coordinates = local_coordinates(anchor, anchor - 1, anchor + 1)
        truth = torch.tensor([[[[5.19]]]])
        result = bridge_targets(*coordinates, truth, torch.ones_like(truth, dtype=torch.bool),
                                "flow", 0., torch.Generator().manual_seed(12))
        self.assertEqual(result["covered"].sum().item(), 2)
        self.assertEqual(result["assigned"].flatten().tolist(), [False, True, False, False])

    def test_local_coordinates_and_bridge_are_unit_invariant(self):
        original, _, _ = fixture()
        scaled, _, _ = fixture(scale=37.)
        for actual, expected in zip(scaled[1:], original[1:]):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        first, second = bridge("flow"), bridge("flow", scale=37.)
        for name in ("initial", "endpoint", "state", "velocity", "assigned", "covered"):
            torch.testing.assert_close(first[name], second[name], rtol=1e-12, atol=1e-12)

    def test_constant_velocity_euler_recovers_endpoint(self):
        coordinates, _, _ = fixture()
        result = bridge("flow", sigma=.05)
        for steps in (1, 2, 4):
            state = result["initial"].clone()
            for _ in range(steps):
                state = clamp_state(state + result["velocity"] / steps,
                                    coordinates[2], coordinates[3])
            torch.testing.assert_close(state, result["endpoint"], rtol=1e-12, atol=1e-12)
        # 这只验证数值求解器代数，不表示学得的速度场实际为常量。

    def test_invalid_truth_and_empty_velocity_supervision_are_finite(self):
        coordinates, truth, valid = fixture()
        truth.fill_(float("nan"))
        valid.fill_(False)
        result = bridge_targets(*coordinates, truth, valid, "flow", .05,
                                torch.Generator().manual_seed(12))
        self.assertFalse(result["assigned"].any())
        self.assertTrue(torch.isfinite(result["state"]).all())
        self.assertTrue(torch.isfinite(result["velocity"]).all())
        prediction = torch.randn_like(result["velocity"], requires_grad=True)
        loss = velocity_loss(prediction, result["velocity"], result["assigned"])
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(prediction.grad.count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
