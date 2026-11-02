"""平均速度恒等式、监督责任与坐标单位的 CPU 检验。"""

import unittest

import torch

from .losses import bridge_samples, masked_mean, meanflow_pair, target_coordinates


def fixture(scale=1.):
    lower = torch.tensor([1., 2., 7.]).reshape(1, 3, 1, 1) / scale
    upper = lower + 2. / scale
    samples = (lower + upper) / 2
    context = {"lower": lower, "upper": upper, "width": upper - lower,
               "sampling_depth": samples, "anchor": torch.zeros_like(lower),
               "valid": torch.ones_like(lower, dtype=torch.bool)}
    truth = torch.tensor([[[[2.75 / scale]]]])
    return context, truth, torch.ones_like(truth, dtype=torch.bool)


class FlowLossTests(unittest.TestCase):
    def test_affine_jvp_and_parameter_gradient(self):
        q = torch.tensor([[[[.2, -.3]]]], dtype=torch.float64)
        r = torch.full((1, 1, 1, 1), .1, dtype=q.dtype)
        t = torch.full_like(r, .7)
        v = torch.full_like(q, .6)
        weight = torch.tensor(.3, dtype=q.dtype, requires_grad=True)

        def field(state, left, right):
            return weight * state + .2 * right - .1 * left

        estimate, target = meanflow_pair(field, q, r, t, v)
        expected = v - (t - r) * (weight.detach() * v + .2)
        torch.testing.assert_close(target, expected)
        self.assertFalse(target.requires_grad)
        (estimate - target).square().mean().backward()
        self.assertTrue(torch.isfinite(weight.grad))
        self.assertGreater(weight.grad.abs().item(), 0.)

    def test_equal_times_reduce_to_instantaneous_matching(self):
        q = torch.ones(1, 3, 2, 2)
        t = torch.full((1, 1, 1, 1), .3)
        v = torch.full_like(q, -.5)
        _, target = meanflow_pair(lambda x, r, t: x.square() + t, q, t, t, v)
        torch.testing.assert_close(target, v, rtol=0, atol=0)

    def test_constant_average_velocity_recovers_target(self):
        anchor = torch.tensor([-.8, .9])
        endpoint = torch.tensor([.4, -.1])
        velocity = anchor - endpoint
        for steps in (1, 2, 4):
            state = anchor.clone()
            for _ in range(steps):
                state = state - velocity / steps
            torch.testing.assert_close(state, endpoint)

    def test_only_nearest_reachable_cell_is_supervised(self):
        context, truth, valid = fixture()
        targets = target_coordinates(context, truth, valid)
        self.assertEqual(targets["covered"].flatten().tolist(), [True, True, False])
        self.assertEqual(targets["assigned"].flatten().tolist(), [False, True, False])
        self.assertEqual(targets["endpoint"].flatten().tolist(), [0., -.25, 0.])

    def test_unreachable_and_invalid_targets_preserve_context(self):
        context, truth, valid = fixture()
        before = {k: v.clone() for k, v in context.items()}
        for bad in (100., float("nan"), -1.):
            targets = target_coordinates(context, truth * 0 + bad, valid)
            self.assertFalse(targets["assigned"].any())
            torch.testing.assert_close(targets["endpoint"], context["anchor"], rtol=0, atol=0)
            samples = bridge_samples(targets, torch.Generator().manual_seed(9))
            self.assertTrue(torch.isfinite(samples["state"]).all())
        for k, value in context.items():
            torch.testing.assert_close(value, before[k], rtol=0, atol=0)

    def test_unit_scale_does_not_change_fraction_or_responsibility(self):
        a = target_coordinates(*fixture())
        b = target_coordinates(*fixture(37.))
        for name in a:
            torch.testing.assert_close(a[name], b[name])

    def test_bridge_endpoints_and_random_stream(self):
        targets = target_coordinates(*fixture())
        generators = [torch.Generator().manual_seed(42) for _ in range(2)]
        a, b = [bridge_samples(targets, gen) for gen in generators]
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(5, generator=generators[0]),
                                   torch.rand(5, generator=generators[1]), rtol=0, atol=0)
        self.assertTrue((a["r"] <= a["t"]).all())
        torch.testing.assert_close(a["state"],
                                   (1 - a["t"]) * targets["endpoint"] + a["t"] * a["start"])

    def test_empty_loss_has_finite_zero_gradient(self):
        value = torch.randn(2, 3, requires_grad=True)
        loss = masked_mean(value.square(), torch.zeros_like(value, dtype=torch.bool))
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        torch.testing.assert_close(value.grad, torch.zeros_like(value))


if __name__ == "__main__":
    unittest.main()
