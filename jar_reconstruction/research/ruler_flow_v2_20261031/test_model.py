"""小尺寸真实标定解码测试；完整图像编码器另由GPU冒烟覆盖。"""

import copy
import unittest

import torch

from jar.pretraining.model import CalibratedDecoder
from research.near_evidence_20261030.test_variants import tiny_inputs, tiny_settings

from .field import EvidenceVelocity
from .model import EvidenceFlowDecoder, EvidenceFlowModel


class TinyFoundation(torch.nn.Module):
    """只测试多参考调用复用；不替代真实ViT的GPU测试。"""

    def forward(self, image):
        value = image.mean((1, 2, 3), keepdim=True).expand(-1, 768, 1, 1)
        return [value for _ in range(4)]


class TinyMatching(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        image = inputs["matching_rgb"]
        batch, views = image.shape[:2]
        image = image.flatten(0, 1)[:, torch.arange(8) % 3]
        return {stride: torch.nn.functional.avg_pool2d(image, stride).unflatten(0, (batch, views))
                for stride in (4, 8, 16)}


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def build(self, **options):
        settings = tiny_settings(peak_count=4)
        original = CalibratedDecoder(settings).eval()
        variant = EvidenceFlowDecoder(settings, {"hidden": 8, **options}).eval()
        variant.load_v2_state(original.state_dict())
        return original, variant

    def test_zero_field_preserves_original_outputs_and_all_twelve_points(self):
        original, variant = self.build()
        for mode in (False, True):
            original.train(mode)
            variant.train(mode)
            before = original(tiny_inputs())
            after = variant(tiny_inputs())
            for name in ("depth", "confidence", "support", "valid"):
                torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
            for stage in ("coarse", "fine"):
                for name in ("scores", "probability", "sampling_depth", "candidate_depth", "selected_index"):
                    torch.testing.assert_close(before[stage][name], after[stage][name], rtol=0, atol=0)
            self.assertEqual(after["fine"]["sampling_depth"].shape[1], 12)

    def test_nonzero_field_preserves_selection_and_stays_inside_cells(self):
        original, variant = self.build(inference_steps=3)
        with torch.no_grad():
            variant.flow_field.output.bias.fill_(10.)
        before = original(tiny_inputs())
        after = variant(tiny_inputs())
        fine = after["fine"]
        torch.testing.assert_close(before["fine"]["scores"], fine["scores"], rtol=0, atol=0)
        torch.testing.assert_close(before["fine"]["selected_index"], fine["selected_index"], rtol=0, atol=0)
        self.assertTrue((fine["candidate_depth"] >= fine["lower"].clamp(2., 8.) - 1e-5).all())
        self.assertTrue((fine["candidate_depth"] <= fine["upper"].clamp(2., 8.) + 1e-5).all())
        self.assertGreater(float((before["depth"] - after["depth"]).detach().abs().sum()), 0.)

    def test_inference_never_reprojects_after_the_one_fine_evidence_pass(self):
        _, variant = self.build(inference_steps=4)
        calls = []
        observe = variant._observe

        def recorded(*args, **kwargs):
            calls.append(args[2])
            return observe(*args, **kwargs)

        variant._observe = recorded
        with torch.no_grad():
            output = variant(tiny_inputs())
        self.assertEqual(calls, [8, 4])
        self.assertEqual(output["flow_evidence_calls"], 1)
        self.assertNotIn("_flow_context", output)

    def test_real_depth_gradient_reaches_new_output_and_then_context(self):
        _, variant = self.build()
        variant.requires_grad_(False)
        variant.flow_field.requires_grad_(True)
        optimizer = torch.optim.Adam(variant.flow_field.parameters(), lr=.001)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = variant(tiny_inputs())
            output["depth"].square().mean().backward()
            gradients = [p.grad for p in variant.flow_field.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(float(variant.flow_field.output.weight.grad.abs().sum()), 0.)
            if step:
                self.assertGreater(float(variant.flow_field.context[0].weight.grad.abs().sum()), 0.)
            optimizer.step()

    def test_jvp_and_parameter_backward_do_not_touch_sampling_second_derivatives(self):
        _, variant = self.build()
        output = variant(tiny_inputs())
        context = output["_flow_context"]
        with torch.no_grad():
            variant.flow_field.output.weight.normal_(std=.01)
        state = context["anchor"].detach() * .7
        r = torch.full_like(state, .2)
        t = torch.full_like(state, .7)
        direction = torch.randn_like(state) * .1

        def evaluate(x, left, right):
            return variant.flow_field.velocity(context, x, left, right)

        value, derivative = torch.func.jvp(
            evaluate, (state, r, t), (direction, torch.zeros_like(r), torch.ones_like(t)))
        epsilon = 1e-3
        plus = evaluate(state + epsilon * direction, r, t + epsilon)
        minus = evaluate(state - epsilon * direction, r, t - epsilon)
        torch.testing.assert_close(derivative, (plus - minus) / (2 * epsilon), rtol=.01, atol=1e-5)
        target = (direction - (t - r) * derivative).detach()
        (value - target).square().mean().backward()
        self.assertTrue(torch.isfinite(variant.flow_field.output.weight.grad).all())
        self.assertGreater(float(variant.flow_field.output.weight.grad.abs().sum()), 0.)

    def test_variable_views_missing_sources_reference_and_pure_rotation(self):
        _, variant = self.build()
        for count in (1, 2, 3, 4):
            item = tiny_inputs()
            index = torch.arange(count) % 3
            for key in ("K", "T_cam_from_rig", "view_mask", "image_mask"):
                item[key] = item[key][:, index].clone()
            item["pyramid"] = {key: value[:, index] for key, value in item["pyramid"].items()}
            for degenerate in (False, True):
                if degenerate:
                    item["view_mask"][:, 1:] = False
                    item["T_cam_from_rig"][..., :3, 3] = 0
                result = variant(item)
                self.assertTrue(torch.isfinite(result["depth"]).all())
                self.assertTrue(torch.isfinite(result["fine"]["flow_state"]).all())
            item["view_mask"][:, 0] = False
            item["image_mask"][:, 0] = False
            self.assertEqual(int(torch.count_nonzero(variant(item)["depth"])), 0)

    def test_geometry_neighborhood_does_not_wrap_or_cross_disjoint_intervals(self):
        value = torch.tensor([[[[1., 100.], [100., 100.]]]])
        lower = torch.tensor([[[[0., 3.], [3., 3.]]]])
        upper = lower + 1
        valid = torch.ones_like(lower, dtype=torch.bool)
        message = EvidenceVelocity.neighborhood(value, lower, upper, valid)
        self.assertEqual(float(message[0, 0, 0, 0]), 0.)
        # A 1x2 grid only has one horizontal neighbor per position.
        value = torch.tensor([[[[0., 2.]]]])
        lower = torch.zeros_like(value)
        message = EvidenceVelocity.neighborhood(value, lower, lower + 1, lower == 0)
        torch.testing.assert_close(message, torch.tensor([[[[1., -1.]]]]), rtol=0, atol=0)

    def test_checkpoint_strict_restore_and_old_key_validation(self):
        original, first = self.build()
        _, second = self.build()
        second.load_state_dict(first.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(first(tiny_inputs())["depth"], second(tiny_inputs())["depth"], rtol=0, atol=0)
        broken = copy.deepcopy(original.state_dict())
        broken.pop("score.weight")
        with self.assertRaises(ValueError):
            second.load_v2_state(broken)

    def test_multireference_reuses_matching_and_keeps_outputs_and_missing_reference(self):
        model = EvidenceFlowModel.__new__(EvidenceFlowModel)
        torch.nn.Module.__init__(model)
        model.foundation = TinyFoundation()
        model.matching = TinyMatching()
        model.decoder = EvidenceFlowDecoder(tiny_settings(peak_count=4), {"hidden": 8})
        model.eval()
        inputs = tiny_inputs()
        inputs["rgb"] = torch.randn(1, 3, 3, 16, 16)
        inputs["matching_rgb"] = inputs["rgb"].clone()
        inputs["view_mask"][:, 1] = False
        inputs["image_mask"][:, 1] = False
        domain = {"near": 2., "far": 8., "depth_scale": 1.}
        with torch.no_grad():
            expected = {reference: model(inputs, domain, reference) for reference in range(3)}
        self.assertEqual(model.matching.calls, 3)
        model.matching.calls = 0
        actual = model.forward_references(inputs, domain)
        self.assertEqual(model.matching.calls, 1)
        for reference in range(3):
            for key in ("depth", "confidence", "valid", "support"):
                torch.testing.assert_close(expected[reference][key], actual[reference][key], rtol=0, atol=0)
        self.assertEqual(int(torch.count_nonzero(actual[1]["depth"])), 0)
        self.assertEqual(set(model.forward_references(inputs, domain, [2, 0])), {2, 0})
        model.train()
        with self.assertRaises(ValueError):
            model.forward_references(inputs, domain)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
