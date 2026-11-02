"""局部条件流的真实标定解码、训练边界和原模型保留测试。"""

import copy
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from jar.pretraining.model import CalibratedDecoder
from research.near_evidence_20261030.test_variants import tiny_inputs, tiny_settings
from research.role_fusion_20261030.variants import transport_with_prior
from .variants import install, training_loss


class FlowIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def build(self, **options):
        original = CalibratedDecoder(tiny_settings(peak_count=4)).requires_grad_(False).eval()
        holder = SimpleNamespace(decoder=copy.deepcopy(original))
        old_ids = {name: id(parameter) for name, parameter in holder.decoder.named_parameters()}
        rng = torch.random.get_rng_state().clone()
        install(holder, {"hidden": 8, **options})
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for name, parameter in holder.decoder.named_parameters():
            if name in old_ids:
                self.assertEqual(id(parameter), old_ids[name])
                parameter.requires_grad_(False)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, holder.decoder.state_dict()[name], rtol=0, atol=0)
        return original, holder

    def labels(self, prediction, item):
        context = prediction["_flow_context"]
        state = .15 * context["high_state"][:, :1]
        depth = (context["anchor_log"][:, :1] + context["radius"][:, :1] * state).exp().detach()
        depth = depth.repeat_interleave(4, -2).repeat_interleave(4, -1)
        return {"depth": depth.expand(-1, item["view_mask"].shape[1], -1, -1).clone(),
                "depth_valid": torch.ones_like(item["image_mask"])}

    def test_original_coarse_and_support_intervals_preserved(self):
        original, model = self.build(inference_steps=1)
        inputs = tiny_inputs()
        before = original(inputs)
        after = model.decoder(inputs)
        for name in ("scores", "candidate_depth", "sampling_depth", "probability"):
            torch.testing.assert_close(before["coarse"][name], after["coarse"][name], rtol=0, atol=0)
        context = after["_flow_context"]
        torch.testing.assert_close(context["anchor"], before["fine"]["sampling_depth"][:, 1::3], rtol=0, atol=0)
        lower = before["fine"]["lower"][:, 0::3].clamp(original.near, original.far)
        upper = before["fine"]["upper"][:, 2::3].clamp(original.near, original.far)
        torch.testing.assert_close(context["lower"], lower, rtol=0, atol=0)
        torch.testing.assert_close(context["upper"], upper, rtol=0, atol=0)

    def test_zero_velocity_equals_reduced_four_and_counts_terminal_observation(self):
        _, model = self.build(inference_steps=0)
        before = model.decoder(tiny_inputs())
        for steps in (1, 2, 4):
            model.decoder.flow_options["inference_steps"] = steps
            after = model.decoder(tiny_inputs())
            torch.testing.assert_close(before["depth"], after["depth"], rtol=0, atol=0)
            self.assertEqual(after["flow_evidence_calls"], steps + 1)

    def test_native_loss_and_flow_loss_reach_both_heads_without_old_gradients(self):
        for objective in ("direct", "flow"):
            _, model = self.build(objective=objective, inference_steps=2)
            item = tiny_inputs()
            prediction = model.decoder(item)
            prediction["unit_scale"] = 1.
            targets = self.labels(prediction, item)
            terms = training_loss(model, prediction, item, targets, {"near": 2., "far": 8.},
                                  {"_generator": torch.Generator().manual_seed(3)})
            truth = targets["depth"][:, :1, ::4, ::4]
            index = (prediction["fine"]["candidate_depth"].log() - truth.log()).abs().argmin(1)
            loss = terms["total"] + F.cross_entropy(prediction["fine"]["scores"], index)
            loss = loss + prediction["depth"].clamp_min(1e-6).log().square().mean()
            loss.backward()
            self.assertGreater(int(terms["assigned_branches"]), 0)
            for name in ("velocity", "quality"):
                gradient = getattr(model.decoder.flow_head, name).weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0.)
            self.assertTrue(all(parameter.grad is None for name, parameter in model.decoder.named_parameters()
                                if not name.startswith("flow_head.")))

    def test_training_targets_do_not_mutate_predictions_inputs_or_future_inference(self):
        _, model = self.build()
        item = tiny_inputs()
        prediction = model.decoder(item)
        prediction["unit_scale"] = 1.
        original_depth = prediction["depth"].detach().clone()
        original_mask = item["image_mask"].clone()
        targets = self.labels(prediction, item)
        target_mask = targets["depth_valid"].clone()
        for factor in (1., 100.):
            changed = {"depth": targets["depth"] * factor, "depth_valid": targets["depth_valid"]}
            training_loss(model, prediction, item, changed, {"near": 2., "far": 8.})
            torch.testing.assert_close(prediction["depth"], original_depth, rtol=0, atol=0)
        torch.testing.assert_close(item["image_mask"], original_mask)
        torch.testing.assert_close(targets["depth_valid"], target_mask)
        with torch.no_grad():
            after = model.decoder(item)
        torch.testing.assert_close(after["depth"], original_depth, rtol=0, atol=0)
        self.assertNotIn("_flow_context", after)

    def test_multiple_views_missing_sources_and_reference(self):
        _, model = self.build(inference_steps=2)
        for count in (1, 2, 3, 4):
            item = tiny_inputs()
            index = torch.arange(count) % 3
            for key in ("K", "T_cam_from_rig", "view_mask", "image_mask"):
                item[key] = item[key][:, index].clone()
            item["pyramid"] = {key: value[:, index] for key, value in item["pyramid"].items()}
            for missing in (False, True):
                if missing:
                    item["view_mask"][:, 1:] = False
                prediction = model.decoder(item)
                self.assertTrue(torch.isfinite(prediction["depth"]).all())
                self.assertTrue(torch.isfinite(prediction["flow_state"]).all())
            item["view_mask"][:, 0] = False
            item["image_mask"][:, 0] = False
            self.assertEqual(int(torch.count_nonzero(model.decoder(item)["depth"])), 0)

    def test_nonzero_updates_stay_in_original_intervals(self):
        _, model = self.build(inference_steps=4)
        with torch.no_grad():
            model.decoder.flow_head.velocity.bias.fill_(10.)
        prediction = model.decoder(tiny_inputs())
        fine = prediction["fine"]
        self.assertTrue((fine["candidate_depth"] >= fine["lower"] - 1e-5).all())
        self.assertTrue((fine["candidate_depth"] <= fine["upper"] + 1e-5).all())

    def test_strict_restore_and_seeded_noise(self):
        _, first = self.build(inference_noise_sigma=.05, inference_seed=23)
        _, second = self.build(inference_noise_sigma=.05, inference_seed=23)
        second.decoder.load_state_dict(first.decoder.state_dict(), strict=True)
        with torch.no_grad():
            left = first.decoder(tiny_inputs())
            right = second.decoder(tiny_inputs())
        torch.testing.assert_close(left["depth"], right["depth"], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
