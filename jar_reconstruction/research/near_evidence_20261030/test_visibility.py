"""CPU 真投影小场景：可见/遮挡/未知、候选资格、梯度和 checkpoint hook。"""

import math
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from research.near_evidence_20261030.visibility import (
    auxiliary_loss, capture_attention, source_indices, visibility_targets,
)


class ToySource(nn.Module):
    def __init__(self, views):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(1, views + 1, 3, 1, 1))
        self.calls = 0

    def forward(self, reference, source, valid, derivative, half):
        self.calls += 1
        logits = self.logits.expand(valid.shape[0], valid.shape[1] + 1, *valid.shape[2:])
        mask = torch.cat((valid, torch.ones_like(valid[:, :1])), 1)
        weight = logits.masked_fill(~mask, -1e4).softmax(1)
        return {"weights": weight[:, :-1], "null_mass": weight[:, -1:]}


class ToyModel(nn.Module):
    def __init__(self, compact=True, reference=2, checkpointed=False):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.source = ToySource(3 if compact else 4)
        self.compact = compact
        self.reference = reference
        self.checkpointed = checkpointed

    def attention_arguments(self, valid):
        return None, None, valid, None, None

    def forward(self, inputs, scale=1., center=4., interval=0.2):
        source = self.decoder.source
        indices = source_indices(4, 3 if self.compact else 4, self.reference)
        result = {"unit_scale": scale, "reference_index": self.reference}
        for name, stride in (("coarse", 8), ("fine", 4)):
            h, w = (value // stride for value in inputs["rgb"].shape[-2:])
            candidate = torch.tensor([center - 1., center, center + 2.]) / scale
            candidate = candidate[None, :, None, None].expand(1, 3, h, w)
            valid = torch.ones(1, len(indices), 3, h, w, dtype=torch.bool)
            if self.reference in indices:
                valid[:, indices.index(self.reference)] = False
            arguments = self.attention_arguments(valid)
            if self.checkpointed:
                checkpoint(source, *arguments, use_reentrant=False)
            else:
                source(*arguments)
            result[name] = {"sampling_depth": candidate, "candidate_depth": candidate + 100.,
                            "lower": candidate - interval / scale, "upper": candidate + interval / scale,
                            "valid": torch.ones_like(candidate, dtype=torch.bool)}
        result["depth"] = torch.full((1, 1, 16, 16), 4.) + next(source.parameters()).sum() * 0
        return result


class RealAttentionModel(ToyModel):
    def __init__(self):
        super().__init__(checkpointed=True)
        from jar.pretraining.model import CandidateViewAttention

        self.decoder.source = CandidateViewAttention(8, heads=2, groups=2)

    def attention_arguments(self, valid):
        batch, views, depth, height, width = valid.shape
        generator = torch.Generator().manual_seed(17)
        reference = torch.nn.functional.normalize(torch.randn(batch, 8, height, width, generator=generator), dim=1)
        source = torch.nn.functional.normalize(torch.randn(batch, views, 8, depth, height, width,
                                                          generator=generator), dim=2)
        derivative = torch.ones(batch, views, 2, depth, height, width)
        return reference, source, valid, derivative, torch.ones(depth)


def scene():
    K = torch.tensor([[8., 0., 8.], [0., 8., 8.], [0., 0., 1.]])[None, None].repeat(1, 4, 1, 1)
    T = torch.eye(4)[None, None].repeat(1, 4, 1, 1)
    T[0, 0, 0, 3] = -.5
    T[0, 1, 0, 3] = .5
    T[0, 3, 1, 3] = -.5
    inputs = {"rgb": torch.zeros(1, 4, 3, 16, 16), "matching_rgb": torch.zeros(1, 4, 3, 32, 32),
              "K": K, "T_cam_from_rig": T, "view_mask": torch.ones(1, 4, dtype=torch.bool),
              "image_mask": torch.ones(1, 4, 16, 16, dtype=torch.bool)}
    depth = torch.full((1, 4, 16, 16), 4.)
    depth[:, 1] = 2.  # 源1在真实表面前有遮挡物；源0可见。
    mask = torch.ones_like(depth, dtype=torch.bool)
    mask[:, 3] = False  # 源3没有深度，不允许当作负例。
    targets = {"depth": depth, "depth_valid": mask}
    domain = {"correspondence": True, "eval_range": [1., 10.]}
    return inputs, targets, domain


class VisibilityTests(unittest.TestCase):
    def run_model(self, model=None, **kwargs):
        inputs, targets, domain = scene()
        model = model or ToyModel()
        with capture_attention(model) as captured:
            prediction = model(inputs, **kwargs)
        return model, inputs, targets, domain, captured, prediction

    def test_positive_occluded_unknown_and_veto_gradient(self):
        model, inputs, targets, domain, captured, prediction = self.run_model()
        original = targets["depth_valid"].clone()
        inputs["image_mask"][0, 2, 4, 4] = False
        output = auxiliary_loss(prediction, inputs, targets, domain, captured)
        self.assertGreater(output["positive_pairs"], 0)
        self.assertGreater(output["occluded_pairs"], 0)
        self.assertGreater(output["stages"]["fine"]["unknown_eligible_pairs"], 0)
        self.assertAlmostEqual(float(output["loss"].detach()), -math.log(.75), places=6)
        self.assertAlmostEqual(float(output["null_mass_mean"]), .25, places=6)
        self.assertTrue(torch.equal(original, targets["depth_valid"]))
        output["loss"].backward()
        gradient = model.decoder.source.logits.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient[0, 1, 1]), 0.)
        self.assertLess(float(gradient[0, 3, 1]), 0.)

    def test_compact_and_full_slots_match(self):
        records = []
        for compact in (True, False):
            _, inputs, targets, domain, captured, prediction = self.run_model(ToyModel(compact=compact))
            records.append(auxiliary_loss(prediction, inputs, targets, domain, captured))
        for key in ("positive_pairs", "occluded_pairs", "known_pairs", "supervised_rays"):
            self.assertEqual(records[0][key], records[1][key])
        torch.testing.assert_close(records[0]["loss"], records[1]["loss"])
        self.assertEqual(source_indices(4, 3, 2), [0, 1, 3])

    def test_sampling_depth_used_and_metric_scale_preserved(self):
        losses = []
        for scale in (1., 10.):
            _, inputs, targets, domain, captured, prediction = self.run_model(scale=scale)
            result = auxiliary_loss(prediction, inputs, targets, domain, captured)
            losses.append(result["loss"])
            self.assertGreater(result["known_pairs"], 0)
        torch.testing.assert_close(*losses)

    def test_candidate_interval_must_cover_gt(self):
        _, inputs, targets, domain, captured, prediction = self.run_model(center=10.)
        result = auxiliary_loss(prediction, inputs, targets, domain, captured)
        self.assertEqual(result["known_pairs"], 0)
        self.assertEqual(result["supervised_rays"], 0)
        self.assertEqual(float(result["loss"].detach()), 0.)
        result["loss"].backward()

    def test_exact_projected_distance_filters_label_transfer(self):
        _, inputs, targets, domain, captured, prediction = self.run_model(center=1., interval=10.)
        # 最近的采样候选为3，GT为4；严格0.01个F4像素阈值排除两台非同心源。
        result = auxiliary_loss(prediction, inputs, targets, domain, captured, projection_limit=.01)
        self.assertEqual(result["known_pairs"], 0)

    def test_masks_missing_and_uncertified_domains_skip(self):
        _, inputs, targets, domain, captured, prediction = self.run_model()
        inputs["view_mask"][:, 1] = False
        result = auxiliary_loss(prediction, inputs, targets, domain, captured)
        self.assertEqual(result["occluded_pairs"], 0)
        self.assertGreater(result["positive_pairs"], 0)
        self.assertEqual(float(result["loss"].detach()), 0.)
        skipped = auxiliary_loss(prediction, inputs, targets, {"correspondence": False}, captured)
        self.assertTrue(skipped["skipped"])
        self.assertEqual(skipped["known_pairs"], 0)
        inputs["image_mask"][:, 2] = False
        empty = auxiliary_loss(prediction, inputs, targets, domain, captured)
        self.assertEqual(empty["known_pairs"], 0)

    def test_farther_source_depth_is_unknown(self):
        _, inputs, targets, domain, captured, prediction = self.run_model()
        targets["depth"][:, 0] = 8.
        result = auxiliary_loss(prediction, inputs, targets, domain, captured)
        self.assertEqual(result["positive_pairs"], 0)
        self.assertGreater(result["occluded_pairs"], 0)

    def test_checkpoint_recompute_does_not_capture_again(self):
        model, inputs, targets, domain, captured, prediction = self.run_model(ToyModel(checkpointed=True))
        self.assertEqual(len(captured.records), 2)
        output = auxiliary_loss(prediction, inputs, targets, domain, captured)
        output["loss"].backward()
        self.assertEqual(len(captured.records), 2)
        self.assertGreater(model.decoder.source.calls, 2)
        self.assertTrue(torch.isfinite(model.decoder.source.logits.grad).all())

    def test_real_attention_parameters_receive_gradients_without_gt_gradient(self):
        model, inputs, targets, domain, captured, prediction = self.run_model(RealAttentionModel())
        targets["depth"].requires_grad_(True)
        result = auxiliary_loss(prediction, inputs, targets, domain, captured)
        result["loss"].backward()
        source = model.decoder.source
        for parameter in (source.reliability.weight, source.null.weight, source.pair[0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(source.reliability.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(source.null.weight.grad.abs().sum()), 0.)
        self.assertIsNone(targets["depth"].grad)
        self.assertEqual(len(captured.records), 2)

    def test_label_not_transferred_across_depth_boundary(self):
        _, inputs, targets, _, captured, prediction = self.run_model(center=3.6, interval=.5)
        inputs["K"][:, :, 0, 0] = 11.
        targets["depth"][:, 0, :, 6] = 2.
        supervision = visibility_targets(prediction, inputs, targets, "coarse", captured.records[0], 8)
        self.assertEqual(int(supervision["labels"][0, 0, 1, 1]), -1)
        targets["depth"][:, 0, :, 6] = 4.
        clear = visibility_targets(prediction, inputs, targets, "coarse", captured.records[0], 8)
        self.assertEqual(int(clear["labels"][0, 0, 1, 1]), 1)

    def test_capture_must_close_before_loss(self):
        inputs, targets, domain = scene()
        model = ToyModel()
        with capture_attention(model) as captured:
            prediction = model(inputs)
            with self.assertRaisesRegex(RuntimeError, "退出"):
                auxiliary_loss(prediction, inputs, targets, domain, captured)

    def test_bce_is_explicit_only(self):
        _, inputs, targets, domain, captured, prediction = self.run_model()
        result = auxiliary_loss(prediction, inputs, targets, domain, captured, mode="bce")
        self.assertAlmostEqual(float(result["loss"].detach()), math.log(2), places=6)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
