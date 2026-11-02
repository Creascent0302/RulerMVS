"""仅验证损失定义与梯度方向，不代表真实模型精度已经提升。"""

import math
import unittest

import torch

from .objectives import balanced_depth_error, excess_error, fine_actual_regret


def candidate_example(scale=1.):
    logits = torch.zeros(1, 2, 1, 1, requires_grad=True)
    sampling = torch.tensor([8., 11.]).reshape(1, 2, 1, 1).requires_grad_()
    actual = torch.tensor([9.8, 12.]).reshape(1, 2, 1, 1).requires_grad_()
    truth = torch.full((1, 1, 4, 4), 9.8 * scale, requires_grad=True)
    inputs = {"image_mask": torch.ones_like(truth, dtype=torch.bool),
              "view_mask": torch.ones(1, 1, dtype=torch.bool)}
    targets = {"depth": truth, "depth_valid": torch.ones_like(truth, dtype=torch.bool)}
    prediction = {"unit_scale": scale, "fine": {
        "sampling_depth": sampling, "candidate_depth": actual,
        "probability": logits.softmax(1), "valid": torch.ones_like(actual, dtype=torch.bool),
        "ref_valid": torch.ones(1, 1, 1, 1, dtype=torch.bool),
    }}
    return prediction, inputs, targets, logits


class ObjectiveTest(unittest.TestCase):
    def test_actual_candidate_reverses_wrong_sampling_score_gradient(self):
        prediction, inputs, targets, logits = candidate_example()
        result = fine_actual_regret(prediction, inputs, targets, {})
        old_gradient = torch.autograd.grad(result["old_risk"], logits, retain_graph=True)[0]
        new_gradient = torch.autograd.grad(result["new_risk"], logits, retain_graph=True)[0]
        regret_gradient = torch.autograd.grad(result["regret"], logits)[0]
        # 旧风险偏好采样11；实际9.8已精确命中，正确梯度应提高第一个logit。
        self.assertGreater(float(old_gradient[0, 0]), 0.)
        self.assertLess(float(new_gradient[0, 0]), 0.)
        torch.testing.assert_close(new_gradient, regret_gradient)
        self.assertEqual(int(result["valid_pixels"]), 1)

    def test_risk_does_not_directly_update_depth_or_targets(self):
        prediction, inputs, targets, _ = candidate_example()
        result = fine_actual_regret(prediction, inputs, targets, {})
        (result["old_risk"] + result["new_risk"] + result["regret"]).backward()
        self.assertIsNone(prediction["fine"]["sampling_depth"].grad)
        self.assertIsNone(prediction["fine"]["candidate_depth"].grad)
        self.assertIsNone(targets["depth"].grad)

    def test_invalid_candidates_and_empty_ray(self):
        prediction, inputs, targets, logits = candidate_example()
        prediction["fine"]["candidate_depth"] = torch.tensor([9.8, float("nan")]).reshape(1, 2, 1, 1)
        result = fine_actual_regret(prediction, inputs, targets, {})
        self.assertAlmostEqual(float(result["new_risk"].detach()), 0., places=6)
        self.assertEqual(int(result["valid_pixels"]), 1)
        prediction["fine"]["valid"].zero_()
        result = fine_actual_regret(prediction, inputs, targets, {})
        self.assertTrue(all(torch.isfinite(value) for value in result.values()))
        self.assertEqual(int(result["valid_pixels"]), 0)
        result["regret"].backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_mask_eval_range_and_scene_unit_normalization(self):
        ordinary = candidate_example()
        scaled = candidate_example(scale=17.3)
        a = fine_actual_regret(*ordinary[:3], {})
        b = fine_actual_regret(*scaled[:3], {})
        for key in a:
            torch.testing.assert_close(a[key], b[key])
        excluded = fine_actual_regret(*scaled[:3], {"eval_range": [1., 10.]})
        self.assertEqual(int(excluded["valid_pixels"]), 0)
        prediction, inputs, targets, _ = ordinary
        inputs["view_mask"].zero_()
        self.assertEqual(int(fine_actual_regret(prediction, inputs, targets, {})["valid_pixels"]), 0)

    def test_distance_bands_are_balanced_not_pixel_weighted(self):
        truth = torch.tensor([1., 1., 1., 20.]).reshape(1, 1, 1, 4)
        prediction = torch.tensor([2., 2., 2., 20.]).reshape_as(truth)
        valid = torch.ones_like(truth, dtype=torch.bool)
        value = balanced_depth_error(prediction, truth, valid, (10., 40.))
        self.assertAlmostEqual(float(value.detach()), (math.log(2.) + .2) / 2, places=6)
        # 同时转换数值与边界，且不触及相对误差分母下限，保持损失不变。
        scaled = balanced_depth_error(prediction * 3, truth * 3, valid, (30., 120.))
        torch.testing.assert_close(value, scaled)

    def test_all_empty_band_is_finite_differentiable_zero(self):
        prediction = torch.tensor([2., 5.], requires_grad=True)
        truth = torch.tensor([float("nan"), 0.])
        value = balanced_depth_error(prediction, truth, torch.zeros(2, dtype=torch.bool), (1., 3.))
        self.assertEqual(float(value.detach()), 0.)
        value.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_excess_error_zero_when_student_is_better(self):
        estimate = torch.tensor([10., 10.5], requires_grad=True)
        teacher = torch.tensor([11., 12.], requires_grad=True)
        truth = torch.tensor([10., 10.], requires_grad=True)
        value = excess_error(estimate, teacher, truth, torch.ones(2, dtype=torch.bool))
        self.assertEqual(float(value.detach()), 0.)
        value.backward()
        torch.testing.assert_close(estimate.grad, torch.zeros_like(estimate))
        self.assertIsNone(teacher.grad)
        self.assertIsNone(truth.grad)

    def test_excess_error_trains_towards_truth_without_teacher_gradient(self):
        estimate = torch.tensor([8.], requires_grad=True)
        teacher = torch.tensor([11.], requires_grad=True)
        truth = torch.tensor([10.], requires_grad=True)
        value = excess_error(estimate, teacher, truth, torch.ones(1, dtype=torch.bool))
        self.assertAlmostEqual(float(value.detach()), math.log(10 / 8) - math.log(11 / 10), places=6)
        value.backward()
        self.assertLess(float(estimate.grad), 0.)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(truth.grad)
        empty = excess_error(estimate, torch.tensor([float("nan")]), truth, torch.ones(1, dtype=torch.bool))
        self.assertEqual(float(empty.detach()), 0.)


if __name__ == "__main__":
    unittest.main()
