"""诊断统计的解析算例，不运行模型或读取数据。"""

import unittest

import torch

from research.ruler_flow_20261030.diagnostics import aggregate, analyze, noise_statistics


def prediction(reduced=False):
    samples = torch.tensor([1.5, 2., 2.5, 3.5, 4., 4.5, 5.5, 6., 6.5, 7.5, 8., 8.5])[None, :, None, None].expand(1, -1, 2, 2)
    lower, upper = samples - .25, samples + .25
    if reduced:
        lower = lower.unflatten(1, (4, 3)).amin(2)
        upper = upper.unflatten(1, (4, 3)).amax(2)
        samples = samples[:, 1::3]
    winner = torch.ones(1, 1, 2, 2, dtype=torch.long)
    depth = samples.gather(1, winner)
    fine = {"sampling_depth": samples, "candidate_depth": samples, "lower": lower, "upper": upper,
            "valid": torch.ones_like(samples, dtype=torch.bool), "selected_index": winner, "depth": depth}
    if reduced:
        fine["anchor_depth"] = samples
    return {"fine": fine, "unit_scale": 1., "depth": depth.repeat_interleave(4, -2).repeat_interleave(4, -1)}


class DiagnosticsTest(unittest.TestCase):
    def test_equal_interval_coverage_different_endpoint_oracle(self):
        truth = torch.tensor([[[2., 2.8], [4.6, 9.]]], dtype=torch.double)
        masks = {"full": torch.ones_like(truth, dtype=torch.bool)}
        domain = {"near": 1., "far": 10.}
        old, reference, _ = analyze(prediction(), domain, truth, masks)
        new, _, equality = analyze(prediction(True), domain, truth, masks, reference)
        self.assertEqual(equality["gt_coverage_mismatch"], 0)
        self.assertEqual(old["full"]["metrics"]["interval_gt_coverage"], .5)
        self.assertEqual(new["full"]["metrics"]["interval_gt_coverage"], .5)
        self.assertGreater(new["full"]["metrics"]["continuous_oracle_MAE"], old["full"]["metrics"]["continuous_oracle_MAE"])
        self.assertEqual(new["full"]["metrics"]["mean_abs_log_movement"], 0.)
        merged = aggregate([old, new])
        self.assertEqual(merged["full"]["gt_pixels"], 8)

    def test_noise_population_variance(self):
        truth = torch.ones(1, 2, 2, dtype=torch.double) * 4
        predictions = [truth - 1, truth, truth + 1]
        masks = {"full": torch.ones_like(truth, dtype=torch.bool)}
        values, differences = noise_statistics(predictions, truth, truth, masks)
        self.assertAlmostEqual(values["full"]["metrics"]["output_variance"], 2 / 3)
        self.assertAlmostEqual(values["full"]["metrics"]["mean_seed_MAE"], 2 / 3)
        self.assertEqual(differences["full"].numel(), 12)
        self.assertEqual(differences["full"].max().item(), 2.)


if __name__ == "__main__":
    unittest.main()
