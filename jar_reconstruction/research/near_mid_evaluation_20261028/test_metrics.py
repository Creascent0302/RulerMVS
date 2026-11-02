"""评测定义检查：单位、分母、深度区间和最近邻查询参考集。"""

import unittest

import torch

from .metrics import VisibleSurface, depth_metrics, depth_sums, regions, surface_metrics


class MetricsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.depth = torch.full((1, 1, 8, 12), 5., dtype=torch.float64)
        self.valid = torch.ones_like(self.depth, dtype=torch.bool)
        self.domain = {"geometry_units": "meters", "depth_scale": 1.}

    def test_exact_plane_and_full_reference(self):
        surface = VisibleSurface(self.depth, self.valid, torch.eye(3), self.domain, 32)
        sums = surface.measure(self.depth, self.valid)
        result = surface_metrics(sums, surface.thresholds)
        self.assertEqual(result["Chamfer_L1"], 0.)
        self.assertEqual(result["Fscore_0.05"], 1.)
        self.assertEqual(sums["gt_queries"], 32)
        self.assertEqual(sums["gt_reference_points"], 96)

    def test_missing_prediction_stays_in_denominator(self):
        valid = self.valid.clone()
        valid[..., 0, 0] = False
        result = depth_metrics(depth_sums(self.depth, valid, self.depth, self.valid))
        self.assertIsNone(result["full_gt_MAE"])
        self.assertEqual(result["delta_1"], 95 / 96)
        self.assertEqual(result["coverage"], 95 / 96)

    def test_depth_statistics(self):
        gt = torch.tensor([1., 2.]).reshape(1, 1, 1, 2)
        pred = torch.tensor([2., 5.]).reshape_as(gt)
        valid = torch.ones_like(gt, dtype=torch.bool)
        result = depth_metrics(depth_sums(pred, valid, gt, valid))
        self.assertEqual(result["MAE"], 2.)
        self.assertAlmostEqual(result["RMSE"], 5 ** .5)
        self.assertEqual(result["AbsRel"], 1.25)
        self.assertEqual(result["signed_bias"], 2.)

    def test_distance_units_and_boundaries(self):
        gt = torch.tensor([9., 10., 39., 40.]).reshape(1, 1, 1, 4)
        mask = torch.ones_like(gt, dtype=torch.bool)
        masks = regions(gt, mask, self.domain)
        self.assertEqual([int(masks[k].sum()) for k in ("near", "mid", "far")], [1, 2, 1])
        scaled = regions(gt * 2, mask, {"geometry_units": "scene_unit", "depth_scale": 4.})
        for key in ("near", "mid", "far"):
            self.assertTrue(torch.equal(masks[key], scaled[key]))


if __name__ == "__main__":
    unittest.main()
