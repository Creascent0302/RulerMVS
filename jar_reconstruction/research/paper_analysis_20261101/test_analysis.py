"""配对统计的数学边界测试；不加载模型或使用GPU。"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.paper_analysis_20261101.analyze import paired_bootstrap, scene_cluster


def record(sequence, index, own_sum, reference_sum, pixels=10, own_coverage=None):
    def values(error, count):
        return {"depth": {"full": {"gt_pixels": pixels, "predicted_pixels": count, "abs_sum": error}}}
    return sequence, index, {"methods": {
        "own": values(own_sum, pixels if own_coverage is None else own_coverage),
        "other": values(reference_sum, pixels)}}


class PairedStatisticsTests(unittest.TestCase):
    def test_weather_variants_are_not_independent_scenes(self):
        self.assertEqual(scene_cluster("VKITTI2", "VKITTI2/Scene18/rain"),
                         scene_cluster("VKITTI2", "VKITTI2/Scene18/clone"))
        self.assertEqual(scene_cluster("Hypersim_subset", "hypersim_ai_005_005_cam_00"),
                         scene_cluster("Hypersim_subset", "hypersim_ai_005_005_cam_01"))

    def test_cluster_bootstrap_known_ratio(self):
        records = [record("scene_a", 0, 10, 20), record("scene_a", 1, 20, 40),
                   record("scene_b", 2, 100, 200, pixels=200)]
        result, rows = paired_bootstrap(records, "own", "other", 100, 1)
        self.assertEqual(result["sequences"], 2)
        self.assertEqual(result["frames"], 3)
        self.assertEqual(result["relative_reduction_percent"], 50.)
        self.assertEqual(result["ci_low"], 50.)
        self.assertEqual(result["ci_high"], 50.)
        self.assertAlmostEqual(result["own_mae"], 130 / 220)
        self.assertEqual(len(rows), 3)

    def test_one_sequence_has_no_confidence_interval(self):
        result, _ = paired_bootstrap([record("one_scene", 0, 10, 20)], "own", "other", 100, 1)
        self.assertIsNone(result["ci_low"])
        self.assertEqual(result["bootstrap_repetitions"], 0)
        self.assertEqual(result["status"], "descriptive_one_sequence")

    def test_partial_coverage_not_claimed_paired_same_pixels(self):
        result, _ = paired_bootstrap([record("one_scene", 0, 10, 20, own_coverage=5)],
                                     "own", "other", 100, 1)
        self.assertEqual(result["status"], "skipped_incomplete_prediction_coverage")


if __name__ == "__main__":
    unittest.main()
