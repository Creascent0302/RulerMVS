"""检查可视化保留稀疏覆盖、未知和无标签含义，不用填洞美化结果。"""

import unittest

import torch

from tools.visualize_results import comparison_maps, select_frames, summary_rows


def example():
    inputs = {
        "rgb": torch.zeros(1, 2, 3, 8, 8),
        "K": torch.tensor([[4.0, 0, 4], [0, 4, 4], [0, 0, 1]]).repeat(1, 2, 1, 1),
        "T_cam_from_rig": torch.eye(4).repeat(1, 2, 1, 1),
        "view_mask": torch.tensor([[True, False]]),
        "image_mask": torch.ones(1, 2, 8, 8, dtype=torch.bool),
    }
    depth = torch.zeros(1, 2, 8, 8)
    depth[0, 0, 4, 4] = 2.5
    scene = {
        "points": torch.tensor([[0.0, 0, 2], [0, 0, 4], [0.5, 0, 2], [0.5, 0.5, 2]]),
        "valid": torch.tensor([True, True, False, True]),
        "state": torch.tensor([1, 1, 1, 2]),
    }
    return scene, {"inputs": inputs, "targets": {"depth": depth, "depth_valid": depth > 0}}


class VisualizationTests(unittest.TestCase):
    def test_z_buffer_keeps_one_pixel_and_excludes_unknown_and_invalid(self):
        scene, batch = example()
        camera, gt, valid, prediction, predicted, error = comparison_maps(
            scene, batch, [[-5, 5], [-5, 5], [0, 8]]
        )
        self.assertEqual(camera, 0)
        self.assertEqual(int(predicted.sum()), 1)
        self.assertEqual(float(prediction[4, 4]), 2)
        self.assertEqual(float(error[4, 4]), 0.5)
        self.assertEqual(int(torch.isfinite(error).sum()), 1)
        self.assertEqual(int(valid.sum()), 1)

    def test_absent_labels_are_not_zero_error_or_model_inputs(self):
        scene, batch = example()
        before = comparison_maps(scene, batch, [[-5, 5], [-5, 5], [0, 8]])
        batch["targets"] = {}
        after = comparison_maps(scene, batch, [[-5, 5], [-5, 5], [0, 8]])
        torch.testing.assert_close(before[3], after[3])
        self.assertFalse(after[2].any())
        self.assertTrue(torch.isnan(after[-1]).all())

    def test_selection_uses_input_groups_not_best_error(self):
        records = [
            {"sample": 0, "metadata": {"camera_completeness": "partial"}},
            {"sample": 1, "has_depth_labels": 1,
             "metadata": {"camera_completeness": "partial"}},
            {"sample": 2, "has_depth_labels": 1, "depth_mae_m": 100,
             "metadata": {"camera_completeness": "complete"}},
            {"sample": 3, "has_depth_labels": 1, "depth_mae_m": 0,
             "metadata": {"camera_completeness": "complete"}},
        ]
        self.assertEqual([r["sample"] for r in select_frames(records, 3)], [2, 1, 0])

    def test_summary_does_not_convert_missing_metrics_to_zero(self):
        summary = {
            "frame_count": 1,
            "metrics_frame_mean": {"unknown_fraction": 1},
            "depth_pixel_weighted": {"mae_m": None, "rmse_m": None, "coverage": None},
            "latency_ms_mean": 5,
            "peak_cuda_memory_bytes": 1024,
        }
        rows = summary_rows({"summary": summary, "samples": [{"metadata": {}}]})
        self.assertIsNone(rows[0]["深度MAE_米"])
        self.assertIsNone(rows[0]["深度覆盖率"])
        self.assertEqual(rows[0]["有深度标签帧数"], 0)


if __name__ == "__main__":
    unittest.main()
