"""用解析平面验证深度类型对照，覆盖零速度、像素采样及不同传感器中心。"""

import unittest

import numpy as np

from jar.carla_depth_audit import (
    bilinear_sample,
    comparison,
    matched_samples,
    radar_to_camera,
    recover_radar_points,
)


class DepthAuditTests(unittest.TestCase):
    def test_signed_direction_and_zero_direction(self):
        records = np.array([[2, 1, -1, 8], [-2, -1, 1, 8], [0, 0, 0, 8], [2, 1, 1, 101]])
        points, valid = recover_radar_points(records)
        np.testing.assert_allclose(points[0], points[1])
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), [8, 8])
        np.testing.assert_array_equal(valid, [True, True, False, False])

    def test_camera_radar_translation_and_rotation(self):
        radar_pose = {"x": 2, "y": 0, "z": 1}
        camera_pose = {"x": 2, "y": 0, "z": 2.5}
        point = radar_to_camera(np.array([[10, 2, -1]]), radar_pose, camera_pose)
        np.testing.assert_allclose(point, [[2, 2.5, 10]])
        # 雷达向右旋转 90 度，局部 x 变为 CARLA rig 的 y。
        radar_pose["yaw"] = 90
        point = radar_to_camera(np.array([[10, 0, 0]]), radar_pose, camera_pose)
        np.testing.assert_allclose(point, [[10, 1.5, 0]], atol=1e-12)

    def test_bilinear_center_convention(self):
        y, x = np.mgrid[:8, :10]
        image = 3 * x + 2 * y
        uv = np.array([[2.5, 3.5], [4, 2]])
        np.testing.assert_allclose(bilinear_sample(image, uv), [14.5, 16])

    def test_analytical_plane_distinguishes_depth_types(self):
        height, width, focal = 120, 160, 70
        intr = {"fx": focal, "fy": focal, "cx": 80, "cy": 60}
        u, v = np.meshgrid(np.arange(12, 148, 8), np.arange(12, 108, 8))
        points = np.stack(
            [
                (u.ravel() - 80) * 10 / focal,
                (v.ravel() - 60) * 10 / focal,
                np.full(u.size, 10),
            ],
            axis=1,
        )
        optical = np.full((height, width), 10.0)
        samples = matched_samples(optical, points, intr)
        metrics = comparison(samples)["hypotheses"]
        self.assertEqual(metrics["optical_z"]["median_relative_error"], 0)
        self.assertGreater(metrics["ray_range"]["median_relative_error"], 0.1)
        y, x = np.mgrid[:height, :width]
        ranges = 10 * np.sqrt(1 + ((x - 80) / focal) ** 2 + ((y - 60) / focal) ** 2)
        samples = matched_samples(ranges, points, intr)
        self.assertGreater(len(samples), 0)
        metrics = comparison(samples)["hypotheses"]
        self.assertLess(metrics["ray_range"]["median_relative_error"], 1e-12)
        self.assertGreater(metrics["optical_z"]["median_relative_error"], 0)

    def test_invalid_and_occlusion_edges_are_excluded(self):
        depth = np.full((40, 40), 10.0)
        depth[:, 20:] = 20
        intr = {"fx": 10, "fy": 10, "cx": 20, "cy": 20}
        points = np.array([[0, 0, 10], [0, 0, -10], [100, 0, 10], [-10, 0, 10]])
        samples = matched_samples(depth, points, intr)
        np.testing.assert_allclose(samples, [[10, 10, np.sqrt(200)]])

    def test_empty_returns_have_explicit_empty_statistics(self):
        points, valid = recover_radar_points(np.empty((0, 4)))
        self.assertEqual(points.shape, (0, 3))
        self.assertEqual(valid.size, 0)
        self.assertEqual(comparison(np.empty((0, 3))), {"count": 0, "hypotheses": {}})


if __name__ == "__main__":
    unittest.main()
