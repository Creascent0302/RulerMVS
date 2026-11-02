import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from jar.data import build_dataset, collate_frames
from jar.data_transforms import resize_frame
from tests.test_carla_capture import config, fixture, prepare


class DepthResizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def item(self, depth):
        depth = depth[None].float()
        height, width = depth.shape[-2:]
        return {
            "inputs": {
                "rgb": torch.linspace(0, 1, 3 * height * width).reshape(1, 3, height, width),
                "K": torch.tensor(
                    [[[60.0, 0.0, width / 2], [0.0, 60.0, height / 2], [0.0, 0.0, 1.0]]]
                ),
                "T_cam_from_rig": torch.eye(4)[None],
                "view_mask": torch.ones(1, dtype=torch.bool),
                "image_mask": torch.ones(1, height, width, dtype=torch.bool),
            },
            "targets": {"depth": depth, "depth_valid": torch.isfinite(depth) & (depth > 0)},
            "metadata": {},
        }

    def test_slanted_plane_uses_exact_resized_pixel_centers(self):
        y, x = torch.meshgrid(torch.arange(32), torch.arange(48), indexing="ij")
        depth = 10 / (1 + 0.003 * x + 0.002 * y)
        sample = self.item(depth)
        output = resize_frame(sample, [8, 12], depth_resize_mode="edge_aware_inverse")
        yy, xx = torch.meshgrid(torch.arange(8), torch.arange(12), indexing="ij")
        expected = 10 / (1 + 0.003 * (4 * xx + 1.5) + 0.002 * (4 * yy + 1.5))
        torch.testing.assert_close(output["targets"]["depth"][0], expected)
        self.assertTrue(output["targets"]["depth_resize_interpolated"].all())
        nearest = resize_frame(sample, [8, 12])["targets"]["depth"][0]
        self.assertGreater(float((nearest - expected).abs().mean()), 0.01)

    def test_thin_object_keeps_nearest_hit_and_marks_boundary(self):
        depth = torch.full((8, 12), 40.0)
        depth[:, 2] = 8.0
        sample = self.item(depth)
        old = resize_frame(sample, [2, 3])
        output = resize_frame(sample, [2, 3], depth_resize_mode="edge_aware_inverse")
        torch.testing.assert_close(output["targets"]["depth"], old["targets"]["depth"])
        self.assertEqual(set(output["targets"]["depth"].flatten().tolist()), {8.0, 40.0})
        self.assertTrue(output["targets"]["depth_resize_boundary"][..., 0].all())
        self.assertFalse(output["targets"]["depth_resize_interpolated"][..., 0].any())

    def test_invalid_neighbors_cannot_fill_missing_labels(self):
        depth = torch.full((8, 12), 8.0)
        depth[1, 1] = 0.0  # 邻域缺失，但最近邻本身仍有效。
        depth[2, 6] = 0.0  # 最近邻也缺失，不能由其他三个像素补出标签。
        sample = self.item(depth)
        old = resize_frame(sample, [2, 3])
        output = resize_frame(sample, [2, 3], depth_resize_mode="edge_aware_inverse")
        torch.testing.assert_close(output["targets"]["depth_valid"], old["targets"]["depth_valid"])
        self.assertFalse(output["targets"]["depth_resize_interpolated"][0, 0, :2].any())
        self.assertEqual(float(output["targets"]["depth"][0, 0, 1]), 0.0)
        self.assertTrue(torch.isfinite(output["targets"]["depth"]).all())

    def test_labels_do_not_change_rgb_calibration_or_other_labels(self):
        sample = self.item(torch.full((8, 12), 8.0))
        sample["targets"]["semantic"] = torch.arange(96).reshape(1, 8, 12)
        old = resize_frame(sample, [2, 3])
        output = resize_frame(sample, [2, 3], depth_resize_mode="edge_aware_inverse")
        sample["targets"]["depth"].mul_(2)
        changed = resize_frame(sample, [2, 3], depth_resize_mode="edge_aware_inverse")
        for key in old["inputs"]:
            torch.testing.assert_close(output["inputs"][key], old["inputs"][key], rtol=0, atol=0)
            torch.testing.assert_close(
                output["inputs"][key], changed["inputs"][key], rtol=0, atol=0
            )
        torch.testing.assert_close(output["targets"]["semantic"], old["targets"]["semantic"])
        expected = F.interpolate(sample["targets"]["depth"][:, None], (2, 3), mode="nearest-exact")
        torch.testing.assert_close(resize_frame(sample, [2, 3])["targets"]["depth"], expected[:, 0])

    def test_dataset_passes_config_and_inference_never_reads_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, "a")
            fixture(root, "b")
            manifest = root / "index"
            prepare(root, manifest, "a", "b", "optical_z")
            cfg = config(root, manifest)
            cfg["data"]["depth_resize_mode"] = "edge_aware_inverse"
            supervised = build_dataset(cfg)[0]
            inferred = build_dataset(cfg, load_targets=False)[0]
            self.assertIn("depth_resize_interpolated", supervised["targets"])
            self.assertEqual(inferred["targets"], {})
            for key, value in supervised["inputs"].items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, inferred["inputs"][key], rtol=0, atol=0)

    def test_generic_collation_pads_diagnostic_masks_as_bool_tensors(self):
        a = resize_frame(
            self.item(torch.full((8, 12), 8.0)), [2, 3], depth_resize_mode="edge_aware_inverse"
        )
        b = resize_frame(self.item(torch.full((8, 12), 8.0)), [2, 3])
        b["inputs"] = {
            key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in b["inputs"].items()
        }
        b["targets"] = {}
        batch = collate_frames([a, b])
        for key in ("depth_resize_interpolated", "depth_resize_boundary"):
            mask = batch["targets"][key]
            self.assertEqual(mask.shape, (2, 2, 2, 3))
            self.assertEqual(mask.dtype, torch.bool)
            torch.testing.assert_close(mask[0, 0], a["targets"][key][0])
            self.assertFalse(mask[0, 1].any())
            self.assertFalse(mask[1].any())


if __name__ == "__main__":
    unittest.main()
