"""几何参考算法的 CPU 验收；训练效果由独立基准报告而非这些断言证明。"""

import unittest
import torch

from jar.baselines import ClassicalMVS
from jar.synthetic import synthetic_rig, make_synthetic_batch
from jar.geometry import pixel_grid, unproject


def plane_batch(n=3, h=32, w=48, depth=8.0):
    """同一世界纹理平面在真实相机中的光线交点；不存在独立随机图像。"""
    K, T = synthetic_rig(n, (h, w))
    uv = pixel_grid(h, w)
    colors = []
    for k, t in zip(K, T):
        points, _ = unproject(uv, torch.full((h, w), depth), k, t)
        y, z = points[..., 1], points[..., 2]
        colors.append(
            torch.stack(
                [
                    0.5 + 0.22 * torch.sin(1.8 * y + 0.6 * z),
                    0.5 + 0.21 * torch.cos(1.1 * y - 2.2 * z),
                    0.5 + 0.20 * torch.sin(2.3 * z + 0.8 * y),
                ],
                0,
            )
        )
    return {
        "rgb": torch.stack(colors)[None],
        "K": K[None],
        "T_cam_from_rig": T[None],
        "view_mask": torch.ones(1, n, dtype=torch.bool),
    }


class ClassicalBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, **kwargs):
        return {
            "model": {"near": 3.0, "far": 20.0, "depth_bins": 48},
            "baseline": dict(stride=4, patch_radius=2, **kwargs),
        }

    def test_planar_depth_and_variable_views(self):
        for method in ("zncc", "census"):
            for n in (2, 3, 4):
                with self.subTest(method=method, cameras=n):
                    out = ClassicalMVS(self.config(), method)(plane_batch(n))
                    valid = out["dense_valid"]
                    errors = (out["dense_depth"][valid] - 8.0).abs()
                    self.assertGreater(int(valid.sum()), 20)
                    self.assertLess(float(errors.median()), 0.30)
                    self.assertLess(float(errors.mean()), 0.75)
                    self.assertEqual(out["dense_depth"].shape, (1, n, 8, 12))
                    self.assertGreaterEqual(int(out["coarse"]["support"][valid].min()), 2)
                    for v in out["scenes"][0].values():
                        if isinstance(v, torch.Tensor):
                            self.assertTrue(torch.isfinite(v).all())
                    self.assertFalse((out["scenes"][0]["state"] == 0).any())

    def test_no_texture_pure_rotation_and_missing_views_stay_unknown(self):
        for method in ("zncc", "census"):
            model = ClassicalMVS(self.config(), method)
            for kind in ("textureless", "rotation", "missing"):
                with self.subTest(method=method, condition=kind):
                    if kind == "rotation":
                        inputs = make_synthetic_batch(N=3, H=32, W=48, pure_rotation=True)["inputs"]
                    else:
                        inputs = plane_batch()
                        if kind == "textureless":
                            inputs["rgb"].fill_(0.5)
                        else:
                            inputs["view_mask"][:, 1:] = False
                    out = model(inputs)
                    self.assertFalse(out["dense_valid"].any())
                    self.assertTrue((out["scenes"][0]["state"] == 2).all())
                    self.assertTrue(torch.isfinite(out["coarse"]["logits"]).all())

    def test_mask_chunking_and_targets_isolation(self):
        batch = make_synthetic_batch(N=2, H=32, W=48)
        inputs = batch["inputs"]
        inputs["image_mask"][..., :8, :] = False
        for method in ("zncc", "census"):
            model = ClassicalMVS(self.config(depth_chunk=7, query_chunk=19), method).eval()
            out = model(inputs)
            batch["targets"]["depth"].fill_(1e5)
            again = model(inputs)
            self.assertTrue(torch.equal(out["dense_depth"], again["dense_depth"]))
            self.assertTrue(torch.equal(out["dense_valid"], again["dense_valid"]))
            self.assertFalse(out["dense_valid"][..., :3, :].any())
            other = ClassicalMVS(self.config(depth_chunk=11, query_chunk=128), method)(inputs)
            self.assertTrue(torch.allclose(out["dense_depth"], other["dense_depth"], atol=2e-4))
            self.assertTrue(torch.equal(out["dense_valid"], other["dense_valid"]))

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            ClassicalMVS({"model": {"near": 0}})
        with self.assertRaises(ValueError):
            ClassicalMVS(self.config(refinement_bins=2))
        with self.assertRaises(ValueError):
            ClassicalMVS(self.config(), "not-a-method")

    def test_output_budget_is_independent_of_validity(self):
        cfg = self.config()
        cfg["model"]["max_queries"] = 37
        model = ClassicalMVS(cfg)
        inputs = plane_batch(4)
        outputs = model(inputs)
        self.assertEqual(len(outputs["scenes"][0]["points"]), 37)
        self.assertEqual(outputs["scenes"][0]["aux"]["source_candidate_count"], 384)
        self.assertEqual(outputs["dense_depth"].shape, (1, 4, 8, 12))
        inputs["rgb"].fill_(0.5)
        empty = model(inputs)
        self.assertEqual(len(empty["scenes"][0]["points"]), 37)
        self.assertTrue((empty["scenes"][0]["state"] == 2).all())


if __name__ == "__main__":
    unittest.main()
