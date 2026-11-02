import unittest
import torch
from jar import build_model, load_config
from jar.synthetic import make_synthetic_batch
from jar.common import make_scene
from jar.fusion import spatial_hash_fuse, enforce_roi


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_hash_fusion_conservative_parallel_surfaces_unknown(self):
        points = torch.tensor(
            [[1.0, 0.0, 0.0], [1.001, 0.0, 0.0], [1.02, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )
        normals = torch.tensor([[-1.0, 0.0, 0.0]]).expand(4, 3).clone()
        features = torch.randn(4, 8, requires_grad=True)
        logits = torch.tensor([[0.0, 5.0, 0.0]]).expand(4, 3).clone()
        ev = {
            "surface": torch.tensor([True, True, True, False]),
            "free": torch.zeros(4, dtype=torch.bool),
            "known": torch.tensor([True, True, True, False]),
            "support": torch.tensor([3, 3, 3, 0]),
            "confidence": torch.ones(4),
        }
        scene = make_scene(points, normals, torch.ones_like(points) * 0.05, features, logits, ev)
        fused = spatial_hash_fuse(scene, 0.1)
        self.assertTrue((fused["state"] == 2).any())
        self.assertLessEqual(int(fused["support"].max()), 3)
        self.assertNotEqual(int(fused["fusion_assignment"][0]), int(fused["fusion_assignment"][2]))
        self.assertNotEqual(int(fused["fusion_assignment"][0]), int(fused["fusion_assignment"][3]))
        fused["features"].sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        three = {key: value[:3] for key, value in scene.items()}
        merged = spatial_hash_fuse(three, 0.1)
        self.assertEqual(len(merged["points"]), 2)
        self.assertEqual(int(merged["fusion_assignment"][0]), int(merged["fusion_assignment"][1]))
        self.assertTrue(
            torch.allclose(
                merged["points"][merged["fusion_assignment"][0]], torch.tensor([1.0005, 0.0, 0.0])
            )
        )

    def test_confirmed_surface_fusion_mixed_precision_features(self):
        points = torch.tensor([[1.0, 0.0, 0.0], [1.001, 0.0, 0.0]])
        features = torch.tensor([[1.0, 3.0], [3.0, 5.0]], dtype=torch.bfloat16, requires_grad=True)
        evidence = dict(
            surface=torch.ones(2, dtype=torch.bool),
            free=torch.zeros(2, dtype=torch.bool),
            known=torch.ones(2, dtype=torch.bool),
            support=torch.tensor([2, 2]),
            confidence=torch.ones(2, dtype=torch.float32),
        )
        scene = make_scene(
            points,
            torch.tensor([[-1.0, 0.0, 0.0]]).expand(2, 3),
            torch.ones(2, 3) * 0.05,
            features,
            torch.tensor([[0.0, 5.0, 0.0]]).expand(2, 3),
            evidence,
        )
        fused = spatial_hash_fuse(scene, 0.1)
        self.assertEqual(len(fused["points"]), 1)
        self.assertEqual(fused["features"].dtype, torch.bfloat16)
        torch.testing.assert_close(fused["features"].float(), torch.tensor([[2.0, 4.0]]))
        fused["features"].float().sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_roi_gates_public_scene_without_removing_unknown_or_supervision(self):
        points = torch.tensor([[3.0, 0.0, 0.0], [11.0, 0.0, 0.0], [3.0, 5.0, 0.0]])
        scene = {
            "points": points,
            "valid": torch.ones(3, dtype=torch.bool),
            "state": torch.ones(3, dtype=torch.long),
            "confidence": torch.ones(3),
        }
        scene["query_stage"] = dict(scene)
        restricted = enforce_roi(scene, [[2.0, 10.0], [-4.0, 4.0], [-2.0, 2.0]])
        self.assertEqual(restricted["state"].tolist(), [1, 2, 2])
        self.assertEqual(restricted["valid"].tolist(), [True, False, False])
        self.assertEqual(len(restricted["points"]), 3)
        self.assertTrue(restricted["query_stage"]["valid"].all())
        cfg = load_config("configs/unit_tiny.json")
        with torch.no_grad():
            actual = build_model(cfg).eval()(make_synthetic_batch(N=3, H=32, W=48)["inputs"])[
                "scenes"
            ][0]
        self.assertFalse(actual["valid"][~actual["in_roi"]].any())
        self.assertTrue((actual["state"][~actual["in_roi"]] == 2).all())
