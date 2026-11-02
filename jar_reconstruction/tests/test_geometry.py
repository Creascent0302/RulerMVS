import unittest
import torch
from jar.geometry import (
    project,
    unproject,
    pixel_grid,
    sample_maps,
    resize_crop_intrinsics,
    safe_masked_softmax,
    validate_inputs,
)
from jar.synthetic import make_synthetic_batch, synthetic_rig
from jar import build_model, load_config


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_roundtrip_jacobian_and_optical_axis_depth(self):
        K, T = synthetic_rig(3, (48, 64), pure_rotation=True)
        K, T = K.double(), T.double()
        # Include translation and non-axis-aligned rotations in the derivative.
        T[:, :3, 3] = torch.tensor(
            [[0.1, 0.2, 0.3], [-0.2, 0.1, 0], [0.3, -0.1, 0.2]], dtype=torch.double
        )
        uv = torch.tensor([[25.0, 20.0], [33.0, 25.0], [40.0, 17.0]], dtype=torch.double)
        z = torch.tensor([4.0, 7.0, 11.0], dtype=torch.double)
        X, ray = unproject(uv, z, K[1], T[1])
        pixels, depth, valid, J = project(X, K, T, (48, 64))
        self.assertTrue(valid[1].all())
        torch.testing.assert_close(pixels[1], uv, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(depth[1], z, atol=1e-10, rtol=1e-10)
        eps = 1e-5
        for axis in range(3):
            offset = torch.zeros_like(X)
            offset[:, axis] = eps
            positive = project(X + offset, K, T, (48, 64))[0]
            negative = project(X - offset, K, T, (48, 64))[0]
            torch.testing.assert_close(
                (positive - negative) / (2 * eps), J[..., axis], atol=1e-7, rtol=1e-7
            )
        torch.testing.assert_close(
            (J[1] * ray[:, None]).sum(-1), torch.zeros_like(uv), atol=1e-10, rtol=0
        )

    def test_resize_crop_and_feature_center_coordinates(self):
        K, T = synthetic_rig(1, (64, 96))
        point = torch.tensor([[6.0, 0.3, 0.2]])
        uv = project(point, K, T, (64, 96))[0]
        transformed = resize_crop_intrinsics(K, scale_xy=(0.75, 0.5), crop_xy=(3, 2))
        changed = project(point, transformed, T, (30, 69))[0]
        expected = (uv + 0.5) * torch.tensor([0.75, 0.5]) - 0.5 - torch.tensor([3.0, 2.0])
        torch.testing.assert_close(changed, expected)
        # A stride convolution's feature j is centered on original pixel 8*j.
        grid = pixel_grid(4, 5, stride=8)
        values = grid.permute(2, 0, 1).unsqueeze(0)
        sampled = sample_maps(values, grid.reshape(1, -1, 2), stride=8)
        torch.testing.assert_close(sampled[0], grid.reshape(-1, 2))
        half = sample_maps(values, torch.tensor([[[12.0, 12.0]]]), stride=8)
        torch.testing.assert_close(half, torch.tensor([[[12.0, 12.0]]]))

    def test_masks_all_invalid_and_calibration_validation(self):
        K, T = synthetic_rig(2, (32, 48))
        X = torch.tensor([[4.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [4.0, 100.0, 0.0]])
        uv, z, valid, J = project(X, K, T, (32, 48), torch.tensor([True, False]))
        self.assertEqual(valid.tolist(), [[True, False, False], [False, False, False]])
        mask = torch.zeros(2, 32, 48, dtype=torch.bool)
        self.assertFalse(project(X, K, T, (32, 48), image_mask=mask)[2].any())
        self.assertTrue(
            torch.isfinite(uv).all() and torch.isfinite(z).all() and torch.isfinite(J).all()
        )
        a = torch.randn(2, 5, requires_grad=True)
        result = safe_masked_softmax(a, torch.zeros_like(a, dtype=torch.bool))
        self.assertEqual(float(result.sum()), 0.0)
        result.sum().backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        batch = make_synthetic_batch(N=2, H=32, W=48)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            validate_inputs(batch["inputs"])
        batch["inputs"]["T_cam_from_rig"][0, 0, 0, :3] *= -1
        with self.assertRaisesRegex(ValueError, "right-handed"):
            validate_inputs(batch["inputs"])

    def test_coarse_pure_rotation_and_constant_texture_reject_match(self):
        cfg = load_config("configs/unit_tiny.json")
        for mode in ["rotation", "constant", "missing"]:
            with self.subTest(mode=mode):
                batch = make_synthetic_batch(N=3, H=32, W=48)
                inp = batch["inputs"]
                if mode == "rotation":
                    K, T = synthetic_rig(3, (32, 48), pure_rotation=True)
                    inp["K"] = K[None]
                    inp["T_cam_from_rig"] = T[None]
                elif mode == "constant":
                    inp["rgb"].fill_(0.5)
                else:
                    inp["view_mask"].fill_(False)
                model = build_model(cfg).eval()
                with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                    coarse = model.coarse(inp, model.encoder(inp))
                self.assertTrue(torch.isfinite(coarse["logits"]).all())
                self.assertFalse(coarse["support"].any())
                self.assertFalse(coarse["confidence"].any())


if __name__ == "__main__":
    unittest.main()
