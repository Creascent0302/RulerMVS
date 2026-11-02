"""薄适配的几何和缺失标签边界；完整性能不在此测试。"""

import importlib
import unittest
import sys
import tempfile
from pathlib import Path
import torch
from jar.synthetic import make_synthetic_batch
from official_methods.methods.niantic_common import native_intrinsics, depth_intrinsics


class NianticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_geometric_flip_preserves_ray(self):
        batch = make_synthetic_batch(N=4, H=64, W=96)
        inputs = batch["inputs"]
        T, K = inputs["T_cam_from_rig"], inputs["K"]
        reflected = T.clone()
        reflected[..., 0, :] *= -1
        K2 = K.clone()
        K2[..., 0, 2] = 95 - K2[..., 0, 2]
        point = torch.tensor([21.0, 17.0, 1.0])
        mirrored = torch.tensor([74.0, 17.0, 1.0])
        direction = torch.linalg.inv(T)[..., :3, :3] @ torch.linalg.solve(K, point)[..., None]
        direction2 = (
            torch.linalg.inv(reflected)[..., :3, :3] @ torch.linalg.solve(K2, mirrored)[..., None]
        )
        torch.testing.assert_close(direction, direction2)

    def test_original_checkpoint_options_and_resolution(self):
        from official_methods.methods.simplerecon import Adapter
        from official_methods.upstream.simplerecon import options

        model = Adapter({"model": {"image_height": 64, "image_width": 96, "pretrained": False}})
        original_module = options.Options.__module__
        previous = sys.modules.get("options")
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory) / "native.ckpt"
            try:
                sys.modules["options"] = options
                options.Options.__module__ = "options"
                torch.save(
                    {
                        "state_dict": model.native_model.state_dict(),
                        "hyper_parameters": {"opts": options.Options()},
                    },
                    path,
                )
            finally:
                options.Options.__module__ = original_module
                if previous is None:
                    del sys.modules["options"]
                else:
                    sys.modules["options"] = previous
            # 原官方对象采用全局 options.Options；集成器显式映射该命名空间。
            larger = Adapter(
                {"model": {"image_height": 96, "image_width": 128, "pretrained": False}}
            )
            larger.load_native_weights(path)
            self.assertTrue(larger.weight_loading_report["regenerated_pixel_grids"])
            torch.testing.assert_close(
                next(model.parameters()), next(larger.parameters()), rtol=0, atol=0
            )

    def test_sparse_and_missing_depth(self):
        for name in ("simplerecon", "mvsanywhere"):
            with self.subTest(method=name):
                module = importlib.import_module("official_methods.methods." + name)
                model = module.Adapter(
                    {
                        "model": {"image_height": 64, "image_width": 96, "pretrained": False},
                        "training": {"geometric_flip": False},
                    }
                ).train()
                batch = make_synthetic_batch(N=4, H=64, W=96)
                inputs, targets = batch["inputs"], batch["targets"]
                targets["depth_valid"].zero_()
                targets["depth_valid"][:, :, 25, 35] = True
                prepared = model.prepare_training_inputs(inputs, targets)
                output = model(prepared)
                loss = model.loss(output, prepared, targets)["total"]
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(
                    all(
                        torch.isfinite(p.grad).all()
                        for p in model.parameters()
                        if p.grad is not None
                    )
                )
                model.zero_grad(set_to_none=True)
                targets["depth_valid"].zero_()
                output = model(model.prepare_training_inputs(inputs, targets))
                loss = model.loss(output, inputs, targets)["total"]
                self.assertEqual(loss.item(), 0.0)
                loss.backward()
                self.assertTrue(
                    all(
                        torch.isfinite(p.grad).all()
                        for p in model.parameters()
                        if p.grad is not None
                    )
                )
                hole = {**inputs, "image_mask": inputs["image_mask"].clone()}
                hole["image_mask"][..., 1, 1] = False
                with self.assertRaises(ValueError):
                    model(hole)
                del model


if __name__ == "__main__":
    unittest.main()
