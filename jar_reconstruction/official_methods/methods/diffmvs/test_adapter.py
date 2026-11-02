"""真实官方图的 CPU 合成验收；不评估随机权重精度。"""

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import unittest

import torch

from jar.data import collate_frames
from jar.synthetic import SyntheticRigDataset
from .adapter import Adapter, COMMIT, UPSTREAM, native_modules


def config(variant="diffmvs"):
    return {
        "model": {
            "variant": variant,
            "allow_random_init": True,
            "depth_range": [2.0, 80.0],
            "reference_indices": [0],
        }
    }


def scene(views=4):
    return collate_frames([SyntheticRigDataset(1, (64, 96), views, seed=1701)[0]])


class DiffMVSTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_source_fidelity(self):
        manifest = json.loads((UPSTREAM.parent / "source_manifest.json").read_text())
        self.assertEqual(COMMIT, manifest["commit"])
        for item in manifest["files"]:
            value = (UPSTREAM / item["path"]).read_bytes()
            if item["path"] == "models/module.py":
                value = value.replace(
                    b"# Compatibility: removed unused import-time cuda:0 inverse warm-up.\n"
                    b"# Model operators and learned parameters are unchanged.",
                    b'x = torch.inverse(torch.ones((1, 1), device="cuda:0"))',
                )
            self.assertEqual(hashlib.sha256(value).hexdigest(), item["sha256"])

    def test_real_network_forward_backward_two_variants(self):
        for variant in ("diffmvs", "casdiffmvs"):
            for views in (2, 3, 4):
                with self.subTest(variant=variant, views=views):
                    torch.manual_seed(19)
                    batch = scene(views)
                    model = Adapter(config(variant))
                    prediction = model(batch["inputs"], batch["targets"])
                    self.assertEqual(prediction["depth"].shape, (1, 1, 64, 96))
                    self.assertTrue(torch.isfinite(prediction["depth"]).all())
                    losses = model.loss(prediction, batch["inputs"], batch["targets"])
                    self.assertTrue(torch.isfinite(losses["total"]))
                    losses["total"].backward()
                    self.assertTrue(all(
                        parameter.grad is not None
                        for parameter in model.parameters() if parameter.requires_grad
                    ))
                    for prefix in ("feature", "context", "depthnet", "update_block"):
                        gradients = [
                            parameter.grad for name, parameter in model.native.named_parameters()
                            if name.startswith(prefix) and parameter.grad is not None
                        ]
                        self.assertTrue(gradients, prefix)
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients), prefix)
                        self.assertTrue(any(g.abs().sum() > 0 for g in gradients), prefix)

    def test_independent_native_input_and_output_parity(self):
        batch = scene()
        inputs = batch["inputs"]
        cfg = config()
        cfg["model"]["reference_indices"] = [2]
        model = Adapter(cfg).eval()
        Native, _ = native_modules()
        native = Native(deepcopy(model.args), test=True).eval()
        native.load_state_dict(model.native.state_dict(), strict=True)
        order = [2, 0, 1, 3]
        cameras = torch.zeros(1, 4, 2, 4, 4)
        cameras[:, :, 0] = inputs["T_cam_from_rig"][:, order]
        cameras[:, :, 1, :3, :3] = inputs["K"][:, order]
        projections = {}
        for stage, stride in enumerate((8, 4, 2, 1), 1):
            projections[f"stage{stage}"] = cameras.clone()
            projections[f"stage{stage}"][:, :, 1, :2] /= stride
        depths = torch.linspace(1 / 80.0, 1 / 2.0, 384)[None]
        with torch.no_grad():
            torch.manual_seed(812)
            official = native([inputs["rgb"][:, i] for i in order], projections, depths)
            torch.manual_seed(812)
            adapted = model(inputs)
        for key in ("depth", "photometric_confidence"):
            for expected, actual in zip(official[key], adapted["native"][0][key]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_checkpoint_optimizer_reload_and_inference_label_boundary(self):
        batch = scene()
        model = Adapter(config())
        optimizer, _ = model.configure_optimizer()
        result = model(batch["inputs"], batch["targets"])
        loss = model.loss(result, batch["inputs"], batch["targets"])["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        buffer = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
        buffer.seek(0)
        state = torch.load(buffer, weights_only=False)
        restored = Adapter(config())
        restored.load_state_dict(state["model"], strict=True)
        restored_optimizer, _ = restored.configure_optimizer()
        restored_optimizer.load_state_dict(state["optimizer"])
        self.assertEqual(len(optimizer.state), len(restored_optimizer.state))
        with torch.no_grad():
            torch.manual_seed(9)
            expected = model.eval()(batch["inputs"])["depth"]
            batch["targets"]["depth"].fill_(0.001)
            torch.manual_seed(9)
            actual = restored.eval()(batch["inputs"])["depth"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "推理禁止"):
            restored(batch["inputs"], batch["targets"])
        with self.assertRaisesRegex(ValueError, "扩散训练需要"):
            restored.train()(batch["inputs"])

    def test_missing_source_is_removed_not_replaced(self):
        batch = scene()
        inputs = deepcopy(batch["inputs"])
        inputs["view_mask"][:, 3] = False
        inputs["rgb"][:, 3] = float("nan")
        model = Adapter(config()).eval()
        compact = {
            key: value[:, :3] for key, value in inputs.items()
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == 4
        }
        with torch.no_grad():
            torch.manual_seed(16)
            actual = model(inputs)["depth"]
            torch.manual_seed(16)
            expected = model(compact)["depth"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        inputs["view_mask"][:, 0] = False
        with self.assertRaisesRegex(ValueError, "参考相机必须存在"):
            model(inputs)

    def test_scale_covariance_and_fixed_range(self):
        torch.manual_seed(19)
        inputs = scene()["inputs"]
        original = Adapter(config()).eval()
        cfg = config()
        cfg["model"]["depth_range"] = [4.0, 160.0]
        scaled = Adapter(cfg).eval()
        scaled.load_state_dict(original.state_dict(), strict=True)
        larger = deepcopy(inputs)
        larger["T_cam_from_rig"][..., :3, 3] *= 2
        with torch.no_grad():
            torch.manual_seed(6)
            expected = original(inputs)["depth"] * 2
            torch.manual_seed(6)
            actual = scaled(larger)["depth"]
        # 多次grid_sample/随机噪声细化的float32误差会放大，不宣称逐位尺度等变。
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=5e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(DiffMVSTest)
    )
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "commit": COMMIT,
            "device": "cpu",
            "architecture": ["DiffMVS", "CasDiffMVS"],
            "weight_initialization": "random; no accuracy claim",
            "tests_run": result.testsRun,
            "success": result.wasSuccessful(),
            "failures": [str(test) for test, _ in result.failures],
            "errors": [str(test) for test, _ in result.errors],
            "gpu": "not run; eight GPUs reserved for V5 training",
        }, indent=2) + "\n")
    raise SystemExit(not result.wasSuccessful())
