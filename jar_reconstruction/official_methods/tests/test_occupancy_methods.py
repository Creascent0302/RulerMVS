"""官方发布审计、坐标桥接和真实GF2 GPU测试。"""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from official_methods.methods.octreeocc import audit_repository, create_adapter

ROOT = Path(__file__).resolve().parents[1]


class OctreeReleaseAudit(unittest.TestCase):
    def test_source_evidence_is_explicit(self):
        report = audit_repository()
        self.assertEqual(report["status"], "blocked_upstream_incomplete")
        kinds = {item["kind"] for item in report["findings"]}
        self.assertIn("invalid_official_config", kinds)
        self.assertIn("missing_import", kinds)
        with self.assertRaisesRegex(RuntimeError, "官方发布不完整"):
            create_adapter({})


def seed(value=17):
    torch.manual_seed(value)
    np.random.seed(value)


@unittest.skipUnless(torch.cuda.is_available(), "官方GF2核心算子是CUDA；没有CPU替代模型")
class GaussianFormer2Native(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "artifacts").mkdir(exist_ok=True)

    def test_four_view_training_checkpoint_and_native_equality(self):
        from mmengine.logging import MMLogger

        MMLogger.get_instance("gf2_tests", log_level="ERROR")
        from official_methods.methods.gaussianformer2 import Adapter
        from jar.synthetic import make_synthetic_batch

        seed()
        config = json.loads((ROOT / "configs/gaussianformer2_smoke.json").read_text())
        # 严格相等验收使用无碰撞fixture；64高斯小ROI上游spconv碰撞另记诊断，不能混淆。
        config["model"]["smoke_num_anchors"] = 4
        model = Adapter(config).cuda()
        batch = make_synthetic_batch(N=4, H=64, W=96, seed=17)
        inputs = {k: v.cuda() for k, v in batch["inputs"].items()}
        targets = {k: v.cuda() for k, v in batch["targets"].items()}
        # 必須实际使用第二代关键模块。
        self.assertEqual(type(model.native.lifter).__name__, "GaussianLifterV2")
        self.assertTrue(model.native.head.use_localaggprob)
        self.assertEqual(model.native_config.model.img_backbone.depth, 101)
        self.assertEqual(model.native_config.model.encoder.num_decoder, 4)
        self.assertFalse(any(p.requires_grad for p in model.native.lifter.projection.parameters()))

        args = model.native_inputs(inputs)
        point = torch.tensor([8.0, 0.0, 0.0, 1.0], device="cuda")
        camera = inputs["T_cam_from_rig"][0, 0] @ point
        project = inputs["K"][0, 0] @ camera[:3]
        wrapped = args["metas"]["projection_mat"][0, 0] @ point
        torch.testing.assert_close(wrapped[:2] / wrapped[2], project[:2] / project[2] + 0.5)
        self.assertNotIn("occ_label", args["metas"])

        model.train()
        optimizer = model.configure_optimizer()
        prediction = model(inputs)
        losses = model.loss(prediction, inputs, targets)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(value).all() for value in grads.values()))
        for component in ["img_backbone", "encoder", "random_anchors"]:
            self.assertTrue(
                any(component in name and value.abs().sum() > 0 for name, value in grads.items()),
                component,
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 35.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            seed(41)
            adapted = model(inputs)
            seed(41)
            native = model.native(**model.native_inputs(inputs))
            torch.testing.assert_close(
                adapted["native"]["pred_occ"][-1], native["pred_occ"][-1], rtol=0, atol=0
            )
            torch.testing.assert_close(
                adapted["occupancy"][0]["state"],
                (native["final_occ"].reshape(model.shape) != 17).to(torch.uint8),
            )
            # 在固定RGB/标定/RNG下任意更换外部targets；forward没有标签参数。
            changed = copy.deepcopy(targets)
            changed["depth"].fill_(200.0)
            changed["semantic"].fill_(-1)
            seed(41)
            again = model(inputs)
            torch.testing.assert_close(
                adapted["native"]["pred_occ"][-1], again["native"]["pred_occ"][-1], rtol=0, atol=0
            )
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            # 破坏一个实际已训练参数，重新加载必须恢复。
            parameter = next(p for p in model.parameters() if p.requires_grad)
            with torch.no_grad():
                parameter.add_(1.0)
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            with torch.no_grad():
                seed(41)
                restored = model(inputs)
            torch.testing.assert_close(
                adapted["native"]["pred_occ"][-1],
                restored["native"]["pred_occ"][-1],
                rtol=0,
                atol=0,
            )
        report = {
            "status": "passed",
            "gpu": torch.cuda.get_device_name(),
            "shape": list(model.shape),
            "gaussians": 4,
            "backbone": "two ResNet101 DCNv2",
            "decoder_blocks": 4,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "finite_gradient_tensors": len(grads),
            "losses": {k: float(v.detach()) for k, v in losses.items()},
            "native_adapter_max_abs_difference": 0.0,
            "checkpoint_prediction_difference": 0.0,
            "labels_as_inference_input": False,
            "performance_reproduction": False,
        }
        (ROOT / "artifacts/gaussianformer2_native_smoke.json").write_text(
            json.dumps(report, indent=2)
        )

    def test_initializer_training_export_and_native_free_loss(self):
        from mmengine.logging import MMLogger

        MMLogger.get_instance("gf2_initializer_tests", log_level="ERROR")
        from official_methods.methods.gaussianformer2 import Adapter
        from jar.synthetic import make_synthetic_batch

        seed()
        config = json.loads((ROOT / "configs/gaussianformer2_smoke.json").read_text())
        config["model"].update(training_stage="initializer", smoke_num_anchors=4)
        model = Adapter(config).cuda().train()
        trainable = [name for name, value in model.native.named_parameters() if value.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith(("lifter.initialize_backbone.", "lifter.projection."))
                for name in trainable
            )
        )
        backbone = model.native.lifter.initialize_backbone.img_backbone
        self.assertFalse(any(p.requires_grad for p in backbone.layer1.parameters()))
        self.assertFalse(any(p.requires_grad for p in backbone.conv1.parameters()))
        self.assertTrue(any(p.requires_grad for p in backbone.layer2.parameters()))
        batch = make_synthetic_batch(N=4, H=64, W=96, seed=17)
        inputs = {k: v.cuda() for k, v in batch["inputs"].items()}
        targets = {k: v.cuda() for k, v in batch["targets"].items()}
        original_k = inputs["K"].clone()
        original_rgb = inputs["rgb"].clone()
        augmented = model.prepare_training_inputs(inputs, targets)
        self.assertIn("occupancy", targets)
        torch.testing.assert_close(inputs["K"], original_k)
        torch.testing.assert_close(inputs["rgb"], original_rgb)
        self.assertFalse(torch.equal(augmented["rgb"], original_rgb))
        transform = augmented["K"][0, 0] @ original_k[0, 0].inverse()
        self.assertTrue(
            torch.allclose(transform, torch.eye(3, device="cuda"), atol=1e-5)
            or torch.allclose(
                transform, transform.new_tensor([[-1, 0, 95], [0, 1, 0], [0, 0, 1]]), atol=1e-5
            )
        )
        prediction = model(augmented)
        self.assertNotIn("occupancy", prediction)
        losses = model.loss(prediction, augmented, targets)
        optimizer = model.configure_optimizer()
        losses["total"].backward()
        for component in ("initialize_backbone", "projection"):
            gradients = [
                p.grad
                for name, p in model.named_parameters()
                if component in name and p.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.0)
        optimizer.step()
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            path = Path(directory) / "initializer.pt"
            model.export_initializer(path)
            state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
            occupancy_config = copy.deepcopy(config)
            occupancy_config["model"].update(
                training_stage="occupancy", initializer_checkpoint=str(path)
            )
            reloaded = Adapter(occupancy_config)
            for key, tensor in state.items():
                torch.testing.assert_close(
                    reloaded.native.lifter.state_dict()[key], tensor, rtol=0, atol=0
                )
            self.assertFalse(
                any(
                    p.requires_grad for p in reloaded.native.lifter.initialize_backbone.parameters()
                )
            )
            self.assertTrue(reloaded.native.lifter.random_anchors.requires_grad)
        # 官方lovasz忽略free=17；全free时仍须返回标量且保留零梯度链。
        from loss.utils.lovasz_softmax import lovasz_softmax

        logits = torch.randn(12, 18, device="cuda", requires_grad=True)
        labels = torch.full((12,), 17, device="cuda", dtype=torch.long)
        value = lovasz_softmax(logits.softmax(-1), labels, ignore=17)
        self.assertEqual(value.ndim, 0)
        self.assertEqual(float(value), 0.0)
        value.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))
        logits = torch.randn(1, 18, 12, device="cuda", requires_grad=True)
        native_free_loss = model.native_losses.losses[0](
            {
                "pred_occ": [logits.softmax(1)],
                "sampled_xyz": torch.zeros(1, 12, 3, device="cuda"),
                "sampled_label": labels[None],
                "occ_mask": torch.ones(1, 12, device="cuda", dtype=torch.bool),
            }
        )
        self.assertEqual(native_free_loss.ndim, 0)
        self.assertTrue(torch.isfinite(native_free_loss))
        native_free_loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        model.eval()
        self.assertIs(model.prepare_training_inputs(inputs, targets), inputs)
        (ROOT / "artifacts/gaussianformer2_initializer_smoke.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "initializer_loss": float(losses["total"].detach()),
                    "trainable_parameter_tensors": len(trainable),
                    "frozen_layer1_preserved": True,
                    "backbone_and_projection_finite_gradients": True,
                    "export_reload_exact": True,
                    "native_photometric_and_geometry_consistent_flip": True,
                    "all_free_lovasz_scalar_zero": True,
                    "all_free_native_occupancy_loss_finite_backward": True,
                    "official_stage1_training_config_available": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
