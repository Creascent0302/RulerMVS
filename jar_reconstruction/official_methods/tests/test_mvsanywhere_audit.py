"""MVSAnywhere 薄适配和训练循环的独立 CPU 契约测试。

小模块仅验证参数更新/续训等基础设施，不代表完整 Hero 前向测试。
官方优化器函数直接从已固定源码提取执行，避免为此加载两个 ViT。
"""

import ast
import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from jar.runtime import Runtime, seed_all
from official_methods import cli
from official_methods.methods.mvsanywhere import Adapter
from official_methods.methods.niantic_common import NianticAdapter, UPSTREAM


def native_optimizer_function():
    source = UPSTREAM / "mvsanywhere/src/mvsanywhere/experiment_modules/sr_depth_model.py"
    module = ast.parse(source.read_text())
    model = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "DepthModel"
    )
    method = next(
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef) and node.name == "configure_optimizers"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["configure_optimizers"]


class NativeProbe(nn.Module):
    """只记录适配后张量，刻意不模拟 Hero 的预测能力。"""

    def __init__(self):
        super().__init__()
        self.run_opts = SimpleNamespace(model_num_views=4, image_height=8, image_width=12)
        self.calls = []

    def forward(self, phase, cur, src, **kwargs):
        self.calls.append((phase, copy.deepcopy(cur), copy.deepcopy(src), kwargs))
        depth = cur["image_b3hw"][:, :1].abs() + 1
        return {"depth_pred_s0_b1hw": depth}


def probe_adapter():
    adapter = Adapter.__new__(Adapter)
    NianticAdapter.__init__(adapter, {"model": {"reference_indices": [2]}})
    adapter.native_model = NativeProbe()
    return adapter


def calibrated_inputs():
    images = torch.arange(4).float()[None, :, None, None, None].expand(1, 4, 3, 8, 12) / 4
    intrinsics = torch.tensor([[9.0, 0, 5.5], [0, 9.0, 3.5], [0, 0, 1.0]])
    intrinsics = intrinsics[None, None].expand(1, 4, 3, 3).clone()
    transforms = torch.eye(4)[None, None].expand(1, 4, 4, 4).clone()
    # 参考相机 2 的相机中心在 x=0；源中心距离依次为 3、1、2 米。
    transforms[0, :, 0, 3] = torch.tensor([-3.0, -1.0, 0.0, -2.0])
    return {"rgb": images, "K": intrinsics, "T_cam_from_rig": transforms}


class OptimizerProbe(nn.Module):
    def __init__(self):
        super().__init__()
        for name in (
            "cost_volume",
            "matching_model",
            "encoder",
            "depth_decoder",
            "cost_volume_net",
        ):
            setattr(self, name, nn.Linear(2, 2, dtype=torch.float64))
        self.run_opts = SimpleNamespace(
            lr=1e-4, lr_da_encoder=5e-6, lr_da_decoder=5e-5, wd=1e-4, lr_steps=[2, 4]
        )


class LoopProbe(nn.Module):
    """含随机分支，用于检查真实公共训练循环的精确恢复。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head = nn.Linear(3, 1)
        self.dropout = nn.Dropout(0.4)
        self.register_buffer("seen", torch.tensor(0))

    def forward(self, inputs):
        self.seen.add_(1)
        return self.head(self.dropout(inputs["rgb"].mean(dim=(1, 3, 4))))

    def loss(self, prediction, inputs, targets):
        valid = targets["depth_valid"]
        target = targets["depth"].masked_fill(~valid, 0).mean(dim=(1, 2, 3))[:, None]
        return {"total": (prediction - target).square().mean()}

    def configure_optimizer(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return optimizer, torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)


class MVSAnywhereAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_source_order_and_reference_anchor(self):
        adapter = probe_adapter().eval()
        inputs = calibrated_inputs()
        current, sources, order = adapter.prepare_native(inputs, 2)
        self.assertEqual(order.tolist(), [[1, 3, 0]])
        torch.testing.assert_close(current["cam_T_world_b44"], inputs["T_cam_from_rig"][:, 2])
        self.assertEqual(sources["cam_T_world_b44"][0, :, 0, 3].tolist(), [-1.0, -2.0, -3.0])
        mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        torch.testing.assert_close(current["image_b3hw"] * std + mean, inputs["rgb"][:, 2])

    def test_eval_ignores_training_range_and_flip(self):
        adapter = probe_adapter().eval()
        inputs = calibrated_inputs()
        expected = adapter(inputs)
        contaminated = dict(
            inputs, _training_depth_range=torch.full((1, 4, 2), 999.0), _geometric_flip=True
        )
        actual = adapter(contaminated)
        torch.testing.assert_close(expected["depth"], actual["depth"], rtol=0, atol=0)
        for phase, current, _, kwargs in adapter.native_model.calls:
            self.assertEqual(phase, "test")
            self.assertNotIn("min_depth", current)
            self.assertNotIn("max_depth", current)
            self.assertTrue(kwargs["estimate_depth_range"])
            self.assertEqual(kwargs["num_refinement_steps"], 1)

    def test_resume_does_not_reload_external_pretraining(self):
        options = SimpleNamespace(
            lr=1e-4,
            wd=1e-4,
            lr_steps=[70000, 80000],
            lr_da_encoder=5e-6,
            lr_da_decoder=5e-5,
        )
        constructor = Mock(return_value=NativeProbe())
        modules = {
            "mvsanywhere.options": SimpleNamespace(Options=lambda: options),
            "mvsanywhere.experiment_modules.sr_depth_model": SimpleNamespace(
                DepthModel=constructor
            ),
        }
        config = {
            "_loading_shared_checkpoint": True,
            "model": {
                "image_height": 48,
                "image_width": 80,
                "pretrained": True,
                "weights": "/missing/hero.ckpt",
                "da_weights": "/missing/da.pth",
            },
        }
        with (
            patch.dict("sys.modules", modules),
            patch.object(Adapter, "load_native_weights") as load_native,
        ):
            with patch("torch.load", side_effect=AssertionError("不应重新读取预训练权重")):
                Adapter(config)
        load_native.assert_not_called()
        self.assertFalse(constructor.call_args.args[0].pretrained)
        self.assertIsNone(constructor.call_args.args[0].da_weights_path)

    def test_invalid_configuration_fails_before_model_creation(self):
        for height, width in [(50, 80), (64, 100), (0, 96), (64.0, 96)]:
            with self.subTest(height=height, width=width), self.assertRaisesRegex(ValueError, "16"):
                Adapter({"model": {"image_height": height, "image_width": width}})
        with self.assertRaisesRegex(ValueError, "depth_bins"):
            Adapter({"model": {"depth_bins": 0}})

    def test_grouped_optimizer_matches_three_official_optimizers(self):
        seed_all(61)
        native = OptimizerProbe()
        adapted = probe_adapter()
        adapted.native_model = copy.deepcopy(native)
        originals, schedules = native_optimizer_function()(native)
        combined, schedule = adapted.configure_optimizer()
        for step in range(8):
            for index, (left, right) in enumerate(
                zip(native.parameters(), adapted.native_model.parameters())
            ):
                gradient = torch.full_like(left, (index + 1) * 0.01 + step * 0.001)
                left.grad = gradient.clone()
                right.grad = gradient.clone()
            for optimizer, scheduler in zip(originals, schedules):
                optimizer.step()
                scheduler.step()
            combined.step()
            schedule.step()
            for left, right in zip(native.parameters(), adapted.native_model.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=1e-14)
            for original, group in zip(originals, combined.param_groups):
                self.assertAlmostEqual(original.param_groups[0]["lr"], group["lr"], places=17)

    def test_training_resume_matches_continuous_across_epoch(self):
        runtime = Runtime(torch.device("cpu"), 0, 1, 0, False, True)
        config = {
            "method": "mvsanywhere",
            "seed": 73,
            "synthetic_length": 3,
            "image_size": [32, 48],
            "training": {
                "batch_size": 1,
                "num_workers": 0,
                "max_steps": 4,
                "gradient_accumulation_steps": 2,
                "checkpoint_every": 1,
            },
        }
        with tempfile.TemporaryDirectory(prefix="mvs_training_audit_", dir="/tmp") as directory:
            folder = Path(directory)

            def run(name, steps, checkpoint=""):
                local = copy.deepcopy(config)
                local["training"]["max_steps"] = steps
                args = SimpleNamespace(
                    synthetic=True, checkpoint=checkpoint, output=str(folder / name)
                )
                seed_all(73)
                with patch.object(cli, "build_adapter", LoopProbe), redirect_stdout(io.StringIO()):
                    cli.train(args, local, runtime)
                return folder / name / "last.pt"

            uninterrupted = run("full", 4)
            partial = run("partial", 1)
            resumed = run("resumed", 4, str(partial))
            full = torch.load(uninterrupted, map_location="cpu", weights_only=False)
            restored = torch.load(resumed, map_location="cpu", weights_only=False)
            for key, value in full["model"].items():
                torch.testing.assert_close(value, restored["model"][key], rtol=0, atol=0)
            self.assertEqual(full["scheduler"], restored["scheduler"])
            self.assertEqual(full["step"], restored["step"])
            self.assertEqual(full["batch_in_epoch"], restored["batch_in_epoch"])


if __name__ == "__main__":
    unittest.main()
