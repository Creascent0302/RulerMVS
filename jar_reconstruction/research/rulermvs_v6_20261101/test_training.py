"""V6训练接入：独立学习率、阶段冻结、输入协议及验证状态隔离。"""

import copy
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from jar.runtime import capture_rng
from research.early_geometry_20261030.test_encoding import TinyViT, inputs
from research.rulermvs_v6_20261101.training import (
    EqualPixelEvaluation, V6Hooks, optimizer_for, regional_selection_risk, single_model_validation,
)


def tiny_config():
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "research/rulermvs_v5_20261031/configs/train_public.json").read_text())
    config.update(architecture="rulermvs_v6", input_mode="equal_pixels",
                  handoff={"mode": "parent", "within_group": "mean", "upsample_feature": "winner"})
    config["model"].update(channels=8, raw_channels=8, depth_bins=8, peak_count=2)
    config["bridge"] = {"bottleneck": 4, "hidden": 8}
    config["stages"] = [
        {"number": 1, "trainable_prefixes": ["decoder."], "foundation_blocks": 0,
         "decoder_lr": 1e-4, "handoff_lr": 1e-3},
        {"number": 2, "trainable_prefixes": ["decoder.", "bridge.", "matching.lateral."],
         "foundation_blocks": 1, "decoder_lr": 1e-5, "handoff_lr": 1e-4,
         "bridge_lr": 2e-5, "fpn_lr": 3e-5, "foundation_lr": 1e-6},
    ]
    return config


def tiny_model(config):
    hooks = V6Hooks(config)
    with patch("jar.pretraining.encoders._build_native_foundation", side_effect=TinyViT):
        model = hooks.model_factory(config, initialize=False)
    return model, hooks


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(67)

    def test_stages_and_handoff_have_independent_optimizers(self):
        config = tiny_config()
        model, _ = tiny_model(config)
        model.set_stage(1)
        optimizer = optimizer_for(model, config["stages"][0], config)
        self.assertEqual({g["group_name"] for g in optimizer.param_groups}, {"decoder", "handoff"})
        for group in optimizer.param_groups:
            self.assertEqual(group["lr"], config["stages"][0][group["group_name"] + "_lr"])
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        self.assertTrue(optimizer.state)
        model.set_stage(2)
        following = optimizer_for(model, config["stages"][1], config)
        self.assertFalse(following.state)
        self.assertEqual({g["group_name"] for g in following.param_groups},
                         {"decoder", "handoff", "bridge", "fpn", "foundation"})
        self.assertTrue(all(not p.requires_grad for p in model.matching.backbone.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.foundation.model.norm.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.foundation.model.blocks[-1].parameters()))

    def test_optimizer_restore_preserves_group_rates_and_moments(self):
        config = tiny_config()
        model, _ = tiny_model(config)
        model.set_stage(1)
        optimizer = optimizer_for(model, config["stages"][0], config)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        snapshot = io.BytesIO()
        torch.save(optimizer.state_dict(), snapshot)
        snapshot.seek(0)
        restored = optimizer_for(model, config["stages"][0], config)
        restored.load_state_dict(torch.load(snapshot, weights_only=False))
        for original, actual in zip(optimizer.param_groups, restored.param_groups):
            for key in ("group_name", "lr", "initial_lr", "weight_decay"):
                self.assertEqual(original[key], actual[key])
        for parameter, state in optimizer.state.items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, restored.state[parameter][key]))

    def test_equal_pixels_train_and_validation_do_not_touch_geometry_or_targets(self):
        config = tiny_config()
        hooks = V6Hooks(config)
        data = inputs()
        targets = {"depth": torch.ones(1, 3, 32, 48)}
        batch = {"inputs": data, "targets": targets}
        domain = {"depth_scale": 1.}
        prepared, returned_domain = hooks.prepare_batch(batch, domain)
        self.assertIs(prepared["targets"], targets)
        self.assertIs(returned_domain, domain)
        self.assertIsNot(prepared["inputs"]["matching_rgb"], data["matching_rgb"])
        for key in data:
            if key != "matching_rgb":
                self.assertIs(prepared["inputs"][key], data[key])

        class Echo(nn.Module):
            def forward(self, received, domain, reference=0):
                return received

        evaluated = EqualPixelEvaluation(Echo())(data, domain)
        self.assertTrue(torch.equal(evaluated["matching_rgb"], prepared["inputs"]["matching_rgb"]))
        native = V6Hooks(dict(config, input_mode="native"))
        self.assertIs(native.prepare_batch(batch, domain)[0], batch)

    def test_validation_restores_buffers_modes_and_rng_even_on_failure(self):
        model = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout())
        model.train()
        model[0].eval()
        before = copy.deepcopy(model.state_dict())
        runtime = SimpleNamespace(device=torch.device("cpu"), distributed=False)
        rng = capture_rng(runtime)
        with self.assertRaisesRegex(RuntimeError, "forced"):
            with single_model_validation(model, runtime):
                self.assertFalse(model.training)
                model[0].running_mean.fill_(99)
                torch.rand(17)
                raise RuntimeError("forced")
        self.assertTrue(model.training)
        self.assertFalse(model[0].training)
        self.assertTrue(model[1].training)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))
        self.assertTrue(torch.equal(rng["torch"], torch.get_rng_state()))

    def test_explicit_zero_learning_rate_for_unfrozen_group_is_rejected(self):
        config = tiny_config()
        model, _ = tiny_model(config)
        model.set_stage(1)
        stage = dict(config["stages"][0], handoff_lr=0.)
        with self.assertRaisesRegex(ValueError, "正学习率"):
            optimizer_for(model, stage, config)


class RegionalSelectionTests(unittest.TestCase):
    weights = {"full": .70, "near": .15, "mid": .10, "far": .05}

    @staticmethod
    def row():
        return {
            "full": {"pixels": 10, "mae": 1., "rmse": 2., "abs_rel": .1},
            "near": {"pixels": 4, "mae": .5, "rmse": 1., "abs_rel": .2},
            "mid": {"pixels": 6, "mae": 2., "rmse": 3., "abs_rel": .05},
            "far": {"pixels": 0},
        }

    def test_empty_far_reweights_remaining_regions(self):
        row = self.row()
        risks = {name: (math.log1p(r["mae"] / 10.) + math.log1p(r["rmse"] / 10.)
                        + math.log1p(r["abs_rel"])) / 3.
                 for name, r in row.items() if r["pixels"]}
        expected = sum(self.weights[name] * value for name, value in risks.items()) / .95
        actual = regional_selection_risk([row], [{"depth_scale": 1.}], self.weights)
        self.assertAlmostEqual(actual, expected)

    def test_independent_dataset_unit_scaling_preserves_macro(self):
        row = self.row()
        other = copy.deepcopy(row)
        for region in other.values():
            if region["pixels"]:
                region["mae"] *= 100.
                region["rmse"] *= 100.
        single = regional_selection_risk([row], [{"depth_scale": 1.}], self.weights)
        mixed = regional_selection_risk([row, other],
                                         [{"depth_scale": 1.}, {"depth_scale": 100.}], self.weights)
        self.assertAlmostEqual(single, mixed)
        explicit = regional_selection_risk([row, other],
            [{"selection_depth_scale": 10., "depth_scale": 1.},
             {"selection_depth_scale": 1000., "depth_scale": 100.}], self.weights)
        self.assertAlmostEqual(single, explicit)

    def test_near_degradation_changes_score_even_when_full_unchanged(self):
        row = self.row()
        worse = copy.deepcopy(row)
        worse["near"]["mae"] *= 2.
        domain = {"depth_scale": 1.}
        self.assertGreater(regional_selection_risk([worse], [domain], self.weights),
                           regional_selection_risk([row], [domain], self.weights))
        self.assertEqual(regional_selection_risk([worse], [domain], {"full": 1.}),
                         regional_selection_risk([row], [domain], {"full": 1.}))


if __name__ == "__main__":
    unittest.main()
