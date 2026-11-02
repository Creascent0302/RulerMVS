"""独立训练配置和参数分组的CPU回归，不读取TEST标签。"""

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from research.ruler_flow_v2_20261031.training import flow_optimizer
from research.v2_restart_20261029.training import selection_risk


class TrainingTests(unittest.TestCase):
    def config(self, name):
        return json.loads((Path(__file__).parent / "configs" / (name + ".json")).read_text())

    def test_separate_output_and_full_budget(self):
        public = self.config("train_public")
        carla = self.config("train_carla_new")
        self.assertEqual(public["architecture"], "ruler_flow_v2")
        self.assertEqual(carla["initial_model_checkpoint"], public["output"] + "/best.pt")
        self.assertEqual(sum(s["max_epochs"] for s in public["stages"]), 11)
        self.assertEqual(sum(s["max_epochs"] for s in carla["stages"]), 24)
        self.assertEqual(len(public["datasets"]), 7)
        self.assertEqual({d["reference_index"] for d in carla["validation_datasets"]}, {0, 1, 2, 3})
        for config in (public, carla):
            self.assertIn("ruler_flow_v2_20261031", config["output"])
            self.assertEqual(config["train"]["precision"], "fp32")
            self.assertFalse(config["train"]["allow_tf32"])
            self.assertFalse(config["evaluation"]["test_after_training"])
            self.assertEqual(config["train"]["minimum_free_gib"], 200)
            self.assertEqual(config["train"]["stage_policy"], "continuous_new")
            self.assertTrue(all(s["min_epochs"] == s["max_epochs"] for s in config["stages"]))

    def test_pilot_control_conditions(self):
        for phase in ("public", "carla_new"):
            configs = [self.config("pilot_" + phase + "_" + name)
                       for name in ("control", "direct", "meanflow")]
            for key in ("checkpoint", "data_config", "seed", "batch_size", "epochs",
                        "train_samples_per_domain", "lr", "new_lr", "augmentation"):
                self.assertEqual(len({str(c[key]) for c in configs}), 1)
            self.assertEqual(configs[1]["flow"]["objective"], "direct")
            self.assertEqual(configs[2]["flow"]["objective"], "meanflow")

    def test_optimizer_groups_new_field_separately(self):
        model = nn.Module()
        model.decoder = nn.Module()
        model.decoder.flow_field = nn.Linear(2, 1)
        model.decoder.original = nn.Linear(2, 1)
        model.matching = nn.Module()
        model.matching.backbone = nn.Linear(2, 1)
        model.matching.head = nn.Linear(2, 1)
        model.foundation = nn.Linear(2, 1)
        stage = {kind + "_lr": value for kind, value in
                 (("flow", .001), ("decoder", .00001), ("matching", .000001),
                  ("fpn", .000005), ("foundation", .0000001))}
        optimizer = flow_optimizer(model, stage, {"train": {"weight_decay": .0001}})
        found = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
        self.assertEqual(set(found), {"flow", "decoder", "matching", "fpn", "foundation"})
        self.assertEqual(found["flow"], .001)
        all_parameters = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(all_parameters), len(set(all_parameters)))
        self.assertEqual(set(all_parameters), {id(p) for p in model.parameters()})

    def test_validation_score_is_scale_invariant(self):
        a = [{"full": {"mae": 1., "rmse": 2., "abs_rel": .1}}]
        b = [{"full": {"mae": 100., "rmse": 200., "abs_rel": .1}}]
        self.assertAlmostEqual(selection_risk(a, [{"depth_scale": 1.}]),
                               selection_risk(b, [{"depth_scale": 100.}]))


if __name__ == "__main__":
    unittest.main()
