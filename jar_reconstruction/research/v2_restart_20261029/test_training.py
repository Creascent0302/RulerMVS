"""独立训练选模和配置回归；不读取真实TEST或加载baseline权重。"""

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.v2_restart_20261029.training import selection_risk
from research.v2_training_20261028.sampling import RebalancedFullPassSampler


class TrainingProtocolTest(unittest.TestCase):
    def test_selection_scale_invariance_and_monotonicity(self):
        row = {"full": {"mae": 1., "rmse": 2., "abs_rel": .2}}
        domain = {"depth_scale": .5}
        score = selection_risk([row], [domain])
        rescaled = {"full": {"mae": 100., "rmse": 200., "abs_rel": .2}}
        self.assertAlmostEqual(score, selection_risk([rescaled], [{"depth_scale": 50.}]))
        worse = copy.deepcopy(row)
        worse["full"]["rmse"] = 3.
        self.assertGreater(selection_risk([worse], [domain]), score)

    def test_fresh_public_and_separate_carla(self):
        folder = Path(__file__).parent / "configs"
        public = json.loads((folder / "train_public.json").read_text())
        carla = json.loads((folder / "train_carla_new.json").read_text())
        self.assertFalse(public["initial_model_checkpoint"])
        self.assertNotIn("teacher_checkpoint", public)
        self.assertNotIn("teacher_checkpoint", carla)
        self.assertEqual(carla["initial_model_checkpoint"], public["output"] + "/best.pt")
        self.assertEqual(len(carla["validation_datasets"]), 4)
        self.assertEqual({d["reference_index"] for d in carla["validation_datasets"]}, {0, 1, 2, 3})
        for config in (public, carla):
            self.assertEqual(config["train"]["stage_policy"], "continuous_new")
            self.assertFalse(config["evaluation"]["test_after_training"])
            self.assertTrue(all(s["min_epochs"] == s["max_epochs"] for s in config["stages"]))
            self.assertTrue(all(not s.get("require_previous_improvement") for s in config["stages"]))

    def test_sampler_covers_all_train_and_resume_suffix(self):
        args = ([11, 23, 5], 1, 0, 2, 2, 1, 123)
        options = {"domain_multipliers": [1., 1.5, 4.], "domain_schedule": "balanced_prefix"}
        full = RebalancedFullPassSampler(*args, **options)
        for domain, length in enumerate(args[0]):
            visited = {index for d, entries in full.updates if d == domain for index, epoch in entries}
            self.assertEqual(visited, set(range(length)))
        resumed = RebalancedFullPassSampler(*args, start_update=3, **options)
        self.assertEqual(list(resumed), list(full)[3:])


if __name__ == "__main__":
    unittest.main()
