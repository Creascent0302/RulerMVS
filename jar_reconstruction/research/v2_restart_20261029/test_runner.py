"""连续独立重训的真实 runner CPU 回归；不启动 CUDA 或读取真实数据。"""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jar.pretraining import runner
from research.v2_training_20261028 import test_runner as old_runner

ToyModel = old_runner.ToyModel


class ContinuousRunnerTest(unittest.TestCase):
    setUp = old_runner.RunnerTest.setUp
    hooks = old_runner.RunnerTest.hooks
    load = old_runner.RunnerTest.load
    events = old_runner.RunnerTest.events

    def config(self, epochs):
        config = old_runner.RunnerTest.config(self, epochs, policy=False)
        config.pop("initial_model_checkpoint")
        config["train"]["stage_policy"] = "continuous_new"
        return config

    def test_positive_global_best_is_saved_without_stage_rollback(self):
        config = self.config([2, 1])
        hooks = self.hooks({1: [.2, .4], 2: [.3]})
        runner._run(config, None, self.runtime, hooks)
        best_weight = hooks.evaluate.weights[(1, True, 0)]
        last_stage1 = hooks.evaluate.weights[(1, True, 1)]
        final_weight = hooks.evaluate.weights[(2, True, 0)]
        self.assertNotEqual(last_stage1, best_weight)
        self.assertEqual(ToyModel.starts, [(1, 0.), (2, last_stage1)])
        for name in ("best.pt", "pretrained.pt"):
            self.assertEqual(float(self.load(name)["model"]["weight"]), best_weight)
            self.assertEqual(self.load(name)["selection_score"], .2)
            self.assertEqual(self.load(name)["global_step"], 1)
        self.assertEqual(float(self.load("stage1_last.pt")["model"]["weight"]), last_stage1)
        for name in ("final.pt", "last.pt", "stage2_last.pt"):
            self.assertEqual(float(self.load(name)["model"]["weight"]), final_weight)
        self.assertEqual(self.load("final.pt")["global_step"], 3)
        self.assertEqual(self.load("last.pt")["state"]["global_best_score"], .2)
        self.assertFalse((self.root / "run" / "incumbent.pt").exists())

    def test_boundary_resume_keeps_last_parameters_and_global_score(self):
        config = self.config([2, 1])
        hooks = self.hooks({1: [.2, .4], 2: [.3]})
        original = runner.save_training

        def stop_at_boundary(path, model, optimizer, state, config, runtime):
            original(path, model, optimizer, state, config, runtime)
            if state["stage_index"] == 1 and optimizer is None:
                raise InterruptedError("stage boundary")

        with patch.object(runner, "save_training", stop_at_boundary):
            with self.assertRaises(InterruptedError):
                runner._run(config, None, self.runtime, hooks)
        expected = float(self.load("last.pt")["model"]["weight"])
        runner._run(config, self.root / "run" / "last.pt", self.runtime, hooks)
        self.assertEqual(ToyModel.starts[-1], (2, expected))
        self.assertEqual(self.load("best.pt")["selection_score"], .2)
        self.assertEqual(self.load("last.pt")["state"]["global_best_score"], .2)

    def test_best_write_before_last_recovers_selection_without_loading_best(self):
        config = self.config([2, 1])
        hooks = self.hooks({1: [.2, .4, .5], 2: [.3]})
        original = runner.atomic_save

        def stop_after_best(record, path, minimum_free_gib):
            original(record, path, minimum_free_gib)
            if Path(path).name == "best.pt":
                raise InterruptedError("best persisted")

        with patch.object(runner, "atomic_save", stop_after_best):
            with self.assertRaisesRegex(RuntimeError, "best persisted"):
                runner._run(config, None, self.runtime, hooks)
        self.assertEqual(self.load("last.pt")["state"]["global_best_score"], float("inf"))
        runner._run(config, self.root / "run" / "last.pt", self.runtime, hooks)
        self.assertEqual(self.load("pretrained.pt")["selection_score"], .2)
        self.assertEqual(self.load("last.pt")["state"]["best_global_step"], 1)

    def test_smoke_saves_final_and_never_claims_full_best(self):
        config = self.config([1, 1])
        config["smoke"] = True
        hooks = self.hooks({})
        runner._run(config, None, self.runtime, hooks)
        self.assertEqual(self.load("final.pt")["global_step"], 2)
        self.assertFalse((self.root / "run" / "best.pt").exists())
        self.assertFalse((self.root / "run" / "pretrained.pt").exists())
        self.assertEqual(len(self.events("no_full_validation_best")), 1)

    def test_nonfinite_full_validation_keeps_final_but_refuses_best_export(self):
        config = self.config([1])
        hooks = self.hooks({1: [float("nan")]})
        with self.assertRaisesRegex(RuntimeError, "有限完整VAL"):
            runner._run(config, None, self.runtime, hooks)
        self.assertGreater(float(self.load("final.pt")["model"]["weight"]), 0.)
        self.assertFalse((self.root / "run" / "pretrained.pt").exists())

    def test_existing_weights_require_explicit_resume(self):
        config = self.config([1])
        hooks = self.hooks({1: [.2]})
        runner._run(config, None, self.runtime, hooks)
        before = (self.root / "run" / "last.pt").read_bytes()
        with self.assertRaisesRegex(FileExistsError, "禁止覆盖"):
            runner._run(config, None, self.runtime, hooks)
        self.assertEqual((self.root / "run" / "last.pt").read_bytes(), before)

    def test_validation_datasets_are_independent_from_training_domains(self):
        config = self.config([1])
        config["validation_datasets"] = [{"name": "val_upper", "length": 2},
                                          {"name": "val_down", "length": 3}]
        hooks = self.hooks({1: [.2]})
        original = hooks.evaluate
        seen = []

        def evaluate(model, dataset, config, runtime, **kwargs):
            seen.append([row["name"] for row in dataset.domains])
            return original(model, dataset, config, runtime, **kwargs)

        hooks.evaluate = evaluate
        runner._run(config, None, self.runtime, hooks)
        self.assertEqual(seen, [["val_upper", "val_down"], ["val_upper", "val_down"]])
        self.assertEqual(self.events("sampling_plan")[0]["domain_counts"], {"domain0": 1})


if __name__ == "__main__":
    unittest.main()
