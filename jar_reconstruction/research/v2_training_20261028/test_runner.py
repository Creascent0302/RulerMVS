"""生产runner的CPU流程回归：守门、阶段继承、批处理hook和断点恢复。"""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jar.pretraining import runner


class ToyModel(torch.nn.Module):
    architecture = "runner_cpu_test"
    starts = []

    def __init__(self, config, initialize=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def set_stage(self, number):
        self.stage = number
        self.starts.append((number, float(self.weight.detach())))

    def forward(self, inputs, domain):
        return {"depth": inputs["x"] * self.weight * domain.get("factor", 1.)}


class ToyDataset:
    def __init__(self, config, split):
        if split not in {"train", "val"}:
            raise AssertionError("流程测试不得读取TEST")
        self.domains = config["datasets"]
        self.datasets = [range(domain.get("length", 1)) for domain in self.domains]

    def __len__(self):
        return sum(map(len, self.datasets))

    def __getitem__(self, key):
        return {"inputs": {"x": torch.ones(1)}, "targets": {"y": torch.ones(1)},
                "domain_index": key[0]}


def toy_collate(samples):
    return {"inputs": {"x": torch.stack([sample["inputs"]["x"] for sample in samples])},
            "targets": {"y": torch.stack([sample["targets"]["y"] for sample in samples])},
            "domain_index": samples[0]["domain_index"]}


def toy_objective(prediction, inputs, targets, domain, config, correspondence_weight):
    return {"total": (prediction["depth"] - targets["y"]).square().mean()}


def toy_optimizer(model, stage, config):
    return torch.optim.SGD([{"params": list(model.parameters()), "lr": .1, "initial_lr": .1}])


class ScriptedEvaluation:
    def __init__(self, scores):
        self.scores = scores
        self.calls = {}
        self.weights = {}

    def __call__(self, model, dataset, config, runtime, limit=None, split="val"):
        full = limit is None
        key = model.stage, full
        index = self.calls.get(key, 0)
        self.calls[key] = index + 1
        self.weights[key + (index,)] = float(model.weight.detach())
        # 子集即使更好也不得成为全局守门权重。
        score = self.scores[model.stage][index] if full else -10.
        return {"score": score, "full": full, "split": split}


class RunnerTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = SimpleNamespace(device=torch.device("cpu"), rank=0,
                                       world_size=1, distributed=False, is_main=True)
        ToyModel.starts = []
        for target, replacement in [
            ("MixedDataset", ToyDataset),
            ("collate_mixed", toy_collate),
            ("load_pretraining_state", lambda model, state: model.load_state_dict(state)),
        ]:
            context = patch.object(runner, target, replacement)
            context.start()
            self.addCleanup(context.stop)
        for target, replacement in [
            ("torch.cuda.is_available", False),
            ("torch.cuda.max_memory_allocated", 0),
        ]:
            context = patch(target, return_value=replacement)
            context.start()
            self.addCleanup(context.stop)

    def config(self, epochs, optional_stage=None, policy=True, lengths=(1,)):
        manifest = self.root / "train.jsonl"
        manifest.write_text("{}\n")
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        config = {
            "output": str(self.root / "run"), "seed": 321,
            "initial_model_checkpoint": str(self.root / "initial.pt"),
            "datasets": [{"name": f"domain{i}", "length": length,
                          "manifests": {"train": str(manifest)},
                          "train_manifest_sha256": digest}
                         for i, length in enumerate(lengths)],
            "stages": [{"number": i + 1, "name": f"stage{i + 1}",
                        "max_epochs": count, "min_epochs": 1,
                        "correspondence_weight": 0.}
                       for i, count in enumerate(epochs)],
            "train": {"minimum_free_gib": 0, "allow_tf32": False,
                      "batch_size": 1, "accumulation": 1, "workers": 0,
                      "precision": "fp32", "grad_clip": 10., "log_every": 10,
                      "checkpoint_every": 10, "validation_samples_per_domain": 1,
                      "full_validation_every": 1},
            "evaluation": {"test_after_training": False},
        }
        if policy:
            config["train"]["stage_policy"] = "retain_global_best"
        if optional_stage:
            config["stages"][optional_stage - 1]["require_previous_improvement"] = True
        initial = ToyModel(config)
        torch.save({"architecture": initial.architecture, "model": initial.state_dict()},
                   config["initial_model_checkpoint"])
        return config

    def hooks(self, scores, prepare_batch=None):
        hooks = SimpleNamespace(model_factory=ToyModel, objective=toy_objective,
                                sampler_factory=runner.FullPassSampler,
                                evaluate=ScriptedEvaluation(scores),
                                selection_score=lambda result: result["score"],
                                optimizer_factory=toy_optimizer)
        if prepare_batch:
            hooks.prepare_batch = prepare_batch
        return hooks

    def load(self, name):
        return torch.load(self.root / "run" / name, map_location="cpu", weights_only=False)

    def events(self, event):
        rows = [json.loads(line) for line in (self.root / "run" / "events.jsonl").read_text().splitlines()]
        return [row for row in rows if row["event"] == event]

    def test_failed_stages_restore_original_incumbent(self):
        config = self.config([1, 1])
        hooks = self.hooks({1: [float("inf")], 2: [float("nan")]})
        runner._run(config, None, self.runtime, hooks)
        self.assertEqual(ToyModel.starts, [(1, 0.), (2, 0.)])
        for name in ("incumbent.pt", "pretrained.pt"):
            self.assertEqual(float(self.load(name)["model"]["weight"]), 0.)
        self.assertFalse(self.load("incumbent.pt")["qualified"])
        self.assertFalse(self.load("pretrained.pt")["promotion_eligible"])
        self.assertEqual(self.load("last.pt")["state"]["global_best_score"], 0.)
        self.assertTrue(all(not row["qualified"] for row in self.events("stage_complete")))

    def test_global_best_inheritance_and_optional_stage_skip(self):
        config = self.config([2, 1, 1], optional_stage=3)
        hooks = self.hooks({1: [-.2, -.1], 2: [-.15]})
        runner._run(config, None, self.runtime, hooks)
        best_weight = hooks.evaluate.weights[(1, True, 0)]
        self.assertNotEqual(best_weight, hooks.evaluate.weights[(1, True, 1)])
        self.assertEqual(ToyModel.starts, [(1, 0.), (2, best_weight)])
        self.assertEqual(float(self.load("pretrained.pt")["model"]["weight"]), best_weight)
        self.assertTrue(self.load("pretrained.pt")["promotion_eligible"])
        self.assertEqual(self.load("pretrained.pt")["selection_score"], -.2)
        self.assertEqual([row["stage"] for row in self.events("stage_skipped")], [3])
        self.assertEqual([row["stage_improved"] for row in self.events("stage_complete")], [True, False])
        self.assertEqual(len(self.events("incumbent_updated")), 1)

    def test_stage_boundary_resume_preserves_global_best_and_gate(self):
        config = self.config([1, 1], optional_stage=2)
        hooks = self.hooks({1: [-.2], 2: [-.1]})
        original_save = runner.save_training

        def interrupt_after_boundary(path, model, optimizer, state, config, runtime):
            original_save(path, model, optimizer, state, config, runtime)
            if state["stage_index"] == 1 and optimizer is None:
                raise InterruptedError("模拟阶段交界中断")

        with patch.object(runner, "save_training", interrupt_after_boundary):
            with self.assertRaises(InterruptedError):
                runner._run(config, None, self.runtime, hooks)
        before = (self.root / "run" / "incumbent.pt").read_bytes()
        state = self.load("last.pt")["state"]
        self.assertTrue(state["previous_stage_improved"])
        self.assertEqual(state["global_best_score"], -.2)
        runner._run(config, self.root / "run" / "last.pt", self.runtime, hooks)
        self.assertEqual((self.root / "run" / "incumbent.pt").read_bytes(), before)
        self.assertEqual([number for number, _ in ToyModel.starts], [1, 2])
        self.assertTrue(self.load("pretrained.pt")["promotion_eligible"])
        self.assertEqual(self.load("last.pt")["state"]["global_best_score"], -.2)

    def test_in_stage_resume_restores_improvement_flag(self):
        config = self.config([2, 1], optional_stage=2)
        hooks = self.hooks({1: [-.2, -.1], 2: [-.1]})
        original_save = runner.save_training

        def interrupt_after_full_val(path, model, optimizer, state, config, runtime):
            original_save(path, model, optimizer, state, config, runtime)
            if state["stage_index"] == 0 and state["epoch_in_stage"] == 1:
                raise InterruptedError("模拟阶段内VAL后中断")

        with patch.object(runner, "save_training", interrupt_after_full_val):
            with self.assertRaises(InterruptedError):
                runner._run(config, None, self.runtime, hooks)
        runner._run(config, self.root / "run" / "last.pt", self.runtime, hooks)
        self.assertEqual([row["stage"] for row in self.events("sampling_plan")], [1, 2])
        self.assertEqual([row["stage"] for row in self.events("stage_skipped")], [])
        self.assertEqual(self.load("pretrained.pt")["selection_score"], -.2)
        self.assertEqual(ToyModel.starts[-1][1], hooks.evaluate.weights[(1, True, 0)])

    def test_incumbent_write_before_last_is_recoverable(self):
        config = self.config([2, 1], optional_stage=2)
        hooks = self.hooks({1: [-.2, -.1, -.05], 2: [-.1]})
        original_save = runner.atomic_save

        def interrupt_after_incumbent(record, path, minimum_free_gib):
            original_save(record, path, minimum_free_gib)
            if Path(path).name == "incumbent.pt" and record["selection_score"] < 0:
                raise InterruptedError("模拟incumbent已写而last尚未更新")

        with patch.object(runner, "atomic_save", interrupt_after_incumbent):
            with self.assertRaisesRegex(RuntimeError, "incumbent已写"):
                runner._run(config, None, self.runtime, hooks)
        self.assertEqual(self.load("last.pt")["state"]["global_best_score"], 0.)
        self.assertEqual(self.load("incumbent.pt")["selection_score"], -.2)
        runner._run(config, self.root / "run" / "last.pt", self.runtime, hooks)
        self.assertEqual(self.load("pretrained.pt")["selection_score"], -.2)
        self.assertEqual(self.events("stage_skipped"), [])
        self.assertEqual(ToyModel.starts[-1][1], hooks.evaluate.weights[(1, True, 0)])

    def test_prepare_batch_and_domain_count_audit(self):
        config = self.config([1], lengths=(1, 2))
        prepared = []

        def prepare_batch(batch, domain):
            self.assertEqual(batch["inputs"]["x"].device.type, "cpu")
            prepared.append(domain["name"])
            batch["targets"]["y"] *= 2.
            return batch, {**domain, "factor": 2.}

        hooks = self.hooks({1: [-.1]}, prepare_batch)
        runner._run(config, None, self.runtime, hooks)
        self.assertEqual(sorted(prepared), ["domain0", "domain1", "domain1"])
        self.assertEqual(self.events("sampling_plan")[0]["domain_counts"], {"domain0": 1, "domain1": 2})
        self.assertNotIn("factor", config["datasets"][0])
        self.assertGreater(float(self.load("pretrained.pt")["model"]["weight"]), .5)

    def test_old_policy_keeps_stage_local_selection(self):
        config = self.config([1, 1], policy=False)
        hooks = self.hooks({1: [-.2], 2: [-.1]})
        runner._run(config, None, self.runtime, hooks)
        self.assertFalse((self.root / "run" / "incumbent.pt").exists())
        self.assertEqual(float(self.load("pretrained.pt")["model"]["weight"]),
                         hooks.evaluate.weights[(2, True, 0)])
        self.assertTrue(self.load("pretrained.pt")["promotion_eligible"])

    def test_resume_rejects_changed_config(self):
        config = self.config([1])
        hooks = self.hooks({1: [-.1]})
        runner._run(config, None, self.runtime, hooks)
        changed = copy.deepcopy(config)
        changed["seed"] += 1
        with self.assertRaisesRegex(ValueError, "相同配置"):
            runner._run(changed, self.root / "run" / "last.pt", self.runtime, hooks)


if __name__ == "__main__":
    unittest.main()
