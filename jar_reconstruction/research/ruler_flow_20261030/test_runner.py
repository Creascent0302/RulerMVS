"""训练入口的参数边界、显式损失调用、独立随机流和严格恢复规则。"""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from research.ruler_flow_20261030 import pilot


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
        self.decoder.register_buffer("depth_lattice", torch.ones(2))
        self.decoder.add_module("velocity", nn.Linear(3, 1))
        self.decoder.flow_options = {"inference_steps": 1}


class RunnerTests(unittest.TestCase):
    def test_only_new_parameters_train_and_old_statistics_stay_eval(self):
        model = ToyModel()
        new = {name for name, _ in model.named_parameters() if ".velocity." in name}
        pilot.set_trainable(model, new)
        pilot.train_mode(model, new)
        self.assertTrue(model.training)
        self.assertFalse(model.decoder.training)
        self.assertFalse(model.decoder[1].training)
        self.assertTrue(model.decoder.velocity.training)
        self.assertEqual({name for name, p in model.named_parameters() if p.requires_grad}, new)
        old = {name: value.clone() for name, value in model.state_dict().items() if name not in new}
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], .1)
        x = model.decoder[1](model.decoder[0](torch.ones(2, 3)))
        model.decoder.velocity(x).square().sum().backward()
        optimizer.step()
        for name, value in old.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]))

    def test_control_keeps_original_decoder_training(self):
        model = ToyModel()
        pilot.set_trainable(model, set(), control=True)
        pilot.train_mode(model, set(), control=True)
        self.assertTrue(model.decoder[1].training)
        self.assertTrue(all(p.requires_grad for p in model.decoder.parameters()))

    def test_loss_receives_live_model_and_separate_targets(self):
        model = ToyModel()
        inputs, targets, domain = {"rgb": torch.ones(2, 3)}, {"depth": torch.ones(2, 1)}, {"name": "toy"}
        prediction = {"depth": model.decoder.velocity(inputs["rgb"])}
        reference = prediction["depth"].detach().clone()
        calls = []

        def auxiliary(*arguments):
            calls.append(arguments)
            return {"total": (arguments[1]["depth"] - arguments[3]["depth"]).square().mean()}

        module = SimpleNamespace(training_loss=auxiliary)
        base = {"total": prediction["depth"].sum() * 0, "supervised_pixels": torch.tensor(2)}
        with patch.dict("sys.modules", {"research.ruler_flow_20261030.variants": module}), \
                patch.object(pilot, "depth_objective", return_value=base):
            loss = pilot.objective(model, prediction, inputs, targets, domain, {},
                                   {"variant": {"objective": "flow"}, "actual_fine_risk": False}, {})
        loss["total"].backward()
        self.assertIs(calls[0][0], model)
        self.assertIs(calls[0][3], targets)
        self.assertTrue(torch.equal(prediction["depth"].detach(), reference))
        self.assertIsNotNone(model.decoder.velocity.weight.grad)

    def test_auxiliary_generator_does_not_consume_global_rng(self):
        before = torch.random.get_rng_state()
        generator = torch.Generator().manual_seed(1701)
        torch.randn((4, 3), generator=generator)
        torch.rand((4,), generator=generator)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_only_runtime_grid_is_excluded_from_old_hash(self):
        model = ToyModel()
        names = set(model.state_dict())
        start = pilot.parameter_hash(pilot.old_state(model, names))
        model.decoder.depth_lattice.fill_(2)
        self.assertEqual(start, pilot.parameter_hash(pilot.old_state(model, names)))
        with torch.no_grad():
            model.decoder[0].weight.add_(1)
        self.assertNotEqual(start, pilot.parameter_hash(pilot.old_state(model, names)))

    def test_inference_steps_are_explicit(self):
        model = ToyModel()
        pilot.set_inference_steps(model, 4)
        self.assertEqual(model.decoder.flow_options["inference_steps"], 4)
        del model.decoder.flow_options
        with self.assertRaises(ValueError):
            pilot.set_inference_steps(model, 4)


if __name__ == "__main__":
    unittest.main()
