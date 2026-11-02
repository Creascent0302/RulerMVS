"""研究入口的注入边界与恢复测试，不替代真实网络GPU验证。"""

import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn

from research.role_fusion_20261030 import pilot


class RunnerTests(unittest.TestCase):
    def test_callbacks_restore_after_failure(self):
        original = {name: getattr(pilot.engine, name) for name in
                    ("install", "sources", "depth_objective", "atomic_save")}

        def failing_run(settings):
            self.assertIsNot(pilot.engine.install, original["install"])
            self.assertTrue(settings["verify_initial"])
            raise RuntimeError("intentional failure")

        with patch.object(pilot.engine, "run", failing_run):
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                pilot.run({"output": tempfile.gettempdir()})
        for name, value in original.items():
            self.assertIs(getattr(pilot.engine, name), value)

    def test_checkpoint_marker_and_nonidentity(self):
        captured = {}

        def saved(payload, filename, minimum):
            captured.update(payload)

        def fake_run(settings):
            self.assertFalse(settings["verify_initial"])
            pilot.engine.atomic_save({"experiment": "historical"}, "unused", 200)

        with patch.object(pilot.engine, "atomic_save", saved), patch.object(pilot.engine, "run", fake_run):
            pilot.run({"initial_identity": False, "output": tempfile.gettempdir()})
        self.assertEqual(captured["experiment"], pilot.EXPERIMENT)
        self.assertFalse(captured["initial_identity_expected"])

    def test_control_does_not_require_variant_module(self):
        marker = object()
        self.assertIs(pilot.install(marker, {}), marker)

    def test_new_module_gradients_are_recorded_separately(self):
        def installer(model, settings):
            model.decoder.context_null = nn.Parameter(torch.tensor(0.))
            model.decoder.handoff_gate = nn.Parameter(torch.tensor(-2.))
            return model

        def fake_run(settings):
            model = nn.Module()
            model.decoder = nn.Module()
            pilot.engine.install(model, settings)
            (2 * model.decoder.context_null + 3 * model.decoder.handoff_gate).backward()
            filename = Path(settings["output"]) / "completed.json"
            filename.write_text(json.dumps({"new_parameter_changes": {
                "decoder.context_null": .1, "decoder.handoff_gate": .2,
            }}))

        with tempfile.TemporaryDirectory() as folder:
            with patch.object(pilot, "install", installer), patch.object(pilot.engine, "run", fake_run):
                pilot.run({"output": folder})
            checks = json.loads((Path(folder) / "new_module_checks.json").read_text())
        self.assertTrue(checks["passed"])
        self.assertEqual(checks["modules"]["context_null"]["maximum_absolute_gradient"], 2.)
        self.assertEqual(checks["modules"]["handoff_gate"]["maximum_absolute_gradient"], 3.)

    def test_adapter_only_step_keeps_old_decoder_parameters(self):
        model = nn.Module()
        model.foundation = nn.Linear(2, 2)
        model.decoder = nn.Module()
        model.decoder.old = nn.Linear(2, 2)
        model.decoder.context_projection = nn.Linear(2, 2)
        new = {name for name, _ in model.named_parameters() if "context_projection" in name}
        pilot.engine.set_trainable_parameters(model, new, [])
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        value = model.foundation(torch.ones(1, 2))
        output = model.decoder.old(value + model.decoder.context_projection(value))
        output.square().mean().backward()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        optimizer.step()
        for name, value in model.named_parameters():
            if name not in new:
                self.assertIsNone(value.grad)
                self.assertTrue(torch.equal(before[name], value))
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.named_parameters() if name in new))
        pilot.engine.set_trainable_parameters(model, new)
        self.assertTrue(all(p.requires_grad == name.startswith("decoder.") for name, p in model.named_parameters()))


if __name__ == "__main__":
    unittest.main()
