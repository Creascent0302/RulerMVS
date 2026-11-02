"""条件干预必须只改变指定分支，退出时恢复原值。"""

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from jar.geometry import pixel_grid
from research.early_geometry_20261030 import encoding
from research.early_geometry_20261030.diagnostic import intervention
from research.early_geometry_20261030.test_encoding import inputs


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.model = SimpleNamespace(
            _encoding_options={"camera_mode": "geometry"},
            matching=SimpleNamespace(camera_adapter=encoding.CameraAdapter(8)),
            foundation=SimpleNamespace(camera_adapter=encoding.CameraAdapter(8)),
            decoder=SimpleNamespace(late_gain=nn.Parameter(torch.randn(8))),
        )
        for branch in (self.model.matching, self.model.foundation):
            nn.init.normal_(branch.camera_adapter.net[-1].weight, std=.1)

    def test_condition_only_changes_requested_channels(self):
        item = inputs(torch.float64)
        original_inputs = {name: value.clone() for name, value in item.items()}
        grid = pixel_grid(3, 4, dtype=torch.float64, stride=4)
        args = (item, {"depth_scale": .7}, grid, (4., 4.))
        original_function = encoding.camera_descriptor
        expected = original_function(*args)
        for name, kept in (("baseline_zero", [0, 1, 2, 3, 4, 5]),
                           ("angular_canonical", [0, 1, 2, 6])):
            with intervention(self.model, name):
                changed = encoding.camera_descriptor(*args)
            torch.testing.assert_close(changed[:, :, kept], expected[:, :, kept], rtol=0, atol=0)
            self.assertGreater(float((changed - expected).abs().max()), 0.)
            self.assertIs(encoding.camera_descriptor, original_function)
            for key, value in original_inputs.items():
                torch.testing.assert_close(item[key], value, rtol=0, atol=0)

    def test_adapter_off_is_identity_and_recovers(self):
        feature = torch.rand(1, 8, 2, 3)
        condition = torch.rand(1, 7, 2, 3)
        for name, branch in (("matching_off", self.model.matching), ("vit_off", self.model.foundation)):
            expected = branch.camera_adapter(feature, condition)
            with intervention(self.model, name):
                torch.testing.assert_close(branch.camera_adapter(feature, condition), feature, rtol=0, atol=0)
            torch.testing.assert_close(branch.camera_adapter(feature, condition), expected, rtol=0, atol=0)
            self.assertEqual(len(branch.camera_adapter._forward_hooks), 0)

    def test_late_gain_and_function_restore_after_exception(self):
        gain = self.model.decoder.late_gain.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "planned"):
            with intervention(self.model, "late_off"):
                self.assertEqual(int(torch.count_nonzero(self.model.decoder.late_gain)), 0)
                raise RuntimeError("planned")
        torch.testing.assert_close(self.model.decoder.late_gain, gain, rtol=0, atol=0)
        descriptor = encoding.camera_descriptor
        with self.assertRaisesRegex(RuntimeError, "planned"):
            with intervention(self.model, "baseline_zero"):
                raise RuntimeError("planned")
        self.assertIs(encoding.camera_descriptor, descriptor)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
