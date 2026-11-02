"""CPU测试只证明开关隔离/可训练；精度由真实TRAIN/VAL实验另验。"""

import copy
import unittest
from unittest.mock import patch

import torch

from research.early_geometry_20261030.test_encoding import TinyViT, inputs
from research.rulermvs_v5_20261031.test_bridge import tiny_pair
from research.rulermvs_v6_20261101.model import build_from_checkpoint
from research.rulermvs_v6_20261101.objective import objective
from research.paper_ablation_20261102.geometry import SETTINGS, build_geometry_model


class GeometryAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(20261102)
        _, v5 = tiny_pair()
        cls.saved = {
            "architecture": "rulermvs_v5",
            "config": v5.config,
            "model": v5.state_dict(),
        }
        cls.domain = {"near": 1.0, "far": 10.0, "depth_scale": 1.0}

    def model(self, ablation):
        torch.manual_seed(20261102)
        with patch("jar.pretraining.encoders._build_native_foundation", side_effect=TinyViT):
            model, _ = build_geometry_model(self.saved, ablation)
        return model.eval()

    def test_control_matches_unmodified_v6(self):
        control = self.model("control")
        torch.manual_seed(20261102)
        with patch("jar.pretraining.encoders._build_native_foundation", side_effect=TinyViT):
            native, _ = build_from_checkpoint(self.saved, "parent")
        native.eval()
        for key, value in native.state_dict().items():
            self.assertTrue(torch.equal(value, control.state_dict()[key]), key)
        data = inputs()
        with torch.no_grad():
            expected = native(data, self.domain)
            actual = control(data, self.domain)
        for key in ("depth", "valid", "confidence", "support"):
            self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def test_geometry_and_route_keep_projection_and_masks(self):
        data = inputs()
        control = self.model("control")
        with torch.no_grad():
            control.decoder.set_range(1.0, 10.0)
            prepared = control.decoder._prepare(control.encode(data, self.domain))
            candidates = control.decoder.depth_lattice[None, :, None, None].expand(1, -1, 4, 6)
            widths = torch.full((8,), 1.5)
            expected = control.decoder._observe(prepared, candidates, 8, widths)
            for variant in ("no_source_geometry", "no_scale_routing"):
                model = self.model(variant)
                actual = model.decoder._observe(prepared, candidates, 8, widths)
                for key in ("source_valid", "ref_valid", "ray_derivative", "reference"):
                    self.assertTrue(torch.equal(expected[key], actual[key]), (variant, key))
                if variant == "no_source_geometry":
                    self.assertTrue(torch.equal(expected["source"], actual["source"]))
                else:
                    self.assertFalse(torch.equal(expected["source"], actual["source"]))

    def test_radius_only_changes_fine_candidates(self):
        data = inputs()
        with torch.no_grad():
            before = self.model("control")(data, self.domain)
            after = self.model("no_jacobian_width")(data, self.domain)
        for key in ("depth", "candidate_depth", "scores", "valid"):
            self.assertTrue(torch.equal(before["coarse"][key], after["coarse"][key]), key)
        self.assertEqual(before["fine"]["candidate_depth"].shape,
                         after["fine"]["candidate_depth"].shape)
        self.assertFalse(torch.equal(before["fine"]["candidate_depth"],
                                     after["fine"]["candidate_depth"]))

    def test_each_variant_trains_and_restores_without_label_input(self):
        for ablation in SETTINGS:
            with self.subTest(ablation=ablation):
                model = self.model(ablation)
                model.requires_grad_(False)
                model.decoder.requires_grad_(True)
                model.decoder.train()
                data = inputs()
                data["view_mask"][:, -1] = False
                data["rgb"][:, -1] = float("nan")
                data["matching_rgb"][:, -1] = float("nan")
                targets = {
                    "depth": torch.full_like(data["image_mask"], 3.0, dtype=torch.float32),
                    "depth_valid": data["image_mask"].clone(),
                }
                prediction = model(data, self.domain)
                terms = objective(prediction, data, targets, self.domain, {})
                terms["total"].backward()
                grads = [p.grad for p in model.decoder.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertTrue(any((g != 0).any() for g in grads))
                model.eval()
                with torch.no_grad():
                    expected = model(data, self.domain)["depth"]
                record = {"architecture": "rulermvs_v6", "config": copy.deepcopy(model.config),
                          "model": model.state_dict()}
                with patch("jar.pretraining.encoders._build_native_foundation", side_effect=TinyViT):
                    restored, _ = build_geometry_model(record, ablation)
                restored.eval()
                targets["depth"].mul_(7)
                with torch.no_grad():
                    actual = restored(data, self.domain)["depth"]
                self.assertTrue(torch.equal(expected, actual))
                self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()
