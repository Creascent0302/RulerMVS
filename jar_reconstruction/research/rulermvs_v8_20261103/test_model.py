"""CPU mathematical/metamorphic checks, with real ResNet and tiny ViT fixture.

These do not measure accuracy or latency and never load evaluation datasets.
"""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from jar.models.metric_ruler_network import depth_cell_edges
from jar.pretraining.model import CandidateViewAttention
from research.early_geometry_20261030.test_encoding import TinyViT, inputs
from research.rulermvs_v6_20261101.model import V6Model
from research.rulermvs_v6_20261101.objective import objective
from research.rulermvs_v6_20261101.training import optimizer_for
from research.rulermvs_v8_20261103.model import V8Model, V8Decoder, V8SourceAttention, validate_options


ROOT = Path(__file__).resolve().parents[2]
ACTIVE = {"information_mode": "log_depth", "information_scale": 10.,
          "source_count_strength": 1., "relative_validity": True, "physical_thresholds": True}


def tiny_config():
    config = json.loads((ROOT / "research/v2_restart_20261029/configs/train_public.json").read_text())
    config["model"].update(channels=8, raw_channels=8, depth_bins=8, peak_count=2,
                           compact_sources=True)
    config["bridge"] = {"bottleneck": 4, "hidden": 8}
    config["handoff"] = {"mode": "parent"}
    return config


def make_model(options=None, v6=False):
    config = tiny_config()
    config["v8"] = dict(options or {})
    config["architecture"] = "rulermvs_v6" if v6 else "rulermvs_v8"
    with patch("jar.pretraining.encoders._build_native_foundation", side_effect=TinyViT):
        model = (V6Model if v6 else V8Model)(config, initialize=False).eval()
    return model


def source_inputs(views=3):
    reference = F.normalize(torch.randn(1, 8, 2, 3), dim=1)
    source = F.normalize(torch.randn(1, views, 8, 4, 2, 3), dim=2)
    valid = torch.ones(1, views, 4, 2, 3, dtype=torch.bool)
    derivative = torch.randn(1, views, 2, 4, 2, 3)
    half = torch.linspace(.1, .4, 4)
    return reference, source, valid, derivative, half


class SourceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)
        torch.set_num_threads(2)

    def new_source(self, strength):
        module = V8SourceAttention(8).eval()
        module.configure(validate_options({"source_count_strength": strength}))
        # Nonzero head-specific logits test the actual four-head normalization.
        torch.nn.init.normal_(module.reliability.weight, std=.2)
        torch.nn.init.normal_(module.null.weight, std=.2)
        return module

    def test_zero_switch_exact_legacy_including_fast_flags(self):
        old = CandidateViewAttention(8).eval()
        new = self.new_source(0.)
        new.load_state_dict(old.state_dict(), strict=True)
        args = source_inputs()
        for hoist in (False, True):
            for compact in (False, True):
                old.hoist_reference = new.hoist_reference = hoist
                old.compact_pair = new.compact_pair = compact
                expected, actual = old(*args), new(*args)
                for key in expected:
                    self.assertTrue(torch.equal(expected[key], actual[key]), (hoist, compact, key))

    def test_three_valid_sources_preserve_original_and_nonzero_strength_is_active(self):
        new = self.new_source(1.)
        old = CandidateViewAttention(8).eval()
        old.load_state_dict(new.state_dict(), strict=True)
        args = source_inputs()
        for hoist in (False, True):
            for compact in (False, True):
                old.hoist_reference = new.hoist_reference = hoist
                old.compact_pair = new.compact_pair = compact
                expected, actual = old(*args), new(*args)
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        single = (args[0], args[1][:, :1], args[2][:, :1], args[3][:, :1], args[4])
        self.assertFalse(torch.equal(old(*single)["null_mass"], new(*single)["null_mass"]))

    def test_full_source_set_replication_with_candidate_masks_and_fast_flags(self):
        module = self.new_source(1.)
        reference, source, valid, derivative, half = source_inputs()
        valid[:, 0, 0] = False
        valid[:, :, 1, 0, 0] = False
        repeated = (reference, source.repeat(1, 2, 1, 1, 1, 1), valid.repeat(1, 2, 1, 1, 1),
                    derivative.repeat(1, 2, 1, 1, 1, 1), half)
        for hoist in (False, True):
            for compact in (False, True):
                module.hoist_reference, module.compact_pair = hoist, compact
                original = module(reference, source, valid, derivative, half)
                actual = module(*repeated)
                for key in ("pooled", "null_mass", "information"):
                    torch.testing.assert_close(actual[key], original[key], rtol=2e-6, atol=2e-7)
                summed_weights = actual["weights"][:, :3] + actual["weights"][:, 3:]
                torch.testing.assert_close(summed_weights, original["weights"], rtol=2e-6, atol=2e-7)

    def test_all_invalid_and_gradient_are_finite(self):
        module = self.new_source(1.)
        reference, source, valid, derivative, half = source_inputs()
        valid.zero_()
        # Match _observe's producer contract: invalid descriptors are finite 0.
        source.zero_()
        derivative.fill_(float("nan"))
        result = module(reference, source, valid, derivative, half)
        self.assertTrue(torch.equal(result["null_mass"], torch.ones_like(result["null_mass"])))
        self.assertTrue(torch.equal(result["information"], torch.zeros_like(result["information"])))
        self.assertTrue(torch.equal(result["pooled"], torch.zeros_like(result["pooled"])))
        result["pooled"].sum().backward()
        self.assertTrue(all(p.grad is None or bool(p.grad.isfinite().all()) for p in module.parameters()))

    def test_torch_compile_eager_backend_keeps_correction(self):
        module = self.new_source(1.)
        module.hoist_reference = module.compact_pair = True
        args = source_inputs(views=1)
        expected = module(*args)
        compiled = torch.compile(module, backend="eager", fullgraph=True, dynamic=False)
        actual = compiled(*args)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(84)
        self.domain = {"near": 1., "far": 10., "depth_scale": 1.}

    def test_zero_options_exact_complete_v6_and_strict_state(self):
        old, new = make_model(v6=True), make_model()
        new.load_v6_state(old.state_dict())
        self.assertEqual(set(new.state_dict()), set(old.state_dict()))
        self.assertEqual(sum(p.numel() for p in new.parameters()), sum(p.numel() for p in old.parameters()))
        data = inputs()
        with torch.no_grad():
            expected, actual = old(data, self.domain), new(data, self.domain)
        for key in ("depth", "valid", "confidence", "support", "unknown", "state"):
            self.assertTrue(torch.equal(actual[key], expected[key]), key)
        for stage in ("coarse", "fine"):
            for key, value in expected[stage].items():
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(actual[stage][key], value), (stage, key))

    def test_log_information_unit_transform_and_normalization_constants(self):
        settings = tiny_config()["model"]
        decoder = V8Decoder(settings, {"mode": "parent"}, {"information_mode": "log_depth"}).eval()
        reference, source, valid, derivative, half = source_inputs()
        observation = dict(reference=reference, source=source, source_valid=valid,
                           ray_derivative=derivative, cell_half_width=half,
                           ref_valid=torch.ones(1, 1, 2, 3, dtype=torch.bool))
        prepared = {"raw": [torch.randn(1, 8, 2, 3) for _ in range(4)]}
        candidate = torch.tensor([1., 2., 4., 8.])[None, :, None, None].expand(1, 4, 2, 3)
        for norm in (1., 10.):
            decoder.v8_options["information_scale"] = norm
            expected = None
            for scale in (.25, .5, 1., 2., 4.):
                decoder.set_range(1. / scale, 10. / scale)
                observed = dict(observation, ray_derivative=derivative * scale, cell_half_width=half / scale)
                result = decoder._tokens(prepared, observed, candidate / scale, 8)
                if expected is None:
                    expected = result
                for index in (0, 3):
                    torch.testing.assert_close(result[index], expected[index], rtol=3e-6, atol=3e-6)
                torch.testing.assert_close(result[-1]["information"], expected[-1]["information"], rtol=0, atol=0)
        # No claim that 10 is empirically optimal: both satisfy the same algebra.

    def test_private_thresholds_do_not_accumulate_or_modify_config(self):
        config = tiny_config()
        config["v8"] = dict(ACTIVE)
        pristine = copy.deepcopy(config)
        with patch("jar.pretraining.encoders._build_native_foundation", side_effect=TinyViT):
            model = V8Model(config, initialize=False)
        for scale in (4., .25, 2., 1.):
            model.decoder.set_unit_scale(scale)
            for name, value in model.decoder._physical_settings.items():
                self.assertEqual(model.decoder.settings[name] * scale, value)
        self.assertEqual(config, pristine)
        self.assertEqual(model.config, pristine)
        self.assertIsNot(model.decoder.settings, model.config["model"])

    def test_nominal_threshold_contract_and_input_compatibility(self):
        model = make_model(dict(ACTIVE, proposal_scale_reference="nominal"))
        original_config = copy.deepcopy(model.config)
        for nominal in (26.625, .23671875):
            # Missing nominal_scale declares an unjittered domain, preserving V6.
            model.decoder.set_unit_scale(nominal)
            self.assertEqual({key: model.decoder.settings[key]
                              for key in model.decoder._physical_settings},
                             model.decoder._physical_settings)
            for factor in (.5, 1., 2.):
                actual = nominal * factor
                model.decoder.set_unit_scale(actual, nominal)
                for name, configured in model.decoder._physical_settings.items():
                    self.assertEqual(model.decoder.settings[name], configured / factor)
                    self.assertEqual(model.decoder.settings[name] * actual, configured * nominal)
        self.assertEqual(model.config, original_config)
        model.decoder.v8_options["proposal_scale_reference"] = "input"
        model.decoder.set_unit_scale(26.625, 123.)
        for name, configured in model.decoder._physical_settings.items():
            self.assertEqual(model.decoder.settings[name], configured / 26.625)
        with self.assertRaises(ValueError):
            validate_options({"proposal_scale_reference": "inferred"})
        model.decoder.v8_options["proposal_scale_reference"] = "nominal"
        for invalid in (0., float("nan")):
            with self.assertRaises(ValueError):
                model.decoder.set_unit_scale(1., invalid)

    def test_nominal_no_jitter_preserves_complete_v6_output(self):
        old = make_model(v6=True)
        new = make_model({"physical_thresholds": True, "proposal_scale_reference": "nominal"})
        new.load_v6_state(old.state_dict())
        data = inputs()
        # A non-unit nominal scale catches accidental input-unit threshold use.
        domain = dict(self.domain, depth_scale=26.625)
        with torch.no_grad():
            expected, actual = old(data, domain), new(data, domain)
        for stage in ("coarse", "fine"):
            for key, value in expected[stage].items():
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(actual[stage][key], value), (stage, key))
        self.assertTrue(torch.equal(actual["depth"], expected["depth"]))

    def test_nominal_jitter_forward_encode_and_shared_references(self):
        model = make_model(dict(ACTIVE, proposal_scale_reference="nominal"))
        data = inputs()
        nominal = 2.
        base = dict(self.domain, depth_scale=nominal, nominal_depth_scale=nominal)
        with torch.no_grad():
            expected = model(data, base)
            shared_expected = model.forward_references(data, base, [0, 1])
            for factor in (.5, 2.):
                domain = dict(base, depth_scale=nominal * factor)
                preserved = dict(domain)
                model.encode(data, domain)
                for name, configured in model.decoder._physical_settings.items():
                    self.assertEqual(model.decoder.settings[name], configured / factor)
                actual = model(data, domain)
                shared = model.forward_references(data, domain, [0, 1])
                self.assertEqual(domain, preserved)
                for stage in ("coarse", "fine"):
                    self.assertTrue(torch.equal(actual[stage]["source_valid"], expected[stage]["source_valid"]))
                    torch.testing.assert_close(actual[stage]["candidate_depth"] * factor,
                                               expected[stage]["candidate_depth"], rtol=2e-5, atol=2e-5)
                torch.testing.assert_close(actual["depth"] * factor, expected["depth"], rtol=2e-5, atol=2e-5)
                for reference in (0, 1):
                    torch.testing.assert_close(shared[reference]["depth"] * factor,
                                               shared_expected[reference]["depth"], rtol=2e-5, atol=2e-5)

    def test_full_forward_unit_transform_and_fast_fusion(self):
        model = make_model(ACTIVE)
        data = inputs()
        for fast in (False, True):
            model.decoder.settings["factorized_fusion"] = fast
            model.decoder.source.hoist_reference = fast
            model.decoder.source.compact_pair = fast
            with torch.no_grad():
                expected = model(data, self.domain)
                for scale in (.25, .5, 2., 4.):
                    actual = model(data, dict(self.domain, depth_scale=scale))
                    for stage in ("coarse", "fine"):
                        self.assertTrue(torch.equal(actual[stage]["source_valid"], expected[stage]["source_valid"]))
                        # FP32 projection, logspace and discrete ranks are not
                        # bitwise scale equivariant; test the resulting numbers.
                        torch.testing.assert_close(actual[stage]["scores"], expected[stage]["scores"], rtol=2e-4, atol=1e-4)
                        torch.testing.assert_close(actual[stage]["candidate_depth"] * scale,
                                                   expected[stage]["candidate_depth"], rtol=2e-5, atol=2e-5)
                    torch.testing.assert_close(actual["depth"] * scale, expected["depth"], rtol=2e-5, atol=2e-5)

    def test_relative_mask_threshold_is_invariant_and_pure_rotation_is_invalid(self):
        model = make_model(ACTIVE)
        data = inputs()
        data["T_cam_from_rig"].zero_()
        data["T_cam_from_rig"][:] = torch.eye(4)
        # |J| is around 1e-5 at Z=10 and crosses the old cutoff under units;
        # |J|Z stays on the same side of the new 1e-5 cutoff.
        data["T_cam_from_rig"][0, 1, 0, 3] = -2e-5
        with torch.no_grad():
            work = model.encode(data, self.domain)
            expected_mask = None
            for fast_projection in (False, True):
                model.decoder.settings["projective_ruler"] = fast_projection
                expected = None
                for scale in (.25, 1., 4.):
                    prepared = model.decoder._prepare(work)
                    prepared["T"] = work["T_cam_from_rig"].clone()
                    prepared["T"][..., :3, 3] /= scale
                    candidates = torch.full((1, 1, 4, 6), 10. / scale)
                    observation = model.decoder._observe(prepared, candidates, 8, torch.tensor([.2 / scale]))
                    self.assertFalse(bool(observation["source_valid"][:, 1].any()))
                    self.assertTrue(bool(observation["source_valid"][:, 0, :, :, 1:].all()))
                    if expected is None:
                        expected = observation
                    if expected_mask is None:
                        expected_mask = observation["source_valid"]
                    self.assertTrue(torch.equal(observation["source_valid"], expected_mask))
                    torch.testing.assert_close(observation["ray_derivative"] / scale,
                                               expected["ray_derivative"] / .25, rtol=1e-4, atol=1e-7)

    def test_backward_stage_optimizer_and_checkpoint_reload(self):
        model = make_model(ACTIVE)
        # Existing stage API selects exact original prefixes; no new optimizer group.
        model.set_stage(model.config["stages"][0]["number"])
        model.requires_grad_(False)
        model.decoder.requires_grad_(True)
        model.decoder.train()
        stage = {name + "_lr": 1e-4 for name in ("handoff", "bridge", "foundation", "matching", "fpn", "decoder")}
        optimizer = optimizer_for(model, stage, {"train": {"weight_decay": 0.}})
        data = inputs()
        targets = {"depth": torch.full_like(data["image_mask"], 3., dtype=torch.float32),
                   "depth_valid": data["image_mask"].clone()}
        output = model(data, self.domain)
        loss = objective(output, data, targets, self.domain, {})["total"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or bool(p.grad.isfinite().all()) for p in model.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.decoder.source.parameters() if p.grad is not None), 0.)
        optimizer.step()
        model.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        model.eval()
        with torch.no_grad():
            shared = model.forward_references(data, self.domain, [0, 1])
            regular = model(data, self.domain, 0)
        torch.testing.assert_close(shared[0]["depth"], regular["depth"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
