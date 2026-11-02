"""研究插拔回归：旧源码 zip 作独立参照，不改生产文件，不加载大模型。"""

import copy
from pathlib import Path
import sys
import types
import unittest
import zipfile

import torch

from jar.pretraining.model import CalibratedDecoder
from research.near_evidence_20261030.variants import install_variants


ROOT = Path(__file__).resolve().parents[2]


def archived_decoder(round_name):
    filename = ROOT / "runs/active/near_evidence_20261030" / f"{round_name}_source.zip"
    if not filename.exists():
        raise unittest.SkipTest(f"缺少历史源码参照：{filename}")
    with zipfile.ZipFile(filename) as archive:
        entry = next(name for name in archive.namelist() if name.endswith("jar/pretraining/model.py"))
        source = archive.read(entry).decode()
        helper_entry = next(name for name in archive.namelist() if name.endswith("jar/pretraining/ray_evidence.py"))
        helper_source = archive.read(helper_entry).decode()
    # helper 也独立使用历史内容，不能用待验证的新 helper 作自身参照。
    helper_name = f"_rulermvs_archived_{round_name}_ray_evidence"
    helper = types.ModuleType(helper_name)
    exec(compile(helper_source, str(filename) + "!" + helper_entry, "exec"), helper.__dict__)
    sys.modules[helper_name] = helper
    source = source.replace("from .ray_evidence import", f"from {helper_name} import")
    module = types.ModuleType(f"jar.pretraining._{round_name}_reference")
    module.__package__ = "jar.pretraining"
    exec(compile(source, str(filename) + "!" + entry, "exec"), module.__dict__)
    return module.CalibratedDecoder


def tiny_settings(**changes):
    return {"near": 2., "far": 8., "transition": 4., "depth_bins": 6, "channels": 8,
            "raw_channels": 16, "peak_count": 2, "use_scene": False,
            "local_kind": "ray", "proposal_transport": "bilinear", "compact_sources": True,
            "source_intervention_seed": 7, **changes}


def tiny_inputs():
    generator = torch.Generator().manual_seed(19)
    K = torch.tensor([[8., 0., 8.], [0., 8., 8.], [0., 0., 1.]])[None, None].repeat(1, 3, 1, 1)
    T = torch.eye(4)[None, None].repeat(1, 3, 1, 1)
    T[0, 1, 0, 3] = -.5
    T[0, 2, 0, 3] = .5
    return {
        "pyramid": {stride: torch.randn(1, 3, 8, 16 // stride, 16 // stride, generator=generator)
                    for stride in (4, 8, 16)},
        "raw_maps": [torch.randn(1, 768, 1, 1, generator=generator) for _ in range(4)],
        "K": K, "T_cam_from_rig": T, "view_mask": torch.ones(1, 3, dtype=torch.bool),
        "image_mask": torch.ones(1, 3, 16, 16, dtype=torch.bool),
        "matching_scale": torch.ones(1, 2), "reference_index": 0, "image_size": (16, 16),
    }


def objective(output):
    loss = output["depth"].square().mean()
    for name in ("coarse", "fine"):
        loss = loss + output[name]["scores"].square().mean()
    return loss


class VariantTests(unittest.TestCase):
    def compare_outputs(self, expected, actual):
        for name in ("depth", "confidence", "support", "valid"):
            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
        for stage in ("coarse", "fine"):
            for name in ("depth", "scores", "probability", "sampling_depth", "candidate_depth", "selected_index"):
                torch.testing.assert_close(expected[stage][name], actual[stage][name], rtol=0, atol=0)

    def compare_archived(self, round_name, flags, training):
        settings = tiny_settings(**flags)
        expected = archived_decoder(round_name)(settings)
        neutral = tiny_settings()
        actual = CalibratedDecoder(neutral)
        actual.load_state_dict(expected.state_dict(), strict=True)
        initial_state = {name: value.clone() for name, value in actual.state_dict().items()}
        parameter_ids = {name: id(value) for name, value in actual.named_parameters()}
        rng = torch.random.get_rng_state().clone()
        install_variants(actual, settings)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(parameter_ids, {name: id(value) for name, value in actual.named_parameters()})
        self.assertEqual(initial_state.keys(), actual.state_dict().keys())
        for name, value in actual.state_dict().items():
            torch.testing.assert_close(value, initial_state[name], rtol=0, atol=0)
        install_variants(actual, settings)  # 重复安装不能重置干预 RNG。
        expected.train(training)
        actual.train(training)
        expected.set_range(2., 8.)
        actual.set_range(2., 8.)
        inputs = tiny_inputs()
        for _ in range(2):
            first = expected(inputs)
            second = actual(inputs)
            self.compare_outputs(first, second)
            if training:
                objective(first).backward()
                objective(second).backward()
                for (name, left), (right_name, right) in zip(expected.named_parameters(), actual.named_parameters()):
                    self.assertEqual(name, right_name)
                    if left.grad is None:
                        self.assertIsNone(right.grad)
                    else:
                        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
                expected.zero_grad(set_to_none=True)
                actual.zero_grad(set_to_none=True)

    def test_disabled_install_is_identity(self):
        model = CalibratedDecoder(tiny_settings()).eval()
        copy_model = copy.deepcopy(model)
        source_class, decoder_class = type(model.source), type(model)
        rng = torch.random.get_rng_state().clone()
        self.assertIs(install_variants(model), model)
        self.assertIs(type(model), decoder_class)
        self.assertIs(type(model.source), source_class)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.compare_outputs(copy_model(tiny_inputs()), model(tiny_inputs()))

    def test_round1_relative_forward_and_backward(self):
        self.compare_archived("round1", {"source_relative_evidence": True}, training=True)

    def test_production_default_matches_archived_control(self):
        self.compare_archived("round2", {}, training=True)

    def test_round1_intervention_forward_and_backward(self):
        self.compare_archived("round1", {"source_intervention_probability": 1.}, training=True)

    def test_round2_coarse_attention_forward_and_backward(self):
        self.compare_archived("round2", {"source_relative_coarse_attention": True}, training=True)

    def test_round1_intervention_disabled_in_eval(self):
        self.compare_archived("round1", {"source_intervention_probability": 1.}, training=False)

    def test_round2_combined_flags(self):
        self.compare_archived("round2", {"source_relative_evidence": True,
                                        "source_relative_coarse_attention": True,
                                        "source_intervention_probability": 1.}, training=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
