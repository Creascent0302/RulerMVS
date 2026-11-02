"""Preflight failure paths run on CPU; CUDA execution is explicitly skipped here
when unavailable, never replaced by an allegedly equivalent CPU success.
"""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from tools.check_gpu import (
    collect_preflight,
    compatibility_notes,
    kernel_smoke,
    main as preflight_main,
)
from tools.gpu_acceptance import model_case, unlabeled_fixture, main as acceptance_main


def fake_torch(count=0, available=False, arches=None):
    cuda = SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: count,
        get_arch_list=lambda: ["sm_120"] if arches is None else arches,
        get_device_properties=lambda index: SimpleNamespace(
            name=f"NVIDIA RTX 5090 #{index}", total_memory=32 * 1024**3
        ),
        get_device_capability=lambda index: (12, 0),
    )
    return SimpleNamespace(
        cuda=cuda, __version__="2.7.0+cu128", version=SimpleNamespace(cuda="12.8")
    )


def passed_worker(index, timeout=0):
    return {"status": "passed", "fp32": {"status": "passed"}, "bf16": {"status": "passed"}}


class GPUPreflightLogicTests(unittest.TestCase):
    def test_unlabeled_disk_fixture_has_calibration_without_targets(self):
        from jar.data import CarlaFrameDataset

        with tempfile.TemporaryDirectory() as temporary:
            directory = unlabeled_fixture(temporary)
            frame = json.loads((directory / "manifest.jsonl").read_text())
            self.assertTrue(all("targets" not in camera for camera in frame["cameras"]))
            dataset = CarlaFrameDataset(
                str(directory), "manifest.jsonl", split="test", load_targets=False
            )
            sample = dataset[0]
            self.assertEqual(sample["targets"], {})
            self.assertEqual(tuple(sample["inputs"]["rgb"].shape), (3, 3, 32, 48))
            self.assertEqual(tuple(sample["inputs"]["T_cam_from_rig"].shape), (3, 4, 4))

    def test_no_cuda_does_not_launch_or_certify(self):
        def forbidden(*args, **kwargs):
            self.fail("No-CUDA metadata must not launch a GPU worker")

        report = collect_preflight(8, torch_module=fake_torch(), worker=forbidden, smi=lambda: {})
        self.assertEqual(report["status"], "not_verified")
        self.assertEqual(report["visible_gpu_count"], 0)
        self.assertEqual(report["devices"], [])
        self.assertIn("不能认证", "".join(report["messages_zh"]))

    def test_kernel_failure_overrides_available_and_sm120(self):
        report = collect_preflight(
            8,
            torch_module=fake_torch(8, True),
            smi=lambda: {},
            worker=lambda index, timeout=0: (
                passed_worker(index)
                if index != 6
                else {"status": "failed", "error": "no kernel image"}
            ),
        )
        self.assertTrue(report["cuda_available"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["devices"]), 8)

    def test_count_and_bf16_are_enforced(self):
        report = collect_preflight(
            8, torch_module=fake_torch(2, True), worker=passed_worker, smi=lambda: {}
        )
        self.assertEqual(report["status"], "failed")
        report = collect_preflight(
            1,
            require_bf16=True,
            torch_module=fake_torch(1, True),
            smi=lambda: {},
            worker=lambda index, timeout=0: {"status": "passed", "bf16": {"status": "unsupported"}},
        )
        self.assertEqual(report["status"], "failed")

    def test_ptx_execution_can_pass_without_native_arch_metadata(self):
        report = collect_preflight(
            1,
            torch_module=fake_torch(1, True, ["sm_90", "compute_90"]),
            worker=passed_worker,
            smi=lambda: {},
        )
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["devices"][0]["notes_zh"])
        notes = compatibility_notes("RTX 5090", [12, 0], ["sm_90"], "2.6.0+cu124", "12.4")
        self.assertTrue(any("旧 PyTorch" in note for note in notes))

    def test_cli_no_gpu_is_nonzero_and_writes_clear_report(self):
        missing = collect_preflight(8, torch_module=fake_torch(), smi=lambda: {})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preflight.json"
            with (
                patch("tools.check_gpu.collect_preflight", return_value=missing),
                patch("builtins.print"),
            ):
                code = preflight_main(["--require-gpus", "8", "--output", str(path)])
            self.assertNotEqual(code, 0)
            self.assertEqual(json.loads(path.read_text())["status"], "not_verified")
            with (
                patch("tools.gpu_acceptance.collect_preflight", return_value=missing),
                patch("builtins.print"),
            ):
                code = acceptance_main(["--require-gpus", "8", "--output", temporary])
            self.assertNotEqual(code, 0)
            result = json.loads((Path(temporary) / "report.json").read_text())
            self.assertEqual(result["model_cases"], [])
            self.assertEqual(result["ddp_cases"], [])
            self.assertEqual(result["status"], "not_verified")


@unittest.skipUnless(
    torch.cuda.is_available(), "CUDA 不可用：实际 GPU 内核、模型和 BF16 验收未执行"
)
class ActualCUDAAcceptanceTests(unittest.TestCase):
    def test_each_visible_device_actual_kernels(self):
        for device in range(torch.cuda.device_count()):
            for precision in ("fp32", "bf16"):
                with self.subTest(device=device, precision=precision):
                    result = kernel_smoke(device, precision)
                    self.assertEqual(result["status"], "passed")
                    self.assertEqual(len(result["kernels"]), 5)

    def test_final_model_camera_counts_and_precisions(self):
        for cameras in (2, 3, 4):
            for precision in ("fp32", "bf16"):
                with self.subTest(cameras=cameras, precision=precision):
                    result = model_case("mf_et_jar", cameras, precision, warmup=0, repeats=1)
                    self.assertEqual(result["status"], "passed")
                    self.assertEqual(result["optimizer_steps"], 1)
                    self.assertGreater(result["inference_peak_allocated_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
