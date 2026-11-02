import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from jar.runtime import (
    setup_runtime,
    cleanup_runtime,
    precision_context,
    make_scaler,
    move,
    capture_rng,
    restore_rng,
)


class RuntimeTests(unittest.TestCase):
    def test_cpu_runtime_precision_move_and_rng(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}):
            rt = setup_runtime("cpu")
            self.assertTrue(rt.is_main)
            self.assertFalse(rt.distributed)
            state = capture_rng(rt)
            expected = torch.rand(5)
            restore_rng(state, rt)
            torch.testing.assert_close(torch.rand(5), expected, rtol=0, atol=0)
            with precision_context(rt.device, "bf16"):
                self.assertEqual((torch.ones(2, 2) @ torch.ones(2, 2)).dtype, torch.bfloat16)
                with precision_context(rt.device, "fp32"):
                    self.assertEqual((torch.ones(2, 2) @ torch.ones(2, 2)).dtype, torch.float32)
            self.assertFalse(make_scaler(rt.device, "bf16").is_enabled())
            with self.assertRaisesRegex(ValueError, "CUDA"):
                precision_context(rt.device, "fp16")
            data = move({"a": [torch.ones(1), (torch.zeros(1),)]}, rt.device, non_blocking=True)
            self.assertEqual(data["a"][1][0].device.type, "cpu")
            cleanup_runtime(rt)

    def test_explicit_unavailable_cuda_is_not_silent_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                setup_runtime("cuda")

    @unittest.skipUnless(
        os.environ.get("JAR_RUN_DDP_TESTS") == "1", "Opt-in local gloo sockets: JAR_RUN_DDP_TESTS=1"
    )
    def test_real_two_rank_dynamic_models_and_exact_resume(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=2",
                    "tests/ddp_worker.py",
                    "--output",
                    temporary,
                ],
                cwd=project,
                env={**os.environ, "OMP_NUM_THREADS": "2", "PYTHONPATH": str(project)},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("DDP_TEST_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
