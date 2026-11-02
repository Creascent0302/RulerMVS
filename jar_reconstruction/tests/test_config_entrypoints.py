"""验证文件配置确实控制训练、续训和推理，不再受旧 CLI 参数覆盖。"""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from jar.config import load_config
from jar.engine import build_parser

ROOT = Path(__file__).resolve().parents[1]


class ConfigEntrypointTests(unittest.TestCase):
    def test_old_overrides_are_rejected(self):
        removed = (
            ["--steps", "2"],
            ["--precision", "bf16"],
            ["--tiny"],
            ["--camera-mode", "wide4"],
            ["--no-multifocal"],
            ["--disable-cameras", "tele_left"],
        )
        for mode in ("train", "eval", "infer"):
            for arguments in removed:
                with self.subTest(mode=mode, arguments=arguments):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        build_parser(mode).parse_args(arguments)

    def test_invalid_training_settings_fail_when_loading_file(self):
        for section, key, value in (
            ("train", "max_steps", 0),
            ("train", "precision", "invalid"),
            ("train", "amp", True),
            ("runtime", "threads", 0),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps({section: {key: value}}))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_train_resume_infer_and_eval_use_file_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = load_config(ROOT / "configs/unit_tiny.json")
            cfg["synthetic"].update(num_cameras=3, length=4)
            cfg["train"].update(batch_size=2, accumulation_steps=2, max_steps=1)
            cfg["eval"].update(precision="bf16", warmup=1)
            path = root / "config.json"
            path.write_text(json.dumps(cfg))

            def run(script, destination, *arguments):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / script),
                        "--config",
                        str(path),
                        "--device",
                        "cpu",
                        "--synthetic",
                        "--output",
                        str(destination),
                        *arguments,
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            training = root / "train"
            checkpoint = training / "last.pt"
            run("train.py", training)
            runtime = json.loads((training / "runtime.json").read_text())
            self.assertEqual(runtime["batch_size_per_rank"], 2)
            self.assertEqual(runtime["accumulation_steps"], 2)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved["step"], 1)
            self.assertEqual(saved["config"]["synthetic"]["num_cameras"], 3)

            cfg["train"]["max_steps"] = 2
            path.write_text(json.dumps(cfg))
            run("train.py", training, "--resume", str(checkpoint))
            self.assertEqual(torch.load(checkpoint, weights_only=True)["step"], 2)
            for mode, report in (("infer", "inference"), ("eval", "evaluation")):
                run(f"{mode}.py", root / mode, "--checkpoint", str(checkpoint), "--limit", "1")
                summary = json.loads((root / mode / f"{report}.json").read_text())
                self.assertEqual(summary["precision"], "bf16")
                self.assertEqual(summary["warmup_per_rank"], 1)
                self.assertEqual(summary["labels_loaded"], mode == "eval")
                self.assertEqual(len(summary["samples"]), 1)


if __name__ == "__main__":
    unittest.main()
