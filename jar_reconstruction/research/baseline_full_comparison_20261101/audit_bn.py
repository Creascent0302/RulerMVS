"""固定每帧随机流，对照历史混合BN验证与真实rank0 checkpoint验证。"""

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import write_json
from jar.runtime import cleanup_runtime, setup_runtime
from official_methods.registry import build_adapter
from official_methods.training_validation import validate_depth


def main():
    output = ROOT / "runs/active/baseline_full_comparison_20261101"
    source = output / "training_validation_before_bn_fix.py"
    spec = importlib.util.spec_from_file_location("official_methods._historical_validation", source)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    runtime = setup_runtime("cuda")
    torch.set_num_threads(2)
    reports = {}
    try:
        for method in ("patchmatchnet", "diffmvs"):
            path = ROOT / f"runs/active/v5_comparison_20261101/{method}_adapt/training/best.pt"
            saved = torch.load(path, map_location="cpu", weights_only=False)
            config = dict(saved["config"], _loading_shared_checkpoint=True)
            adapter = build_adapter(config).to(runtime.device).train()
            adapter.load_state_dict(saved["model"], strict=True)
            with torch.no_grad():
                for name, buffer in adapter.named_buffers():
                    buffer.copy_(saved["state_by_rank"][runtime.rank]["buffers"][name].to(runtime.device))
            before = {name: buffer.clone() for name, buffer in adapter.named_buffers()}
            historical = previous.validate_depth(adapter, config, runtime)
            actual = validate_depth(adapter, config, runtime)
            if not all(torch.equal(buffer, before[name]) for name, buffer in adapter.named_buffers()):
                raise RuntimeError("验证修改了训练rank的BN状态")
            reports[method] = {
                "historical_mixed_buffers": historical, "single_saved_checkpoint": actual,
                "original_selection_score": saved["training_state"]["best_score"],
                "same_sample_rng_and_data": True, "local_buffers_restored": True,
            }
            del adapter, saved
        if runtime.is_main:
            write_json(output / "bn_audit.json", reports)
            print(json.dumps({key: {"old": value['historical_mixed_buffers']['selection_score'],
                                   "fixed": value['single_saved_checkpoint']['selection_score']}
                              for key, value in reports.items()}), flush=True)
    finally:
        cleanup_runtime(runtime)


if __name__ == "__main__":
    main()
