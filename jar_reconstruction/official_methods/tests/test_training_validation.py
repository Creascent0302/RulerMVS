"""验证新增训练验收逻辑，使用完整原生模型和真实几何合成图。"""

import copy
from types import SimpleNamespace

import pytest
import torch

from jar.runtime import setup_runtime
from official_methods.cli import train
from official_methods.registry import build_adapter
from official_methods.training_validation import validate_depth


def configuration(method):
    return {
        "method": method, "seed": 27, "synthetic_length": 2,
        "image_size": [64, 96], "data": {"num_views": 4, "geometry_units": "meters"},
        "model": {"allow_random_init": True, "weights": "", "reference_indices": [0],
                  "depth_range": [2., 80.]},
        "training": {"epochs": 2, "max_steps": 2, "batch_size": 1, "num_workers": 0,
                     "validation_every_epochs": 1, "scheduler_interval": "step"},
        "evaluation": {"reference_indices": [0], "depth_range": [2., 80.]},
    }


@pytest.mark.parametrize("method", ["patchmatchnet", "diffmvs"])
def test_validation_preserves_training_rng_and_buffers(method):
    torch.set_num_threads(2)
    cfg = configuration(method)
    adapter = build_adapter(cfg).train()
    state = torch.get_rng_state().clone()
    buffers = {name: value.clone() for name, value in adapter.named_buffers()}
    result = validate_depth(adapter, cfg, setup_runtime("cpu"), synthetic=True)
    assert adapter.training and torch.equal(state, torch.get_rng_state())
    assert all(torch.equal(value, buffers[name]) for name, value in adapter.named_buffers())
    # 随机初始化的原生细化器允许非正深度，必须如实报告覆盖率。
    assert result["frames"] == 2 and 0 <= result["depth"]["prediction_coverage"] <= 1
    assert result["selection_score"] is not None


@pytest.mark.parametrize("method", ["patchmatchnet", "diffmvs"])
def test_training_best_final_resume_completed_job(method, tmp_path):
    torch.set_num_threads(2)
    cfg = configuration(method)
    args = SimpleNamespace(synthetic=True, checkpoint="", output=str(tmp_path))
    result = train(args, copy.deepcopy(cfg), setup_runtime("cpu"))
    assert result["optimizer_steps"] == 2 and result["sample_exposures"] == 2
    assert result["best_score"] is not None
    for filename in ("best.pt", "last.pt", "final.pt"):
        assert (tmp_path / filename).is_file()
    original = torch.load(tmp_path / "final.pt", weights_only=False)
    args.checkpoint = str(tmp_path / "last.pt")
    cfg["_loading_shared_checkpoint"] = True
    resumed = train(args, cfg, setup_runtime("cpu"))
    restored = torch.load(tmp_path / "final.pt", weights_only=False)
    assert resumed["optimizer_steps"] == result["optimizer_steps"]
    assert original["training_state"] == restored["training_state"]
    assert all(torch.equal(value, restored["model"][key]) for key, value in original["model"].items())
