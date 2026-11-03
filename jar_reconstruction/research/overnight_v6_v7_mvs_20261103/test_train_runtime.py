"""独立 CPU 训练状态机检查；真实 GPU/DDP 冒烟另行执行。"""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from research.overnight_v6_v7_mvs_20261103 import common_data
from research.overnight_v6_v7_mvs_20261103 import train


class ToyDataset:
    """小尺寸确定性数据，用于执行原训练循环而非模拟保存/恢复逻辑。"""

    def __init__(self, config, split):
        self.domains = config["datasets"]
        self.datasets = [list(range(4))]

    def __len__(self):
        return 4

    def __getitem__(self, key):
        domain, index, epoch = key
        return {
            "domain_index": domain,
            "inputs": {
                "rgb": torch.full((2, 3, 2, 2), .25 + .05 * index),
                "matching_rgb": torch.full((2, 3, 4, 4), .5),
                "image_mask": torch.ones(2, 2, 2, dtype=torch.bool),
                "view_mask": torch.ones(2, dtype=torch.bool),
            },
            "targets": {
                "depth": torch.full((2, 2, 2), 4. + .1 * index),
                "depth_valid": torch.ones(2, 2, 2, dtype=torch.bool),
            },
        }


def collate(samples):
    return {
        "domain_index": samples[0]["domain_index"],
        **{group: {key: torch.stack([sample[group][key] for sample in samples])
                   for key in samples[0][group]}
           for group in ("inputs", "targets")},
    }


class ToyModel(nn.Module):
    architecture = "toy_v6"

    def __init__(self, config, initialize=True):
        super().__init__()
        self.decoder = nn.Linear(1, 1)
        self.dropout = nn.Dropout(.25)

    def set_stage(self, number):
        self.active_stage = number

    def forward(self, inputs, domain, reference=0):
        assert "targets" not in inputs and "depth" not in inputs
        feature = self.dropout(inputs["rgb"][:, reference].mean((1, 2, 3))[:, None])
        depth = self.decoder(feature).exp()[:, :, None, None].expand(-1, 1, 2, 2)
        return {"depth": depth, "valid": torch.ones_like(depth, dtype=torch.bool),
                "unit_scale": domain["depth_scale"]}


def toy_loss(prediction, inputs, targets, domain, settings, correspondence_weight=0.):
    estimate = prediction["depth"] * prediction["unit_scale"]
    return {"total": (estimate - targets["depth"][:, :1]).square().mean()}


@pytest.fixture
def cpu_runtime(monkeypatch):
    runtime = SimpleNamespace(device=torch.device("cpu"), distributed=False,
                              rank=0, local_rank=0, world_size=1, is_main=True)
    monkeypatch.setattr(train, "setup_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(train, "cleanup_runtime", lambda runtime: None)
    monkeypatch.setattr(train, "offline_only", lambda: None)
    monkeypatch.setattr(train, "build_model", lambda config, initialize=True: ToyModel(config, initialize))
    monkeypatch.setattr(train, "objective", toy_loss)
    monkeypatch.setattr(train, "collate_mixed", collate)
    monkeypatch.setattr(common_data, "SharedDataset", ToyDataset)
    # RestartHooks 从其所属模块读取 collate；共同VAL统计仍真实执行。
    monkeypatch.setattr("research.v2_restart_20261029.training.collate_mixed", collate)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args, **kwargs: 0)
    return runtime


def configuration(tmp_path):
    manifest = tmp_path / "toy.jsonl"
    manifest.write_text('{"synthetic_state_machine_test": true}\n')
    return {
        "method": "v6", "seed": 1234, "output": str(tmp_path / "training"),
        "initialization": {},
        "datasets": [dict(name="toy", geometry_units="meters", depth_scale=2., near=.1,
                          far=10., train_multiplier=1.,
                          manifests={"train": str(manifest), "val": str(manifest)},
                          train_manifest_sha256=train.digest(manifest))],
        "train": {
            "threads": 1, "batch_size": 1, "accumulation": 2, "global_batch_size": 2,
            "workers": 0, "precision": "fp32", "minimum_free_gib": 0,
            "weight_decay": 1e-4, "log_every": 1, "checkpoint_every": 1,
            "full_validation_every": 1, "validation_samples_per_domain": 2,
            "warmup_updates": 1,
        },
        "stages": [{"number": 1, "epochs": 1, "decoder_lr": .01},
                   {"number": 2, "epochs": 1, "decoder_lr": .005}],
        "range_augmentation": {"enabled": True, "nominal_probability": .5,
                               "unit_jitter_probability": 1.},
        "evaluation": {"precision": "fp32", "selection_region_weights": {"full": 1.}},
    }


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_stage_boundaries_are_half_open():
    config = {"stages": [{"number": 1, "epochs": 1}, {"number": 2, "epochs": 2},
                         {"number": 3, "epochs": 3}]}
    assert train.stage_for(config, 0)[1:] == (0, 1)
    assert train.stage_for(config, 1)[0]["number"] == 2
    assert train.stage_for(config, 2)[0]["number"] == 2
    assert train.stage_for(config, 3)[1:] == (3, 6)
    assert train.stage_for(config, 5)[0]["number"] == 3
    with pytest.raises(ValueError, match="epoch"):
        train.stage_for(config, 6)


@pytest.mark.parametrize("method,unit_scale", [("v6", 2.), ("v7", 2.), ("mvsanywhere", 1.)])
def test_evaluation_uses_no_targets_and_preserves_physical_depth(method, unit_scale):
    class EchoDepth(nn.Module):
        def forward(self, inputs, domain, reference=0):
            assert "targets" not in inputs
            assert reference == 1
            if method != "mvsanywhere":
                assert torch.all(inputs["matching_rgb"] == .25)
            depth = torch.full((1, 1, 2, 2), 8. / unit_scale)
            return {"depth": depth, "valid": torch.ones_like(depth, dtype=torch.bool),
                    "unit_scale": unit_scale}

    inputs = {"rgb": torch.full((1, 2, 3, 2, 2), .25),
              "matching_rgb": torch.full((1, 2, 3, 4, 4), .8)}
    domain = {"depth_scale": 2.}
    result = train.EvaluationModel(EchoDepth().eval(), method)(inputs, domain, reference=1)
    assert torch.all(result["depth"] * domain["depth_scale"] == 8.)
    assert torch.all(inputs["matching_rgb"] == .8)


@pytest.mark.parametrize("stop", [1, 2])
def test_training_resume_matches_uninterrupted_across_stage_boundary(tmp_path, cpu_runtime, stop):
    config = configuration(tmp_path)
    config["output"] = str(tmp_path / "uninterrupted")
    train.run(config)
    expected = load(tmp_path / "uninterrupted/final.pt")

    interrupted = copy.deepcopy(config)
    interrupted["output"] = str(tmp_path / "resumed")
    interrupted["train"]["stop_after_updates"] = stop
    train.run(interrupted)
    saved = load(tmp_path / "resumed/last.pt")
    assert saved["state"]["global_step"] == stop
    assert saved["state"]["exposures"] == stop * 2
    interrupted["train"].pop("stop_after_updates")
    train.run(interrupted, str(tmp_path / "resumed/last.pt"))
    actual = load(tmp_path / "resumed/final.pt")
    assert actual["state"] == expected["state"]
    for name, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][name], value, rtol=0., atol=0.)

    # 最后epoch的last可再次恢复并完成导出，不需要额外更新。
    train.run(interrupted, str(tmp_path / "resumed/last.pt"))
    assert load(tmp_path / "resumed/final.pt")["state"] == actual["state"]


def test_changed_training_config_and_manifest_are_rejected(tmp_path, cpu_runtime):
    config = configuration(tmp_path)
    config["train"]["stop_after_updates"] = 1
    train.run(config)
    resume = str(tmp_path / "training/last.pt")
    changed = copy.deepcopy(config)
    changed["seed"] += 1
    with pytest.raises(ValueError, match="严格恢复"):
        train.run(changed, resume)
    manifest = tmp_path / "toy.jsonl"
    manifest.write_text('{"modified": true}\n')
    with pytest.raises(RuntimeError, match="TRAIN清单已改变"):
        train.run(config, resume)
