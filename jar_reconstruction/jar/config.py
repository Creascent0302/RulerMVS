"""Strict JSON experiment configs, default real-data paths intentionally empty."""

import copy
import json
from pathlib import Path

DEFAULT = {
    "model": {
        "name": "mf_et_jar",
        "feature_dim": 64,
        "hidden_dim": 128,
        "backbone_width": 24,
        "near": 2.0,
        "far": 80.0,
        "depth_bins": 48,
        "correlation_groups": 8,
        "sweep_chunk": 8,
        "max_queries": 8192,
        "query_explore": 0.1,
        "refinement_min_step": 0.01,
        "refinement_max_step": 4.0,
        "fusion_cell_size": 0.05,
        "texture_threshold": 0.002,
        "evidence_threshold": 0.15,
        "min_support": 2,
        "roi": [[0.0, 64.0], [-24.0, 24.0], [-2.0, 6.0]],
        "semantic_classes": 0,
        "validate_inputs": True,
    },
    "data": {"root": "", "train_manifest": "", "val_manifest": "", "test_manifest": ""},
    "synthetic": {"length": 8, "height": 64, "width": 96, "num_cameras": 4, "seed": 7},
    "train": {
        "seed": 17,
        "batch_size": 1,
        "num_workers": 0,
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "max_steps": 1,
        "grad_clip": 1.0,
        "precision": "fp32",
        "accumulation_steps": 1,
        "pin_memory": True,
        "checkpoint_every": 100,
        "log_every": 1,
    },
    "eval": {
        "fscore_threshold": 0.2,
        "max_metric_points": 4096,
        "spatial_probe_spacing": 2.0,
        "surface_band": 0.1,
        "precision": "fp32",
        "warmup": 3,
        "num_workers": 0,
        "pin_memory": True,
    },
    "runtime": {"threads": 2, "dist_timeout": 300},
    "loss": {},
}


def merge(base, updates):
    """补齐文件中省略的默认字段；不处理命令行覆盖或配置继承。"""
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path=None):
    cfg = copy.deepcopy(DEFAULT)
    if path:
        cfg = merge(cfg, json.loads(Path(path).read_text()))
    if cfg["model"]["feature_dim"] < 4 or cfg["model"]["hidden_dim"] < 4:
        raise ValueError("feature dimensions must be >=4")
    m = cfg["model"]
    if m["name"] not in ("mf_et_jar", "ruler_direct"):
        raise ValueError("unknown model.name")
    if m["name"] == "ruler_direct" and not isinstance(cfg.get("direct"), dict):
        raise ValueError("ruler_direct模型必须在direct中明确指定解码架构")
    if m["max_queries"] < 1 or m["sweep_chunk"] < 1 or m["correlation_groups"] < 1:
        raise ValueError("query, sweep and group counts must be positive")
    if not 0 <= m["query_explore"] <= 1:
        raise ValueError("query_explore must be in [0,1]")
    if m["min_support"] < 2:
        raise ValueError("metric multi-view reconstruction requires min_support >=2")
    if not 0 <= m["evidence_threshold"] <= 1:
        raise ValueError("evidence_threshold must be in [0,1]")
    if min(cfg["synthetic"][k] for k in ["length", "height", "width", "num_cameras"]) < 1:
        raise ValueError("synthetic dimensions/counts must be positive")
    if (
        cfg["train"]["max_steps"] < 1
        or cfg["train"]["batch_size"] < 1
        or cfg["train"]["checkpoint_every"] < 1
        or cfg["train"]["log_every"] < 1
    ):
        raise ValueError("training counts must be positive")
    if cfg["train"]["accumulation_steps"] < 1:
        raise ValueError("accumulation_steps 必须是正整数")
    for section in ("train", "eval"):
        if cfg[section]["precision"] not in ("fp32", "bf16", "fp16"):
            raise ValueError("precision 仅支持 fp32/bf16/fp16")
        if cfg[section]["num_workers"] < 0:
            raise ValueError("num_workers 不能为负数")
    if cfg["eval"]["warmup"] < 0:
        raise ValueError("warmup 不能为负数")
    if min(cfg["runtime"]["threads"], cfg["runtime"]["dist_timeout"]) < 1:
        raise ValueError("runtime.threads 和 runtime.dist_timeout 必须为正整数")
    if "amp" in cfg["train"]:
        raise ValueError("请删除 train.amp，仅用 train.precision 设置 fp32/bf16/fp16")
    from .multifocal.options import configure

    return configure(cfg)
