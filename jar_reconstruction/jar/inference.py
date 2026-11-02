"""MF-ET-JAR 的 GPU/CPU 推理；多进程按完整帧分片，评估不重复补样。"""

import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from . import build_model
from .config import load_config
from .data import build_dataset, collate_frames
from .metrics import SURFACE_METRIC_VERSION, evaluate_batch
from .space import evaluate_space
from .runtime import setup_runtime, cleanup_runtime, precision_context, move, seed_all


def load_inference_model(cfg, checkpoint, device):
    from .multifocal.checkpoint import checkpoint_model_state

    # 先在CPU读取一次，让完整checkpoint恢复其冻结分支而不依赖旧缓存路径。
    record = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint else None
    model = build_model(cfg, checkpoint_state=checkpoint_model_state(record) if record else None)
    if record is not None:
        if "camera" in cfg or cfg["model"]["name"] in ("mf_et_jar", "ruler_direct"):
            from .multifocal.checkpoint import load_model_weights

            model._checkpoint_load_report = load_model_weights(
                model, record, allow_base_transfer=True
            )
        else:
            if record["config"]["model"] != cfg["model"]:
                raise ValueError("推理模型配置必须与checkpoint一致；设备和精度可以改变")
            model.load_state_dict(record["model"], strict=True)
    return model.to(device).eval()


class Predictor:
    """常驻模型推理API。传入inputs即可；不接收targets或训练teacher。"""

    def __init__(self, cfg, checkpoint="", device="cpu", precision="fp32"):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前CUDA不可用；请先运行 tools/check_gpu.py")
        self.precision = precision
        # 精度预检，禁止不支持的设备/精度静默退回。
        with precision_context(self.device, self.precision):
            pass
        self.model = load_inference_model(cfg, checkpoint, self.device)

    @torch.inference_mode()
    def __call__(self, inputs):
        prepared = move(inputs, self.device, non_blocking=self.device.type == "cuda")
        with precision_context(self.device, self.precision):
            return self.model(prepared)


def frame_indices(length, rank=0, world_size=1, limit=None):
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("非法分布式分片参数")
    if limit is not None and limit < 0:
        raise ValueError("limit 不能为负数")
    total = length if limit is None else min(length, limit)
    return list(range(rank, total, world_size))


def export_prediction(path, scene, inputs, metadata, effective_config=None, outputs=None):
    """只写数值数组，可用np.load(...,allow_pickle=False)读取；保留unknown。"""
    keys = (
        "points",
        "normals",
        "scales",
        "valid",
        "support",
        "confidence",
        "state",
        "state_logits",
        "centers",
        "sizes",
        "field",
        "levels",
        "semantic_logits",
        "in_roi",
    )

    def array(t):
        t = t.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()

    scene = scene or {}
    arrays = {k: array(scene[k]) for k in keys if k in scene}
    dense = (outputs or {}).get("dense")
    if dense is not None:
        arrays.update(
            {
                "dense_" + key: array(value[0])
                for key, value in dense.items()
                if torch.is_tensor(value)
            }
        )
        arrays.update(
            {
                key: array(value[0])
                for key, value in outputs.items()
                if key.startswith("dense_") and torch.is_tensor(value)
            }
        )
    aux = scene.get("aux", {})
    has_mf_aux = aux.get("architecture", "").startswith("mf_") or any(
        k.startswith("mf_") for k in aux
    )
    if has_mf_aux:
        for key in (
            "ref_index",
            "ref_uv",
            "camera_role",
            "source_kind",
            "depth_sigma_m",
            "coverage_radius_m",
            "mf_state_per_view",
            "mf_reason",
        ):
            if torch.is_tensor(aux.get(key)):
                arrays["aux_" + key] = array(aux[key])
        if torch.is_tensor(aux.get("fusion_representative_indices")):
            arrays["fusion_representative_indices"] = array(aux["fusion_representative_indices"])
    for key in (
        "K",
        "T_cam_from_rig",
        "view_mask",
        "image_mask",
        "image_sizes",
        "camera_roles",
        "camera_valid",
    ):
        if key in inputs:
            arrays[key] = array(inputs[key][0])
    evidence = scene.get("ray_evidence")
    if evidence is not None:
        # Preserve the sparse observed-ray evidence; point states alone cannot
        # reproduce a free-space query after exporting the prediction.
        arrays.update(
            {
                "ray_evidence_" + key: array(value)
                for key, value in evidence.items()
                if torch.is_tensor(value)
            }
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    units = (effective_config or {}).get("data", {}).get("geometry_units", "meters")
    unit_label = "米" if units == "meters" else "原场景单位（未经米制认证）"
    info = {
        "metadata": metadata,
        "units": unit_label,
        "geometry_units": units,
        "rig_axes": "右手系：x前/y左/z上",
        "state_codes": {"free": 0, "surface": 1, "unknown": 2},
        "contains_ground_truth": False,
        "note": "只有valid=true的点是确认表面；unknown及无效叶未被删除",
    }
    if dense is not None:
        info["dense_output"] = {
            "depth_definition": f"光轴Z-depth，单位为{unit_label}；stride=1，整数像素中心",
            "valid_definition": "参考图输入有效域，不代表多视图已观测表面",
            "confidence_definition": "预测深度误差的置信度，不作为空间可见性证据",
            "observed_definition": "若包含dense_observed，它独立表示几何验收结果",
            "point_cloud_created": False,
        }
        if not scene:
            info["note"] = "本文件只有稠密深度估计及相机信息，没有将它伪装成已验证表面点云"
    if "camera_ids" in inputs:
        info["camera_ids"] = list(inputs["camera_ids"][0])
        info["camera_roles"] = [
            "wide" if int(role) == 0 else "tele" for role in inputs["camera_roles"][0].tolist()
        ]
        info["image_sizes"] = inputs["image_sizes"][0].tolist()
        info["camera_valid"] = inputs["camera_valid"][0].tolist()
    if effective_config is not None and "camera" in effective_config:
        from .multifocal.options import effective_mode_metadata

        info["model_mode"] = effective_mode_metadata(effective_config)
        info["effective_config"] = effective_config
    if has_mf_aux:
        from .multifocal.evidence import STATE_NAMES, REASON_NAMES

        info["multifocal_aux"] = {
            "state_codes": dict(enumerate(STATE_NAMES)),
            "reason_codes": dict(enumerate(REASON_NAMES)),
            "depth_sigma_note": f"单位为{unit_label}的局部线性噪声传播估计，未做真实数据概率校准",
            "coverage_radius_note": f"单位为{unit_label}的表面采样足迹，与位置不确定性分开",
        }
    if evidence is not None:
        info["ray_evidence"] = {
            key: value for key, value in evidence.items() if not torch.is_tensor(value)
        }
        info["ray_evidence"]["array_prefix"] = "ray_evidence_"
    path.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")


def summarize_records(records):
    """按全部唯一帧汇总，不平均各rank均值；深度误差另按有效像素加权。"""
    ids = [r["sample"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("评估包含重复帧，不能汇总")
    excluded = {
        "sample",
        "rank",
        "latency_ms",
        "peak_cuda_memory_bytes",
        "peak_cuda_reserved_bytes",
        "metadata",
        "prediction_file",
    }
    keys = set().union(*(r.keys() for r in records)) if records else set()
    means = {}
    counts = {}
    for key in sorted(keys - excluded):
        values = [
            float(r[key])
            for r in records
            if r.get(key) is not None and isinstance(r.get(key), (int, float))
        ]
        if not values:
            means[key] = None
            counts[key] = 0
            continue
        if not all(math.isfinite(v) for v in values):
            raise FloatingPointError(f"指标 {key} 非有限")
        means[key] = sum(values) / len(values)
        counts[key] = len(values)
    n = sum(r.get("predicted_depth_pixels", 0) for r in records)
    gt = sum(r.get("labeled_depth_pixels", 0) for r in records)
    absolute = sum(
        r["depth_mae_m"] * r.get("predicted_depth_pixels", 0)
        for r in records
        if r.get("depth_mae_m") is not None
    )
    squared = sum(
        r["depth_rmse_m"] ** 2 * r.get("predicted_depth_pixels", 0)
        for r in records
        if r.get("depth_rmse_m") is not None
    )
    latency = [r["latency_ms"] for r in records]
    summary = {
        "frame_count": len(records),
        "metrics_frame_mean": means,
        "metric_valid_frames": counts,
        "depth_pixel_weighted": {
            "mae_m": absolute / n if n else None,
            "rmse_m": math.sqrt(squared / n) if n else None,
            "coverage": n / gt if gt else None,
            "predicted_pixels": n,
            "labeled_pixels": gt,
        },
        "depth_excluded_predictions": {
            "scope": "至少有一个有效深度标签的相机；无标签预测计数不等同于天空误检数",
            **{
                key: sum(r.get(key, 0) for r in records)
                for key in (
                    "predicted_pixels_in_depth_labeled_views",
                    "predicted_pixels_without_depth_label",
                    "predicted_pixels_outside_gt_roi",
                )
            },
        },
        "latency_ms_mean": float(np.mean(latency)) if latency else None,
        "latency_ms_median": float(np.median(latency)) if latency else None,
        "peak_cuda_memory_bytes": max((r["peak_cuda_memory_bytes"] for r in records), default=0),
        "note": "帧指标为所有唯一帧的宏平均；深度像素加权指标单列；延迟不是多卡总吞吐率",
    }
    if any("dense_labeled_depth_pixels" in record for record in records):
        dense_n = sum(record.get("dense_predicted_depth_pixels", 0) for record in records)
        dense_gt = sum(record.get("dense_labeled_depth_pixels", 0) for record in records)
        weighted = {}
        for name in ("mae_m", "rmse_m", "absrel", "delta1"):
            key = "dense_depth_" + name
            power = 2 if name == "rmse_m" else 1
            total = sum(
                record[key] ** power * record.get("dense_predicted_depth_pixels", 0)
                for record in records
                if record.get(key) is not None
            )
            weighted[name] = (total / dense_n) ** (1 / power) if dense_n else None
        weighted.update(
            coverage=dense_n / dense_gt if dense_gt else None,
            predicted_pixels=dense_n,
            labeled_pixels=dense_gt,
        )
        summary["dense_depth_pixel_weighted"] = weighted
    return summary


def run_prediction(args, evaluate=True):
    from .multifocal.options import effective_mode_metadata

    cfg = load_config(args.config)
    if evaluate and cfg.get("data", {}).get("geometry_units", "meters") != "meters":
        raise ValueError("当前评测入口输出米制指标；scene_unit数据不能混作米制评测")
    precision = cfg["eval"]["precision"]
    warmup = cfg["eval"]["warmup"]
    workers = cfg["eval"]["num_workers"]
    if not evaluate and not args.checkpoint and not args.synthetic:
        raise ValueError("真实数据推理需要 --checkpoint；随机权重仅允许显式合成测试")
    runtime = setup_runtime(args.device, cfg["runtime"]["dist_timeout"])
    try:
        torch.set_num_threads(cfg["runtime"]["threads"])
        seed_all(cfg["train"]["seed"])
        dataset = build_dataset(cfg, args.split, synthetic=args.synthetic, load_targets=evaluate)
        indices = frame_indices(len(dataset), runtime.rank, runtime.world_size, args.limit)
        loader_options = {
            "num_workers": workers,
            "pin_memory": bool(cfg["eval"]["pin_memory"] and runtime.device.type == "cuda"),
        }
        if workers:
            loader_options["multiprocessing_context"] = "spawn"
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=1,
            shuffle=False,
            collate_fn=collate_frames,
            **loader_options,
        )
        predictor = Predictor(cfg, args.checkpoint, runtime.device, precision)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        mode_info = effective_mode_metadata(cfg)
        if runtime.is_main:
            (output / "config.json").write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
            )
            if "camera" in cfg:
                print(
                    json.dumps(
                        {"event": "effective_config", "model_mode": mode_info, "config": cfg},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        records = []
        logpath = output / f"frames.rank{runtime.rank:04d}.jsonl"
        with logpath.open("w") as stream, torch.inference_mode():
            for local_index, batch in enumerate(loader):
                batch = move(batch, runtime.device, non_blocking=runtime.device.type == "cuda")
                if local_index == 0:
                    for _ in range(warmup):
                        predictor(batch["inputs"])
                if runtime.device.type == "cuda":
                    torch.cuda.synchronize(runtime.device)
                    torch.cuda.reset_peak_memory_stats(runtime.device)
                begin = time.perf_counter()
                prediction = predictor(batch["inputs"])
                if runtime.device.type == "cuda":
                    torch.cuda.synchronize(runtime.device)
                elapsed = time.perf_counter() - begin
                allocated = (
                    torch.cuda.max_memory_allocated(runtime.device)
                    if runtime.device.type == "cuda"
                    else 0
                )
                reserved = (
                    torch.cuda.max_memory_reserved(runtime.device)
                    if runtime.device.type == "cuda"
                    else 0
                )
                record = {
                    "sample": indices[local_index],
                    "rank": runtime.rank,
                    "metadata": batch["metadata"][0],
                    "latency_ms": elapsed * 1000,
                    "peak_cuda_memory_bytes": allocated,
                    "peak_cuda_reserved_bytes": reserved,
                }
                if evaluate:
                    metrics = evaluate_batch(
                        prediction,
                        batch["inputs"],
                        batch["targets"],
                        cfg["eval"]["fscore_threshold"],
                        cfg["eval"]["max_metric_points"],
                        cfg["model"]["roi"],
                        observed_only=cfg["eval"].get("observed_only", False),
                        dense_depth_range=(cfg["model"]["near"], cfg["model"]["far"]),
                    )
                    if (
                        prediction.get("scenes")
                        and "depth" in batch["targets"]
                        and bool(
                            batch["targets"].get("depth_valid", batch["targets"]["depth"] > 0).any()
                        )
                    ):
                        metrics.update(
                            evaluate_space(
                                prediction,
                                batch["inputs"],
                                batch["targets"],
                                cfg["model"]["roi"],
                                cfg["eval"]["spatial_probe_spacing"],
                                cfg["eval"]["surface_band"],
                            )
                        )
                    record.update({k: None if v is None else float(v) for k, v in metrics.items()})
                if not evaluate or getattr(args, "save_predictions", False):
                    relative = Path("predictions") / f"{indices[local_index]:09d}.npz"
                    export_prediction(
                        output / relative,
                        prediction["scenes"][0] if prediction.get("scenes") else None,
                        batch["inputs"],
                        batch["metadata"][0],
                        effective_config=cfg,
                        outputs=prediction,
                    )
                    record["prediction_file"] = str(relative)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                records.append(record)
                # 避免下一帧forward期间仍持有上一帧的GPU输出及标签引用。
                del prediction, batch
        if runtime.distributed:
            gathered = [None] * runtime.world_size
            dist.all_gather_object(gathered, records)
            records = [r for shard in gathered for r in shard]
        records.sort(key=lambda r: r["sample"])
        if runtime.is_main:
            expected = min(len(dataset), args.limit) if args.limit is not None else len(dataset)
            if [r["sample"] for r in records] != list(range(expected)):
                raise RuntimeError("评估分片出现遗漏或重复帧")
            report = {
                "checkpoint": args.checkpoint,
                "synthetic": args.synthetic,
                "untrained": not bool(args.checkpoint),
                "mode": "evaluation" if evaluate else "inference",
                "device": str(runtime.device),
                "precision": precision,
                "world_size": runtime.world_size,
                "warmup_per_rank": warmup,
                "latency_includes_cold_start": warmup == 0,
                "labels_loaded": evaluate,
                "samples": records,
                "summary": summarize_records(records),
                "model_mode": mode_info,
                "checkpoint_load": getattr(predictor.model, "_checkpoint_load_report", None),
            }
            if evaluate:
                report["metric_protocol"] = {
                    "surface_version": SURFACE_METRIC_VERSION,
                    "surface_query_cap": cfg["eval"]["max_metric_points"],
                    "surface_threshold_m": cfg["eval"]["fscore_threshold"],
                    "surface_reference": "完整点集；只抽样距离查询侧",
                    "depth_output": "跨相机有效点的稀疏单像素z-buffer渲染",
                    "undefined_without_gt": True,
                    "comparison_note": "此入口不等同于tools/carla_compare.py的固定参考2–80米协议。",
                }
                if any("dense_labeled_depth_pixels" in record for record in records):
                    report["metric_protocol"][
                        "dense_depth_output"
                    ] = "原生稠密光轴深度；dense_*独立报告，不通过稀疏点云渲染"
                    report["metric_protocol"]["dense_depth_range_m"] = [
                        cfg["model"]["near"],
                        cfg["model"]["far"],
                    ]
                    report["metric_protocol"][
                        "dense_mask"
                    ] = "GT/输入有效域；不按预测confidence或dense_observed删掉困难像素"
            for field in ("camera_completeness", "valid_camera_count"):
                groups = sorted(
                    {str(r["metadata"][field]) for r in records if field in r["metadata"]}
                )
                if groups:
                    report["by_" + field] = {
                        key: summarize_records(
                            [r for r in records if str(r["metadata"].get(field)) == key]
                        )
                        for key in groups
                    }
            filename = "evaluation.json" if evaluate else "inference.json"
            (output / filename).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "完成": "评估" if evaluate else "推理",
                        "结果": str(output / filename),
                        "summary": report["summary"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
        return records
    finally:
        cleanup_runtime(runtime)
