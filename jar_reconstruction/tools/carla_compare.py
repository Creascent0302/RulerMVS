"""按固定合同评价 CARLA；参数只读 JSON，网络推理不接收 targets。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.carla_comparison import (
    aggregate_depth,
    comparison_dataset,
    evaluate_depth,
    gt_reference,
    prediction_streams,
    raw_depth_diagnostic,
    read_json,
    surface_diagnostics,
)
from jar.data import collate_frames
from jar.raw_output_audit import aggregate_raw_audits, raw_output_audit
from jar.runtime import move, seed_all


def make_model(experiment, weights, device):
    name = experiment["method"]
    config_path = experiment["model_config"]
    if name in ("mf_et_jar", "census", "zncc", "cas_mvsnet"):
        from jar.config import load_config

        cfg = load_config(config_path)
        if name == "mf_et_jar":
            from jar.inference import load_inference_model

            model = load_inference_model(cfg, weights, device)
            # 推理消融先严格载入原架构的全部权重，再关闭一个无新参数的现有分支。
            # 不改checkpoint配置、不把不兼容权重静默部分载入。
            ablation = experiment.get("decoder_ablation")
            if ablation is not None:
                switches = {
                    "oriented_polish_off": ("oriented_stages", (), "et_oriented_polish_stages", []),
                    "parent_births_off": ("parent_children", False, "et_parent_children", False),
                }
                if ablation not in switches:
                    raise ValueError(
                        "decoder_ablation只允许单项oriented_polish_off或parent_births_off"
                    )
                attribute, value, field, configured = switches[ablation]
                setattr(model.decoder, attribute, value)
                cfg["model"][field] = configured
        elif name == "cas_mvsnet":
            from jar.official_casmvs import OfficialCasMVSNet

            model = OfficialCasMVSNet(cfg)
        else:
            from jar.baselines import ClassicalMVS

            model = ClassicalMVS(cfg, name)
        if weights and name != "mf_et_jar":
            checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    else:
        from official_methods.cli import load_config
        from official_methods.registry import build_adapter

        cfg = load_config(config_path)
        cfg["model"]["reference_indices"] = cfg["evaluation"]["reference_indices"]
        construction = copy.deepcopy(cfg)
        if weights:
            # 完整项目checkpoint会严格覆盖全部层，不重复下载初始化权重。
            construction["model"]["pretrained"] = False
            construction["model"]["weights"] = ""
            construction["_loading_shared_checkpoint"] = True
        model = build_adapter(construction)
        if weights:
            checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
            if checkpoint.get("method") != name:
                raise ValueError(
                    "共同checkpoint方法不一致；原生权重请写入model_config的model.weights"
                )
            model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), cfg


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_device(value):
    return torch.device("cuda:0" if value == "cuda" else value)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NativeTimer:
    """原生网络hook计时；四次参考调用相加，适配器开销留在forward字段。"""

    def __init__(self, model, device):
        self.device = device
        self.records = []
        self.starts = []
        native = next(
            (
                getattr(model, name)
                for name in ("native", "native_model", "net")
                if isinstance(getattr(model, name, None), torch.nn.Module)
            ),
            model,
        )
        self.definition = "原生模块每次调用求和；没有独立原生子模块时为完整方法forward。"
        self.handles = [
            native.register_forward_pre_hook(self.begin),
            native.register_forward_hook(self.end),
        ]

    def begin(self, module, inputs):
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = time.perf_counter()
        self.starts.append(start)

    def end(self, module, inputs, outputs):
        if self.device.type == "cuda":
            end = torch.cuda.Event(enable_timing=True)
            end.record()
        else:
            end = time.perf_counter()
        self.records.append((self.starts.pop(), end))

    def result(self):
        duration = sum(
            start.elapsed_time(end) if self.device.type == "cuda" else (end - start) * 1000
            for start, end in self.records
        )
        self.records.clear()
        return duration

    def close(self):
        for handle in self.handles:
            handle.remove()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="包含protocol、method、model_config、split的实验JSON"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--weights", default=None, help="项目checkpoint路径；省略时读取实验JSON的weights"
    )
    args = parser.parse_args(argv)
    experiment = read_json(args.config)
    protocol = read_json(experiment["protocol"])
    source_hashes = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in (ROOT / "tools/carla_compare.py", ROOT / "jar/carla_comparison.py")
    }
    settings = protocol["evaluation"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "comparison.json").exists() or (output / "frames.jsonl").exists():
        raise FileExistsError("输出目录已含评价记录，请使用新目录保留历史结果。")
    torch.set_num_threads(protocol["runtime"].get("threads", 2))
    seed_all(experiment.get("seed", 17))
    device = resolve_device(args.device)
    weights = experiment.get("weights", "") if args.weights is None else args.weights
    model, model_cfg = make_model(experiment, weights, device)
    timer = NativeTimer(model, device)
    dataset = comparison_dataset(protocol, experiment["split"])
    indices = experiment.get("frame_indices", list(range(len(dataset))))
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(dataset) for i in indices):
        raise ValueError("frame_indices必须是不重复的有效索引；少量子集不称完整评价。")
    records, definitions = [], {}
    warmup = settings.get("warmup", 2)
    ref = settings["reference_index"]
    official = experiment["method"] not in ("mf_et_jar", "census", "zncc", "cas_mvsnet")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    with torch.inference_mode(), (output / "frames.jsonl").open("w") as log:
        for index in indices:
            batch = move(collate_frames([dataset[index]]), device)
            inputs, targets = batch["inputs"], batch["targets"]
            if official and not inputs["view_mask"].all():
                raise ValueError("此官方四目协议不支持缺相机，不能复制视角或静默补黑图。")
            if not records:
                for _ in range(warmup):
                    model(inputs)
                synchronize(device)
                timer.result()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            start = time.perf_counter()
            prediction = model(inputs)
            synchronize(device)
            forward_ms = (time.perf_counter() - start) * 1000
            native_ms = timer.result()
            streams = prediction_streams(prediction, inputs, experiment["method"], model_cfg, ref)
            synchronize(device)
            pipeline_ms = (time.perf_counter() - start) * 1000
            peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            # 后处理只生成新张量，prediction仍是未清理的原始网络输出。
            # 此审计放在时间及显存快照之后，不改变既有计时、指标或GT分母。
            raw_audit = raw_output_audit(prediction)
            target, valid = gt_reference(inputs, targets, settings)
            source = dataset.frames[index]
            row = {
                "sample_id": f"{source['sequence']}/{source['frame_id']}",
                "dataset_index": index,
                "valid_camera_count": int(inputs["view_mask"].sum()),
                "complete": bool(inputs["view_mask"].all()),
                "forward_ms": forward_ms,
                "native_network_ms": native_ms,
                "pipeline_ms": pipeline_ms,
                "peak_cuda_bytes": peak_memory,
                "raw_output_audit": raw_audit,
                "streams": {},
            }
            for name, stream in streams.items():
                definitions[name] = stream["definition"]
                row["streams"][name] = evaluate_depth(
                    stream["depth"], stream["valid"], target, valid, settings
                )
            if settings.get("surface_metrics", False) and prediction.get("scenes"):
                scene = prediction["scenes"][0]
                points = scene["points"][scene["valid"] & (scene["state"] == 1)]
                row["native_points_surface_diagnostic"] = surface_diagnostics(
                    points, inputs, target, valid, settings
                )
            if settings.get("raw_depth_diagnostics", False):
                row["raw_label_boundary_inventory"] = raw_depth_diagnostic(
                    source,
                    protocol["data"]["root"],
                    protocol["data"]["camera_ids"][ref],
                    protocol["data"]["image_size"],
                )
            records.append(row)
            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            print(
                f"{experiment['method']} {len(records)}/{len(indices)} {row['sample_id']} forward={forward_ms:.1f} ms",
                flush=True,
            )
    timer.close()
    complete = [row for row in records if row["complete"]]
    partial = [row for row in records if not row["complete"]]
    report = {
        "experiment": experiment,
        "device": str(device),
        "command": [
            sys.executable,
            str(Path(__file__).resolve()),
            *(sys.argv[1:] if argv is None else argv),
        ],
        "protocol": protocol,
        "model_config": model_cfg,
        "checkpoint": weights,
        "checkpoint_sha256": file_sha256(weights) if weights else None,
        "source_sha256": source_hashes,
        "manifest_sha256": file_sha256(protocol["data"]["manifests"][experiment["split"]]),
        "pretrained_model_weights": model_cfg.get("model", {}).get("weights", ""),
        "full_manifest": len(indices) == len(dataset),
        "frame_count": len(records),
        "raw_output_audit": aggregate_raw_audits([r["raw_output_audit"] for r in records]),
        "stream_definitions": definitions,
        "complete": {"frames": len(complete), "metrics": aggregate_depth(complete, settings)},
        "partial": {"frames": len(partial), "metrics": aggregate_depth(partial, settings)},
        "by_camera_count": {
            str(n): aggregate_depth([r for r in records if r["valid_camera_count"] == n], settings)
            for n in sorted({r["valid_camera_count"] for r in records})
        },
        "timing": {
            "native_network_ms_mean": (
                sum(r["native_network_ms"] for r in records) / len(records) if records else None
            ),
            "native_definition": timer.definition,
            "forward_ms_mean": (
                sum(r["forward_ms"] for r in records) / len(records) if records else None
            ),
            "pipeline_ms_mean": (
                sum(r["pipeline_ms"] for r in records) / len(records) if records else None
            ),
            "note": "FP32单帧；forward含适配器内部四参考轮换/必要处理，pipeline再含共同深度转换；不含读盘、GT评价和原始标签库存，不能冒称纯原生网络时间。",
        },
        "limitations": [
            "完整与缺失相机分表；depth error只看有效预测交集，同时固定GT分母报告coverage和success。",
            "自研final_point_splat是最终稀疏输出，coarse_depth_diagnostic只是中间深度，不能互相替代。",
            "同序列临时开发集不代表独立新地图；所有已见319帧的旧CARLA权重不能宣称未见开发验证。",
            "真实数据没有小障碍物实例标签；深度边界分组不等于小障碍物检测召回。",
        ],
    }
    (output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(output / "comparison.json")


if __name__ == "__main__":
    main()
