"""无标签 VAL 单帧推理诊断；临时插桩，不改模型、权重或候选算法。"""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.data import build_dataset, collate_frames
from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import dataset_config
from jar.pretraining.model import PretrainingModel
from jar.runtime import move, seed_all
from tools.smoke_outdoor_datasets import fingerprint, offline_only
import jar.models.ruler_proposals as proposals


def shapes(value, depth=0):
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if depth > 1:
        return None
    if isinstance(value, dict):
        return {str(key): shapes(item, depth + 1) for key, item in value.items()
                if torch.is_tensor(item) or isinstance(item, (list, tuple, dict))}
    if isinstance(value, (list, tuple)):
        return [shapes(item, depth + 1) for item in value[:6]]
    return value if isinstance(value, (int, float, str, bool)) else None


class StageRecorder:
    """CUDA 事件记录包含时间；只减直接子调用，得到该阶段未细分的时间。"""

    def __init__(self):
        self.pending = []
        self.stack = []
        self.restore = []
        self.shape_records = {}
        self.events_enabled = True

    @contextmanager
    def region(self, name, args):
        path = "/".join([item["name"] for item in self.stack] + [name])
        parent = self.stack[-1] if self.stack else None
        record = {"name": name, "path": path, "children": []}
        if self.events_enabled:
            record["start"] = torch.cuda.Event(enable_timing=True)
            record["end"] = torch.cuda.Event(enable_timing=True)
            record["start"].record()
            self.pending.append(record)
            if parent is not None:
                parent["children"].append(record)
        if path not in self.shape_records:
            self.shape_records[path] = {"inputs": shapes(args)}
        self.stack.append(record)
        try:
            with torch.profiler.record_function(path):
                yield record
        finally:
            self.stack.pop()
            if self.events_enabled:
                record["end"].record()

    def wrap(self, owner, attribute, label):
        original = getattr(owner, attribute)

        def wrapped(*args, **kwargs):
            name = label(args, kwargs) if callable(label) else label
            with self.region(name, args) as record:
                result = original(*args, **kwargs)
                if "outputs" not in self.shape_records[record["path"]]:
                    self.shape_records[record["path"]]["outputs"] = shapes(result)
                return result

        self.restore.append((owner, attribute, original))
        setattr(owner, attribute, wrapped)

    def uninstall(self):
        for owner, attribute, original in reversed(self.restore):
            setattr(owner, attribute, original)
        self.restore.clear()

    def collect(self):
        torch.cuda.synchronize()
        collected = defaultdict(lambda: {"inclusive_ms": 0.0, "exclusive_ms": 0.0, "calls": 0})
        for record in self.pending:
            elapsed = record["start"].elapsed_time(record["end"])
            children = sum(child["start"].elapsed_time(child["end"]) for child in record["children"])
            item = collected[record["path"]]
            item["inclusive_ms"] += elapsed
            item["exclusive_ms"] += elapsed - children
            item["calls"] += 1
        self.pending.clear()
        return dict(collected)


def install_stages(model, recorder):
    recorder.wrap(model, "forward", "network")
    recorder.wrap(model.foundation, "forward", "foundation")
    for index, block in enumerate(model.foundation.model.blocks):
        recorder.wrap(block, "forward", f"block_{index:02d}")
    recorder.wrap(model.matching, "forward", "matching")
    for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"):
        recorder.wrap(getattr(model.matching.backbone, name), "forward", name)
    for index, layer in enumerate(model.matching.lateral):
        recorder.wrap(layer, "forward", f"lateral_{4 * 2 ** index}")
    for index, layer in enumerate(model.matching.smooth):
        recorder.wrap(layer, "forward", f"smooth_{4 * 2 ** index}")
    decoder = model.decoder
    recorder.wrap(decoder, "forward", "decoder")
    recorder.wrap(decoder, "_prepare", "prepare")
    for method, name in (("_observe", "observe"), ("_tokens", "tokens")):
        stride_index = 2 if method == "_observe" else 3
        recorder.wrap(decoder, method, lambda args, kwargs, index=stride_index, name=name:
                      ("coarse_" if args[index] == 8 else "fine_") + name)
    recorder.wrap(decoder, "_prediction", lambda args, kwargs:
                  ("coarse_" if args[7]["stride"] == 8 else "fine_") + "prediction")
    recorder.wrap(decoder.local, "forward", "local_scale_space")
    recorder.wrap(proposals, "transport_proposals", "proposal_transport")
    recorder.wrap(decoder, "_upsample", "upsample")
    recorder.wrap(decoder.source, "forward", "view_attention")
    for name in ("query", "key", "pair", "reliability", "null"):
        recorder.wrap(getattr(decoder.source, name), "forward", name)
    recorder.wrap(decoder, "_tokens_from_source", "context_fusion")
    for name in ("evidence", "position", "raw_query", "raw_output", "entry_norm", "fine_inherit", "score", "offset"):
        recorder.wrap(getattr(decoder, name), "forward", name)


def benchmark(call, warmup, repetitions):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    gpu, wall = [], []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repetitions):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record()
        output = call()
        end.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - start) * 1000)
        gpu.append(begin.elapsed_time(end))
        del output
    return {"gpu_median_ms": statistics.median(gpu), "wall_median_ms": statistics.median(wall),
            "gpu_samples_ms": gpu, "wall_samples_ms": wall,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def operator_report(profiler, repetitions):
    def device_time(item, exclusive=False):
        field = "self_device_time_total" if exclusive else "device_time_total"
        return float(getattr(item, field, 0)) / repetitions / 1000

    rows = []
    for item in profiler.key_averages(group_by_input_shape=True):
        # PyTorch 还把 GPU annotation 作为独立事件输出；它与实际 kernel
        # 有重叠，不能计为 self kernel 或加入算子总和。
        if not item.key.startswith("aten::"):
            continue
        rows.append({"name": item.key, "input_shapes": item.input_shapes,
                     "calls_per_frame": item.count / repetitions,
                     "self_gpu_ms_per_frame": device_time(item, True),
                     "inclusive_gpu_ms_per_frame": device_time(item),
                     "self_cpu_ms_per_frame": item.self_cpu_time_total / repetitions / 1000})
    return sorted(rows, key=lambda row: row["self_gpu_ms_per_frame"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有诊断：{output}")
    output.mkdir(parents=True)
    offline_only()
    torch.set_num_threads(config["cpu_threads"])
    torch.cuda.set_device(config["gpu"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    seed_all(config["seed"])
    device = torch.device("cuda", config["gpu"])
    data_settings = json.loads(Path(config["data_config"]).read_text())
    saved = torch.load(config["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    model = PretrainingModel(saved["config"], initialize=False).to(device).eval().requires_grad_(False)
    model.load_state_dict(saved["model"], strict=True)
    install_position_cache(model.foundation.model)
    del saved
    write_json(output / "audit.json", {"config": config, "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(device), "cuda": torch.version.cuda,
               "source_sha256": {str(path): digest(path) for path in
                                 [Path(__file__), ROOT / "jar/pretraining/model.py", ROOT / "jar/pretraining/encoders.py",
                                  ROOT / "jar/models/ruler_direct.py", ROOT / "jar/models/ruler_proposals.py"]},
               "checkpoint_sha256": digest(config["checkpoint"]),
               "protocol": "VAL中间一帧，无标签加载；FP32/TF32关闭，位置缓存；无编译，无下载。",
               "timing_scope": "单参考网络本体，不含读盘/传输/后处理。CUDA子阶段含host排队空隙；exclusive仅减直接子阶段。"})
    report = {}
    with torch.inference_mode():
        for name in config["datasets"]:
            domain = next(item for item in data_settings["datasets"] if item["name"] == name)
            dataset = build_dataset(dataset_config(domain, data_settings), "val", load_targets=False)
            index = len(dataset) // 2
            cpu_inputs = collate_frames([dataset[index]])["inputs"]
            input_hash = fingerprint(cpu_inputs)
            inputs = move(cpu_inputs, device)
            reference = domain["reference_index"]
            call = lambda: model(inputs, domain, reference)
            native = benchmark(call, config["warmup"], config["repetitions"])
            native_depth = call()["depth"].clone()
            recorder = StageRecorder()
            install_stages(model, recorder)
            try:
                stage_runs = []
                for _ in range(config["stage_repetitions"]):
                    prediction = call()
                    stage_runs.append(recorder.collect())
                difference = (prediction["depth"] - native_depth).abs().max().item()
                stages = {}
                for path in stage_runs[0]:
                    stages[path] = {key: statistics.median(run[path][key] for run in stage_runs)
                                    for key in ("inclusive_ms", "exclusive_ms", "calls")}
                recorder.events_enabled = False
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=True, profile_memory=True) as profiler:
                    for _ in range(config["profiler_repetitions"]):
                        call()
                    torch.cuda.synchronize()
                folder = output / name
                folder.mkdir()
                profiler.export_chrome_trace(str(folder / "trace.json"))
                operators = operator_report(profiler, config["profiler_repetitions"])
                # GPU annotation 和真实 kernel 重叠，不混入算子耗时排序。
                (folder / "operators.txt").write_text("\n".join(
                    f'{item["self_gpu_ms_per_frame"]:.6f} ms  {item["calls_per_frame"]:g} calls  '
                    f'{item["name"]}  {item["input_shapes"]}' for item in operators[:80]) + "\n")
                write_json(folder / "operators.json", operators)
                write_json(folder / "shapes.json", recorder.shape_records)
                report[name] = {"index": index, "split": "val", "load_targets": False,
                                "input_sha256": input_hash, "input_shapes": shapes(cpu_inputs),
                                "num_views": domain["num_views"], "reference_index": reference,
                                "native": native, "stages": stages,
                                "instrumentation_max_depth_difference": difference}
                write_json(output / "profile.json", report)
                print(json.dumps({"dataset": name, "native": native["gpu_median_ms"],
                                  "max_depth_difference": difference, "stages": stages}, ensure_ascii=False), flush=True)
            finally:
                recorder.uninstall()
            del inputs, native_depth, prediction, profiler, dataset, cpu_inputs
            torch.cuda.empty_cache()
    print(f"完成：{output}", flush=True)


if __name__ == "__main__":
    main()
