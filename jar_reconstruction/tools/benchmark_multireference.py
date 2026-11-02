"""无标签DEV速度比较：单上参考，以及四参考完整网络输出。"""

import argparse
import json
from pathlib import Path
import shlex
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.carla_comparison import read_json
from jar.models.ruler_direct_model import RulerDirectModel
from jar.models.shared_initializer import SharedPretrainedInitializer
from jar.runtime import move
from tools.benchmark_raw_feature_reader import load_frames
from tools.benchmark_ruler_models import position_caches, summarize
from tools.carla_compare import file_sha256
from tools.verify_position_cache import measure_alternating


def direct_forward(model, references, upper):
    def invoke(inputs):
        # 同一实例交替测单参考/多参考，每次显式设置，防止状态串扰。
        model.reference_index = upper
        model.settings["reference_indices"] = list(references)
        output = model(inputs)["dense_depth"]
        return output[:, list(references)]

    return invoke


def baseline_forward(model, references):
    def invoke(inputs):
        depths = []
        for reference in references:
            model.reference_index = reference
            depths.append(model(inputs)["depth"])
        return torch.cat(depths, dim=1)

    return invoke


@torch.inference_mode()
def verify_calls(variants, direct, baseline, inputs, upper):
    """确认输出参考槽与实际执行次数；不读取任何GT。"""
    counts = {}
    handles = []
    watched = {
        "direct_matching": direct.matching,
        "direct_foundation": direct.foundation,
        "baseline_matching": baseline.native_model.matching_model,
        "baseline_foundation": baseline.native_model.encoder,
    }
    for name, module in watched.items():
        def count_call(module, arguments, name=name):
            counts[name] += 1
        handles.append(module.register_forward_pre_hook(count_call))
    outputs, records = {}, {}
    try:
        for name, invoke in variants.items():
            counts = dict.fromkeys(watched, 0)
            result = invoke(inputs)
            expected_views = 4 if name.endswith("four") else 1
            if result.shape != (1, expected_views, *inputs["rgb"].shape[-2:]):
                raise AssertionError("输出相机数或图像尺寸错误: " + name)
            if not bool((torch.isfinite(result) & (result > 0)).all()):
                raise AssertionError("无效深度: " + name)
            outputs[name] = result.clone()
            records[name] = {"shape": list(result.shape), "forward_calls": dict(counts)}
    finally:
        for handle in handles:
            handle.remove()
    if records["direct_four"]["forward_calls"]["direct_matching"] != 1:
        raise AssertionError("自研四参考须共享一次匹配编码")
    for name in ("direct", "baseline"):
        difference = (outputs[name + "_single"][:, 0] - outputs[name + "_four"][:, upper]).abs()
        records[name + "_single_four_parity"] = {
            "maximum_depth_difference_m": float(difference.max()),
            "mean_depth_difference_m": float(difference.mean()),
        }
        # 验证再次切回单参考，未沿用最后一个参考或四参考设置。
        repeated = variants[name + "_single"](inputs)
        torch.testing.assert_close(repeated, outputs[name + "_single"], atol=0, rtol=0)
    return records


def run(plan):
    if plan["split"] != "dev_complete":
        raise ValueError("本工具只测DEV速度，不读取TEST")
    output = Path(plan["output"])
    if (output / "summary.json").exists():
        raise FileExistsError("保留已有结果，请指定新的输出目录")
    device = torch.device(plan["device"])
    if device.type != "cuda":
        raise ValueError("网络计时要求CUDA")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = read_json(plan["protocol"])
    upper = int(protocol["evaluation"]["reference_index"])
    if len(protocol["data"]["camera_ids"]) != 4 or protocol["data"]["camera_ids"][upper] != "stereo_up":
        raise ValueError("固定比较需要四目与上相机单参考")
    frames = load_frames(protocol, plan)
    direct = RulerDirectModel.from_checkpoint(plan["direct_checkpoint"]).to(device).eval()
    baseline = SharedPretrainedInitializer(plan["initializer"]).to(device).eval()
    position_caches(direct)
    position_caches(baseline)
    variants = {
        "direct_single": direct_forward(direct, [upper], upper),
        "baseline_single": baseline_forward(baseline, [upper]),
        "direct_four": direct_forward(direct, [0, 1, 2, 3], upper),
        "baseline_four": baseline_forward(baseline, [0, 1, 2, 3]),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    all_times = {name: [] for name in variants}
    rows = []
    for frame in frames:
        inputs = move(frame["inputs"], device)
        checks = verify_calls(variants, direct, baseline, inputs, upper)
        timing = measure_alternating(variants, inputs, device, plan["warmup"], plan["repeats"])
        for name, record in timing.items():
            all_times[name].extend(record["samples_ms"])
        rows.append({"sample_id": frame["sample_id"], "checks": checks, "timing": timing})
        print(json.dumps({"sample_id": frame["sample_id"], "median_ms": {
            name: record["median_ms"] for name, record in timing.items()
        }}, ensure_ascii=False), flush=True)
    source_files = ("tools/benchmark_multireference.py", "jar/models/ruler_direct_model.py",
                    "jar/models/ruler_direct.py", "jar/models/shared_initializer.py")
    for relative in source_files:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / relative).read_text(), encoding="utf-8")
    report = {
        "plan": plan,
        "command": shlex.join([sys.executable, *sys.argv]),
        "gpu": torch.cuda.get_device_name(device),
        "precision": "FP32计算，TF32关闭；双方均启用冻结位置编码缓存",
        "no_targets_loaded": True, "no_test_read": True, "no_training": True,
        "checkpoint_sha256": {"direct": file_sha256(plan["direct_checkpoint"]),
                              "baseline": file_sha256(plan["initializer"]["weights"])},
        "source_sha256": {name: file_sha256(ROOT / name) for name in source_files},
        "protocol_sha256": file_sha256(plan["protocol"]),
        "manifest_sha256": file_sha256(protocol["data"]["manifests"][plan["split"]]),
        "timing": {name: summarize(values) for name, values in all_times.items()},
        "frames": rows,
        "notes": [
            "四参考时间包含四张深度的全部网络计算；单参考只输出上相机，二者不混比。",
            "自研四参考：匹配编码一次，ViT批处理四参考，四次共享参数解码。",
            "baseline四参考：同实例依次完成四次原生首遍前向，未缓存匹配特征；不代表其最优多参考实现。",
            "输入同四目256×448，自研内部匹配图512×896，baseline为256×448；沿用双方既有协议。",
            "自研描述子执行既有FP16往返量化，计算为FP32；baseline使用原生FP32描述子。",
            "时间含RGB编码及输出封装，不含读盘、H2D和外部后处理；不代表车端实测。",
            "旧权重仅在上相机深度上训练；此次其他参考仅验证执行，不评价或宣称其精度。",
        ],
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"complete": str(output), "median_ms": {
        name: row["median_ms"] for name, row in report["timing"].items()
    }}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    run(read_json(arguments.config))
