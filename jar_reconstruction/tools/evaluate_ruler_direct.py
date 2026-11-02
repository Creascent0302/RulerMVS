"""直接标尺模型的真实 RGB 验证与同卡轮转计时；只读取 DEV。"""

import argparse
import copy
import json
from pathlib import Path
import shlex
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.carla_comparison import aggregate_depth, comparison_dataset, evaluate_depth, gt_reference, read_json
from jar.data import collate_frames
from jar.models.ruler_direct_model import RulerDirectModel
from jar.models.shared_initializer import SharedPretrainedInitializer
from jar.runtime import move, seed_all
from tools.benchmark_ruler_models import allocation_probe, enable, position_caches, summarize
from tools.cache_original_initializer import save_json
from tools.carla_compare import file_sha256
from tools.verify_position_cache import measure_alternating


def validate_protocol(protocol, models):
    """本次比较只允许四目完整帧、相同上参考，不能静默换参考。"""
    camera_ids = protocol["data"]["camera_ids"]
    reference = protocol["evaluation"]["reference_index"]
    if len(camera_ids) != 4 or not 0 <= reference < 4 or camera_ids[reference] != "stereo_up":
        raise ValueError("本次固定比较要求四目广角、上相机参考")
    if any(model.reference_index != reference for model in models.values()):
        raise ValueError("模型、baseline与共同协议的参考相机不一致")
    return reference


def validate_complete_inputs(inputs, protocol):
    height, width = protocol["data"]["image_size"]
    if inputs["rgb"].shape != (1, 4, 3, height, width):
        raise ValueError("完整帧比较要求batch=1、同步四目和协议图像尺寸")
    if not bool(inputs["view_mask"].all()):
        raise ValueError("complete清单包含缺失相机，应另列partial评测")
    if inputs.get("camera_ids") != [protocol["data"]["camera_ids"]]:
        raise ValueError("真实输入相机槽位与协议不一致")


def build_models(plan, device):
    """参数构建/恢复在普通模式完成，不创建 inference parameter。"""
    saved = torch.load(plan["original_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    original = saved.get("model", saved.get("state_dict", saved))
    models, metadata = {}, {}
    for name, settings in plan["models"].items():
        training = read_json(settings["training_config"])
        configuration = read_json(plan["original_config"])
        configuration["direct"] = {**training["model"], "reference_index": 2}
        model = RulerDirectModel(configuration, original_state=original)
        checkpoint = settings.get("checkpoint", "")
        if checkpoint:
            record = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
            if record["plan"]["model"] != training["model"]:
                raise ValueError("decoder checkpoint 架构与训练配置不一致")
            model.decoder.load_state_dict(record["model"], strict=True)
        elif not plan.get("random_decoder_smoke", False):
            raise ValueError("正式评测必须显式给出 decoder 权重")
        models[name] = model.to(device).eval()
        metadata[name] = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": file_sha256(checkpoint) if checkpoint else None,
            "training_config": training,
            "decoder_parameters": sum(value.numel() for value in model.decoder.parameters()),
            "total_parameters": sum(value.numel() for value in model.parameters()),
            "scene_image_size": [256, 448], "matching_image_size": [512, 896],
        }
    initializer = copy.deepcopy(plan["initializer"])
    if any(initializer.get(key) for key in ("readout_weights", "decoder_weights", "fusion_weights", "fusion_skip_indices")):
        raise ValueError("baseline 只加载已选原 CARLA 权重，不加载额外适配")
    initializer["cache_position_encoding"] = True
    models["mvsanywhere_firstpass"] = SharedPretrainedInitializer(initializer).to(device).eval()
    metadata["mvsanywhere_firstpass"] = {
        "checkpoint": initializer["weights"],
        "checkpoint_sha256": file_sha256(initializer["weights"]),
        "scene_image_size": [256, 448], "matching_image_size": [256, 448],
    }
    return models, metadata


@torch.inference_mode()
def compare_encoding(model, inputs, cached):
    live = model.encode(inputs)
    expected = cached["inputs"]
    differences = {}
    for name in ("matching_features", "K", "T_cam_from_rig", "view_mask", "image_mask"):
        value = expected[name].to(live[name].device)
        if name == "matching_features":
            value = value[None]
        torch.testing.assert_close(live[name], value, rtol=0, atol=0)
        differences[name] = float((live[name].float() - value.float()).abs().max())
    for index, (value, target) in enumerate(zip(live["raw_maps"], expected["raw_maps"])):
        target = target.to(value.device)
        torch.testing.assert_close(value, target, rtol=0, atol=0)
        differences[f"raw_{index}"] = float((value - target).abs().max())
    allowed = ("matching_features", "raw_maps", "K", "T_cam_from_rig", "view_mask",
               "image_mask", "matching_scale", "image_size", "reference_index")
    cached_inputs = move({key: expected[key] for key in allowed}, inputs["rgb"].device)
    actual, reference = model.decoder(live), model.decoder(cached_inputs)
    for stage in ("coarse", "fine"):
        torch.testing.assert_close(actual[stage]["depth"], reference[stage]["depth"], rtol=0, atol=0)
    torch.testing.assert_close(actual["depth"], reference["depth"], rtol=0, atol=0)
    differences["decoder_depth"] = float((actual["depth"] - reference["depth"]).abs().max())
    return differences


def validate_checkpoint(model, inputs, path, device):
    torch.save(model.checkpoint_record(), path)
    restored = RulerDirectModel.from_checkpoint(path).to(device).eval()
    position_caches(restored)
    with torch.inference_mode():
        a, b = model(inputs), restored(inputs)
        for field in ("dense_depth", "dense_valid", "dense_confidence", "dense_state"):
            torch.testing.assert_close(a[field], b[field], rtol=0, atol=0)
        difference = float((a["dense_depth"] - b["dense_depth"]).abs().max())
    del restored
    return {"checkpoint": str(path), "sha256": file_sha256(path), "max_depth_difference_m": difference}


def get_depth(result, name, reference):
    return result["depth"][0, 0] if name == "mvsanywhere_firstpass" else result["dense_depth"][0, reference]


@torch.inference_mode()
def validate_position_cache(model, controllers, inputs, name, reference):
    enable(controllers, False)
    uncached = get_depth(model(inputs), name, reference).clone()
    enable(controllers, True)
    cold = get_depth(model(inputs), name, reference).clone()
    hot = get_depth(model(inputs), name, reference).clone()
    torch.testing.assert_close(uncached, cold, rtol=0, atol=0)
    torch.testing.assert_close(uncached, hot, rtol=0, atol=0)
    return {"cold_max_difference_m": float((cold - uncached).abs().max()),
            "hot_max_difference_m": float((hot - uncached).abs().max()),
            "modules": [key for key, _ in controllers]}


def run(plan, device):
    if plan["split"] != "dev_complete":
        raise ValueError("本工具只允许 DEV，不读取 TEST")
    if device.type != "cuda":
        raise ValueError("同卡轮转计时需要实际 CUDA 设备")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    seed_all(plan.get("seed", 47))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(plan["output"])
    if (output / "summary.json").exists():
        raise FileExistsError("结果已存在，请保留已有记录并修改 output")
    output.mkdir(parents=True, exist_ok=True)
    protocol = read_json(plan["protocol"])
    reference = protocol["evaluation"]["reference_index"]
    dataset = comparison_dataset(protocol, plan["split"], load_targets=True)
    count = min(len(dataset), int(plan.get("limit", len(dataset))))
    models, metadata = build_models(plan, device)
    reference = validate_protocol(protocol, models)
    if count < 1 or not plan["timing_frames"] or any(index < 0 or index >= count for index in plan["timing_frames"]):
        raise ValueError("DEV及计时帧必须非空，计时索引必须位于本次评测范围")
    controllers = {name: position_caches(model) for name, model in models.items()}
    cache_index = read_json(Path(plan["cache_root"]) / "index.json")
    if cache_index["contract"]["protocol_sha256"] != file_sha256(plan["protocol"]):
        raise ValueError("冻结缓存与 RGB 验证的协议不同")
    if cache_index["contract"]["base_checkpoint_sha256"] != file_sha256(plan["original_checkpoint"]):
        raise ValueError("冻结缓存与完整模型的编码器来源不同")
    from tools.train_ruler_direct import load_direct_record

    cached = load_direct_record(plan["cache_root"], cache_index["sets"][plan["split"]][0], cache_index["contract"])
    first = move(collate_frames([dataset[0]]), device)
    validate_complete_inputs(first["inputs"], protocol)
    sample_id = f"{dataset.frames[0]['sequence']}/{dataset.frames[0]['frame_id']}"
    if sample_id != cached["sample_id"]:
        raise ValueError("缓存和真实 RGB 的首帧 ID 不匹配")
    checks = {}
    for name, model in models.items():
        check = {"position_cache": validate_position_cache(model, controllers[name], first["inputs"], name, reference)}
        if name != "mvsanywhere_firstpass":
            check["encoding_cache"] = compare_encoding(model, first["inputs"], cached)
            check["complete_checkpoint"] = validate_checkpoint(model, first["inputs"], output / (name + "_complete.pt"), device)
        # targets 与 inputs 是独立容器；改变实际 batch 标签后重新执行模型。
        with torch.inference_mode():
            expected = get_depth(model(first["inputs"]), name, reference).clone()
            changed = {**first, "targets": {key: torch.zeros_like(value) if torch.is_tensor(value) else value
                                           for key, value in first["targets"].items()}}
            actual = get_depth(model(changed["inputs"]), name, reference)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            check["targets_changed_max_difference_m"] = float((actual - expected).abs().max())
        checks[name] = check
        print(json.dumps({"verified": name, "checks": check}, ensure_ascii=False), flush=True)
    rows = {name: [] for name in models}
    timing_frames = []
    initial_depths = {}
    changed_outputs = {name: False for name in models}
    for index in range(count):
        batch = move(collate_frames([dataset[index]]), device)
        inputs = batch["inputs"]
        validate_complete_inputs(inputs, protocol)
        truth, valid = gt_reference(inputs, batch["targets"], protocol["evaluation"])
        sample_id = f"{dataset.frames[index]['sequence']}/{dataset.frames[index]['frame_id']}"
        for name, model in models.items():
            with torch.inference_mode():
                result = model(inputs)
            depth = get_depth(result, name, reference)
            streams = {"full": (depth, 1)}
            if name != "mvsanywhere_firstpass":
                streams.update({stage: (result["direct"][stage]["depth"][0, 0], stride)
                                for stage, stride in (("coarse", 8), ("fine", 4))})
            measurements = {}
            for stage, (prediction, stride) in streams.items():
                target, mask = truth[::stride, ::stride], valid[::stride, ::stride]
                if not bool((torch.isfinite(prediction[mask]) & (prediction[mask] > 0)).all()):
                    raise FloatingPointError("不能通过非法/零深度删除 GT 像素")
                measurements[stage] = evaluate_depth(prediction, torch.ones_like(mask), target, mask, protocol["evaluation"])
            rows[name].append({"sample_id": sample_id, "streams": measurements})
            if index == 0:
                initial_depths[name] = depth.clone()
            else:
                changed_outputs[name] |= not torch.equal(depth, initial_depths[name])
            del result
        if index in plan["timing_frames"]:
            timing_frames.append({"sample_id": sample_id, "inputs": inputs})
        print(json.dumps({"DEV_frame": index + 1, "frames": count}), flush=True)
    times, per_frame = {name: [] for name in models}, []
    for frame in timing_frames:
        timing = measure_alternating(models, frame["inputs"], device, plan["warmup"], plan["repeats"])
        for name, record in timing.items():
            times[name].extend(record["samples_ms"])
        per_frame.append({"sample_id": frame["sample_id"], "timing": timing})
    allocation = allocation_probe(models, timing_frames[-1]["inputs"], device)
    sources = ("tools/evaluate_ruler_direct.py", "jar/models/ruler_direct_model.py",
               "jar/models/ruler_direct.py", "jar/models/ruler_attention.py")
    report = {
        "plan": plan, "command": shlex.join([sys.executable, *sys.argv]), "frames": count,
        "gpu": torch.cuda.get_device_name(device), "precision": "FP32, TF32 disabled",
        "no_test_read": True, "random_decoder_smoke": plan.get("random_decoder_smoke", False),
        "checks": checks, "different_RGB_changes_depth": changed_outputs,
        "models": {name: {**metadata[name], "metrics": aggregate_depth(rows[name], protocol["evaluation"]),
                          "timing": summarize(times[name]), "allocation": allocation[name]} for name in models},
        "timing_frames": per_frame, "source_sha256": {name: file_sha256(name) for name in sources},
        "manifest_sha256": file_sha256(protocol["data"]["manifests"][plan["split"]]),
        "protocol_sha256": file_sha256(plan["protocol"]),
        "notes": ["网络时间含编码器、直接候选生成和深度上采样；不含读盘/H2D/GT评价。",
                  "统一四目 256×448、上参考全图输出；自研另用同帧 512×896 匹配图，baseline匹配使用256×448。",
                  "相同位置插值缓存；不缓存图像特征或深度；训练缓存仅用于首帧等同性检查。",
                  "未知像素保留合法深度假设并纳入全GT指标；深度假设不是已验证表面。",
                  "显存报告为所有模型同驻留时的单次调用增量，不是独立模型总显存。"],
    }
    for name, records in rows.items():
        save_json(output / (name + "_frames.json"), records)
    save_json(output / "summary.json", report)
    print(json.dumps({name: {"MAE_m": report["models"][name]["metrics"]["full"]["all"]["MAE_m"],
                            "median_ms": report["models"][name]["timing"]["median_ms"]} for name in models}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:3")
    arguments = parser.parse_args()
    run(read_json(arguments.config), torch.device(arguments.device))
