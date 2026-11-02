"""离线检查公共多视角清单与真实权重；不是正式精度或速度评测。

配置提供 datasets（name/root/manifest/output）、own_checkpoint、initializer，
以及可选 mvsformerpp（config/weights）。每套固定检查首、中、尾三个样本。
所有模型使用同一 RGB/标定；标签只用于展示和独立的一步训练检查。
"""

import argparse
import copy
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from jar.data import CarlaFrameDataset, collate_frames
from jar.runtime import move
from official_methods.data import FourCameraDataset, resize_sample


def offline_only():
    """本工具没有下载功能；额外拒绝 Python 发起的任何 IP 连接。"""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    def deny_ip_connection(event, args):
        if event == "socket.connect" and args[0].family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("烟测必须离线：禁止联网，所有权重必须已存在本地")

    sys.addaudithook(deny_ip_connection)


def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"本地必要文件不存在：{path}")
    return path


def fingerprint(inputs):
    digest = hashlib.sha256()
    for key, value in sorted(inputs.items()):
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check_resize(raw, resized):
    """检查实际像素中心缩放与 K 更新，不使用深度标签。"""
    height, width = raw["inputs"]["rgb"].shape[-2:]
    new_height, new_width = resized["inputs"]["rgb"].shape[-2:]
    scale = torch.tensor([new_width / width, new_height / height])
    rays = torch.tensor([[0.0, 0.0, 1.0], [0.2, -0.3, 1.0], [-0.4, 0.1, 2.0]])
    old = rays @ raw["inputs"]["K"].transpose(-1, -2)
    new = rays @ resized["inputs"]["K"].transpose(-1, -2)
    expected = (old[..., :2] / old[..., 2:] + 0.5) * scale - 0.5
    actual = new[..., :2] / new[..., 2:]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)
    return float((actual - expected).abs().max())


def prepare_samples(settings, config):
    root, manifest = settings["root"], settings["manifest"]
    units = settings.get("geometry_units", "meters")
    dataset = CarlaFrameDataset(root, manifest, "test", geometry_units=units)
    unlabeled = CarlaFrameDataset(root, manifest, "test", load_targets=False, geometry_units=units)
    baseline = FourCameraDataset({
        "data": {"root": root, "test_manifest": manifest, "geometry_units": units,
                 "num_views": config.get("num_views", 4)},
        "image_size": config["image_size"],
    }, "test")
    if len(dataset) < 3:
        raise ValueError("每套至少需要三个真实多视角样本，不能复制补足")
    indices = [0, len(dataset) // 2, len(dataset) - 1]
    prepared = []
    for index in indices:
        raw = dataset[index]
        sample = resize_sample(raw, config["image_size"])
        image_only = resize_sample(unlabeled[index], config["image_size"])
        official = baseline[index]
        assert not image_only["targets"]
        for key, value in sample["inputs"].items():
            torch.testing.assert_close(value, image_only["inputs"][key], rtol=0, atol=0)
            torch.testing.assert_close(value, official["inputs"][key], rtol=0, atol=0)
        for key, value in sample["targets"].items():
            torch.testing.assert_close(value, official["targets"][key], rtol=0, atol=0)
        matching = resize_sample(raw, config["matching_image_size"])
        errors = [check_resize(raw, sample), check_resize(raw, matching)]
        inputs = collate_frames([sample])["inputs"]
        without_targets = collate_frames([image_only])["inputs"]
        for destination in (inputs, without_targets):
            destination["matching_rgb"] = matching["inputs"]["rgb"][None]
            destination["matching_image_mask"] = matching["inputs"]["image_mask"][None]
        assert fingerprint(inputs) == fingerprint(without_targets)
        prepared.append({
            "index": index, "sample": sample, "inputs": inputs,
            "without_targets": without_targets,
            "targets": collate_frames([sample])["targets"],
            "input_sha256": fingerprint(inputs),
            "resize_projection_max_error_px": max(errors),
            "raw_image_size": list(raw["inputs"]["rgb"].shape[-2:]),
            "predictions": {},
        })
    return prepared, len(dataset)


def model_factories(config):
    """均严格加载完整本地权重；禁止把随机结构测试当成预训练结果。"""
    from jar.models.ruler_direct_model import RulerDirectModel
    from jar.models.shared_initializer import SharedPretrainedInitializer

    require_file(config["own_checkpoint"])
    require_file(config["initializer"]["weights"])
    require_file(config["initializer"]["model_config"])
    def build_own():
        model = RulerDirectModel.from_checkpoint(config["own_checkpoint"])
        if "reference_indices" in config:
            model.settings["reference_indices"] = list(config["reference_indices"])
            model.settings["reference_index"] = config["reference_index"]
            model.reference_index = config["reference_index"]
        return model

    factories = [
        ("own_scale_transport", build_own),
        ("MVSAnywhere_firstpass", lambda: SharedPretrainedInitializer(config["initializer"])),
    ]
    for settings in config.get("extra_baselines", []):
        require_file(settings["weights"])
        require_file(settings["model_config"])

        def build_extra(settings=settings):
            from tools.carla_compare import make_model
            model, _ = make_model(settings, settings["weights"], torch.device("cpu"))
            return model

        factories.append((settings["method"], build_extra))
    former = config.get("mvsformerpp", {})
    if former.get("weights"):
        require_file(former["weights"])
        native_config = json.loads(require_file(former["config"]).read_text())
        native_config = copy.deepcopy(native_config)
        native_config["model"].update(
            weights=former["weights"], dinov2_weights="", allow_random_init=False,
            reference_indices=[config["reference_index"]],
        )
        native_config.setdefault("evaluation", {})["reference_indices"] = [config["reference_index"]]

        def build_former():
            from official_methods.methods.mvsformerpp import Adapter
            return Adapter(native_config)

        factories.append(("MVSFormerPP", build_former))
    return factories


def depth_output(name, output, reference):
    if name == "own_scale_transport":
        return output["dense_depth"][0, reference], output["dense_valid"][0, reference]
    depth = output["depth"][0, 0]
    return depth, torch.isfinite(depth) & (depth > 0)


def training_step(model, prepared, device, learning_rate, supervision_range=None):
    """只验证一次原生损失/反向/更新，更新后的模型随即丢弃。"""
    model.train()
    if supervision_range is not None:
        model.cfg.setdefault("loss", {})["depth_range_m"] = list(supervision_range)
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    inputs = move(prepared["inputs"], device)
    targets = move(prepared["targets"], device)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs)
    losses = model.training_loss(prediction, inputs, targets)
    if int(losses["supervised_pixels"]) == 0:
        raise ValueError("固定训练检查样本在模型监督范围内没有有效标签")
    if not all(torch.isfinite(value).all() for value in losses.values()):
        raise ValueError("训练损失出现 NaN/Inf")
    losses["total"].backward()
    gradients = [value.grad for value in parameters if value.grad is not None]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise ValueError("训练梯度缺失或出现 NaN/Inf")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise ValueError("训练梯度范数非有限或为零")
    optimizer.step()
    if not all(torch.isfinite(value).all() for value in parameters):
        raise ValueError("优化后参数出现 NaN/Inf")
    return {
        "status": "passed", "optimizer_steps": 1, "checkpoint_saved": False,
        "learning_rate": learning_rate, "loss_total": float(losses["total"].detach()),
        "gradient_norm_before_clip": float(norm),
        "trainable_parameter_tensors": len(parameters),
        "parameter_tensors_with_gradient": len(gradients),
        "supervised_pixels": int(losses["supervised_pixels"]),
        "supervision_depth_range": model.cfg.get("loss", {}).get("depth_range_m"),
        "note": "仅管线检查；更新发生在所有预测保存之后，不用于后续样本或比较。",
    }


def baseline_training_step(prepared, config, device):
    """重新加载完整 Hero 的原生训练分支；不对冻结推理包装器求伪梯度。"""
    from official_methods.methods.mvsanywhere import Adapter
    from jar.models.shared_initializer import _load_projective_state

    settings = config["initializer"]
    native_config = json.loads(require_file(settings["model_config"]).read_text())
    native_config["_loading_shared_checkpoint"] = True
    native_config["model"].update(pretrained=False, weights="", da_weights="")
    native_config["model"]["num_views"] = config.get("num_views", 4)
    native_config["model"]["reference_indices"] = [config["reference_index"]]
    native_config.setdefault("training", {})["geometric_flip"] = False
    model = Adapter(native_config)
    payload = torch.load(settings["weights"], map_location="cpu", weights_only=False, mmap=True)
    state = payload.get("model", payload.get("state_dict", payload))
    target = model if any(key.startswith("native_model.") for key in state) else model.native_model
    _load_projective_state(target, state)
    model.to(device).train()
    inputs = move(prepared["inputs"], device)
    targets = move(prepared["targets"], device)
    near, far = config.get("training_supervision_depth_range",
                           native_config.get("data", {}).get("supervision_depth_range", [2.0, 80.0]))
    targets = dict(targets)
    targets["depth_valid"] = targets["depth_valid"] & (targets["depth"] >= near) & (targets["depth"] <= far)
    if not bool(targets["depth_valid"][:, config["reference_index"]].any()):
        raise ValueError("Hero 固定训练样本在监督范围内没有有效深度")
    inputs = model.prepare_training_inputs(inputs, targets)
    optimizer, _ = model.configure_optimizer()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs)
    losses = model.loss(prediction, inputs, targets)
    if not all(torch.isfinite(value).all() for value in losses.values()):
        raise ValueError("Hero 原生损失出现 NaN/Inf")
    losses["total"].backward()
    parameters = [value for value in model.parameters() if value.requires_grad]
    gradients = [value.grad for value in parameters if value.grad is not None]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise ValueError("Hero 原生梯度缺失或出现 NaN/Inf")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise ValueError("Hero 原生训练梯度范数无效")
    optimizer.step()
    if not all(torch.isfinite(value).all() for value in parameters):
        raise ValueError("Hero 原生优化后参数出现 NaN/Inf")
    return {
        "status": "passed", "optimizer_steps": 1, "checkpoint_saved": False,
        "loss_total": float(losses["total"].detach()),
        "gradient_norm_before_clip": float(norm),
        "parameter_tensors_with_gradient": len(gradients),
        "supervision_depth_range": [near, far],
        "note": "原生完整 Hero 损失及优化器；仅训练阶段从标签进行官方范围增强。临时更新丢弃。",
    }


def save_preview(prepared, output, reference, far, units="meters"):
    width, height, label_height = 448, 256, 28
    maps = {"GT optical Z": prepared["sample"]["targets"]["depth"][reference].numpy()}
    maps.update(prepared["predictions"])
    columns = max(4, len(maps))
    canvas = Image.new("RGB", (width * columns, 2 * (height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, rgb in enumerate(prepared["sample"]["inputs"]["rgb"]):
        pixels = (rgb.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        x = index * width
        canvas.paste(Image.fromarray(pixels).resize((width, height)), (x, label_height))
        label = f"View {index}" + (" / reference" if index == reference else "")
        draw.text((x + 7, 7), label, fill="black")
    for index, (name, depth) in enumerate(maps.items()):
        value = np.nan_to_num(depth, nan=0, posinf=far, neginf=0)
        image = Image.fromarray((value.clip(0, far) / far * 255).astype(np.uint8)).convert("RGB")
        x, y = index * width, height + label_height
        canvas.paste(image.resize((width, height)), (x, y + label_height))
        draw.text((x + 7, y + 7), f"{name} | 0-{far:g} {units}", fill="black")
    canvas.save(output)


def run(config):
    offline_only()
    os.chdir(PROJECT)
    torch.set_num_threads(2)
    torch.manual_seed(config.get("seed", 47))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config.get("device", "cuda:0"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("此入口要求实际 CUDA GPU；不能自动用 CPU 冒充 GPU 测试")
    torch.cuda.set_device(device)
    reference = config["reference_index"]
    factories = model_factories(config)
    for settings in config["datasets"]:
        prepared, total = prepare_samples(settings, config)
        output_dir = Path(settings["output"])
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "dataset": settings["name"], "manifest_samples": total, "tested_samples": len(prepared),
            "geometry_units": settings.get("geometry_units", "meters"),
            "sample_indices": [item["index"] for item in prepared],
            "config": config, "dataset_config": settings,
            "gpu": torch.cuda.get_device_name(device), "python_ip_network_blocked": True,
            "shared_loader_equal": True, "inputs_equal_without_targets": True,
            "main_image_size": config["image_size"],
            "matching_image_size": config["matching_image_size"],
            "samples": [], "models": {},
            "notes": [
                "本次只验证数据/真实权重推理/一小步训练可运行，不报告精度或速度排名。",
                "以样本的sample_type/synchronized_capture为准；静态多位置与同步双目均不等于CARLA同步四目。",
                "这些公共数据集可能被 MVSAnywhere 预训练使用，不能宣称未见数据零样本评测。",
                "匹配图像从原始 RGB 单独缩放，不将低分辨率主图放大作为匹配输入。",
                "自研实际匹配编码采用 matching_rgb；Hero 首遍包装器只读取主 rgb，忽略 matching_rgb。收到相同字典不等于实际匹配分辨率相同。",
                "自研原有数值候选范围与baseline原生范围机制保持不变；不从标签确定范围。scene_unit数据不称米制精度。",
                *settings.get("notes", []),
            ],
        }
        for item in prepared:
            report["samples"].append({
                "index": item["index"], "metadata": item["sample"]["metadata"],
                "input_sha256": item["input_sha256"], "raw_image_size": item["raw_image_size"],
                "resize_projection_max_error_px": item["resize_projection_max_error_px"],
            })
        if not config.get("mvsformerpp", {}).get("weights"):
            report["models"]["MVSFormerPP"] = {
                "status": "not_run", "reason": "未配置完整本地任务权重，不使用随机初始化替代。",
            }
        for name, factory in factories:
            model = factory().to(device).eval()
            if hasattr(model, "reference_index") and model.reference_index != reference:
                raise ValueError(f"{name} 权重参考槽位与配置不一致")
            model_report = {"status": "passed", "samples": []}
            deterministic = name == "fast_mvsnet"
            model_report["deterministic_cudnn"] = deterministic
            for item in prepared:
                inputs = move(item["inputs"], device)
                input_only = move(item["without_targets"], device)
                # 原生 Fast-MVSNet 的 GN 硬门控会放大卷积的微小数值扰动。
                # 仅限制 GPU 算子执行方式，不改变其网络、权重或细化算法。
                execution = torch.backends.cudnn.flags(
                    deterministic=True, benchmark=False, allow_tf32=False,
                ) if deterministic else nullcontext()
                with torch.inference_mode(), execution:
                    first = model(inputs)
                    second = model(input_only)
                    depth, valid = depth_output(name, first, reference)
                    repeated, repeated_valid = depth_output(name, second, reference)
                if name in ("own_scale_transport", "MVSAnywhere_firstpass") and list(depth.shape) != config["image_size"]:
                    raise ValueError(f"{name} 输出尺寸不符合接口：{tuple(depth.shape)}")
                if not torch.isfinite(depth).all() or not valid.any():
                    raise ValueError(f"{name} 没有有限有效深度输出")
                torch.testing.assert_close(depth, repeated, rtol=0, atol=0)
                torch.testing.assert_close(valid, repeated_valid, rtol=0, atol=0)
                if fingerprint(inputs) != item["input_sha256"]:
                    raise ValueError(f"{name} 原地改变了共享输入")
                array = depth.cpu().numpy().astype(np.float32)
                item["predictions"][name] = array
                stem = f"sample_{item['index']:06d}_{name}"
                np.save(output_dir / f"{stem}_depth.npy", array, allow_pickle=False)
                np.save(output_dir / f"{stem}_valid.npy", valid.cpu().numpy(), allow_pickle=False)
                if "depth_K" in first:
                    np.save(output_dir / f"{stem}_K.npy", first["depth_K"].cpu().numpy(), allow_pickle=False)
                model_report["samples"].append({
                    "index": item["index"], "shape": list(depth.shape),
                    "finite_fraction": float(torch.isfinite(depth).float().mean()),
                    "valid_fraction": float(valid.float().mean()),
                    "depth_min_value": float(depth[valid].min()),
                    "depth_max_value": float(depth[valid].max()),
                    "targets_removed_prediction_max_difference": float((depth - repeated).abs().max()),
                })
                del first, second, depth, repeated, inputs, input_only
            if name == "own_scale_transport" and config.get("training_smoke", True):
                try:
                    model_report["training"] = training_step(
                        model, prepared[0], device, config.get("training_smoke_learning_rate", 1e-6),
                        config.get("training_supervision_depth_range"),
                    )
                except Exception as error:
                    model_report["training"] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                    model_report["status"] = "forward_passed_training_failed"
            report["models"][name] = model_report
            print(json.dumps({"dataset": settings["name"], "model": name, **model_report}), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            if name == "MVSAnywhere_firstpass" and config.get("training_smoke", True):
                try:
                    model_report["training"] = baseline_training_step(prepared[0], config, device)
                except Exception as error:
                    model_report["training"] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                    model_report["status"] = "forward_passed_training_failed"
                gc.collect()
                torch.cuda.empty_cache()
                print(json.dumps({"dataset": settings["name"], "model": name,
                                  "training": model_report["training"]}), flush=True)
                (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        for item in prepared:
            save_preview(item, output_dir / f"preview_{item['index']:06d}.jpg", reference,
                         config.get("preview_max_depth_m", 80.0), settings.get("geometry_units", "meters"))
        print(f"完成：{output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON 配置；路径相对项目根目录")
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))
