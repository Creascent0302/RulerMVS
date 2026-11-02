"""离线公共数据深度评测，明确区分米与原场景单位。

复用 smoke_outdoor_datasets 的本地模型入口。配置额外提供 methods、
evaluation.depth_range/limit，以及每套 datasets[].evaluation_output。
只输出深度指标；不将 scene_unit 数据冒充米制表面或占据评测。
"""

import argparse
from contextlib import nullcontext
import gc
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from jar.data import build_dataset, collate_frames
from jar.runtime import move, seed_all
from official_methods.metrics import depth_statistics
from official_methods.postprocess import depth_on_image_grid
from tools.smoke_outdoor_datasets import depth_output, model_factories, offline_only


def finish_statistics(sums, units):
    """绝对误差使用显式单位；缺失预测在 delta 分母中记失败。"""
    if units not in ("meters", "scene_unit"):
        raise ValueError("geometry_units 必须明确为 meters 或 scene_unit")
    count, total = sums["predicted_pixels"], sums["gt_pixels"]
    return {
        "geometry_units": units,
        "MAE": sums["abs_sum"] / count if count else None,
        "RMSE": math.sqrt(sums["sq_sum"] / count) if count else None,
        "AbsRel": sums["abs_rel_sum"] / count if count else None,
        **{f"delta_{index}": sums[f"delta_{index}_hits"] / total if total else None
           for index in (1, 2, 3)},
        "prediction_coverage": count / total if total else None,
        "unknown_or_missing_fraction": 1 - count / total if total else None,
        "gt_pixels": total,
        "predicted_pixels": count,
        "denominators": "MAE/RMSE/AbsRel按有效预测像素累计；delta按固定有效GT累计，缺失预测记失败。",
    }


def prediction_on_main_grid(name, prediction, inputs, reference, convention):
    """原生 K 优先；没有 K 的低分辨率输出须显式声明半像素格点。"""
    depth, valid = depth_output(name, prediction, reference)
    native_size = list(depth.shape)
    target_size = inputs["rgb"].shape[-2:]
    if "depth_K" in prediction:
        converted = depth_on_image_grid(prediction, inputs)
        refs = converted["reference_indices"]
        if refs != [reference]:
            raise ValueError("一次深度比较必须实际输出配置指定的单个参考相机")
        return converted["depth"][0, 0], converted["depth_valid"][0, 0], {
            "native_shape": native_size,
            "mapping": "depth_K -> input_K; bilinear grid_sample, align_corners=False",
        }
    refs = prediction.get("reference_indices")
    if refs is not None and list(refs) != [reference]:
        raise ValueError("模型实际参考相机与配置不一致")
    if name != "own_scale_transport":
        declared = prediction.get("depth_valid", prediction.get("valid"))
        if torch.is_tensor(declared):
            while declared.ndim > 2:
                declared = declared[0]
            if declared.shape != valid.shape:
                raise ValueError("原生深度有效掩码尺寸不一致")
            valid &= declared.bool()
    valid = valid & torch.isfinite(depth) & (depth > 0)
    mapping = "native output already on main image grid"
    if tuple(depth.shape) != tuple(target_size):
        if convention != "half_pixel":
            raise ValueError("低分辨率深度缺少 depth_K，须显式配置 low_resolution_convention=half_pixel")
        clean = torch.where(valid, depth, 0)
        weights = F.interpolate(valid[None, None].float(), target_size, mode="bilinear", align_corners=False)[0, 0]
        depth = F.interpolate(clean[None, None].float(), target_size, mode="bilinear", align_corners=False)[0, 0]
        depth = depth / weights.clamp_min(1e-8)
        valid = weights > 0.999
        mapping = "explicit half_pixel bilinear interpolate, align_corners=False; invalid support retained"
    return depth, valid, {"native_shape": native_size, "mapping": mapping}


def dataset_config(settings, config):
    return {"data": {
        "protocol": "calibrated_multiview",
        "root": settings["root"],
        "test_manifest": settings["manifest"],
        "num_views": config.get("num_views", 4),
        "geometry_units": settings["geometry_units"],
        "image_size": config["image_size"],
        "matching_image_size": config["matching_image_size"],
        "depth_resize_mode": "nearest-exact",
    }}


def write_report(path, report):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def run(config):
    offline_only()
    evaluation = config["evaluation"]
    near, far = map(float, evaluation["depth_range"])
    if not (math.isfinite(near) and math.isfinite(far) and 0 < near < far):
        raise ValueError("depth_range 须在数据原单位内显式满足 0 < near < far")
    reference = int(config["reference_index"])
    if not 0 <= reference < config.get("num_views", 4):
        raise ValueError("reference_index 超出真实视角数量")
    if config["initializer"]["reference_index"] != reference:
        raise ValueError("MVSAnywhere 与共同协议参考相机不同")
    config["reference_indices"] = [reference]
    methods = config["methods"]
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("methods 不能为空或包含重复项")
    factories = dict(model_factories(config))
    if set(methods) - factories.keys():
        raise ValueError(f"未知或缺少本地权重的模型：{set(methods) - factories.keys()}")
    device = torch.device(config.get("device", "cuda:0"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(config.get("cpu_threads", 4))
    seed_all(config.get("seed", 47))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for settings in config["datasets"]:
        units = settings["geometry_units"]
        if units not in ("meters", "scene_unit"):
            raise ValueError("每套数据须明确 geometry_units")
        output = Path(settings["evaluation_output"])
        output.mkdir(parents=True, exist_ok=True)
        summary_path = output / "summary.json"
        if summary_path.exists():
            raise FileExistsError(f"保留已有评测：{summary_path}；请修改配置中的 evaluation_output")
        data_config = dataset_config(settings, config)
        labels = build_dataset(data_config, "test", load_targets=True)
        images = build_dataset(data_config, "test", load_targets=False)
        limit = int(evaluation.get("limit", len(images)))
        if limit < 1:
            raise ValueError("evaluation.limit 必须为正整数")
        indices = list(range(min(limit, len(images))))
        report = {
            "completed": False, "dataset": settings, "geometry_units": units,
            "config": config, "manifest_samples": len(images), "indices": indices,
            "reference_index": reference, "methods": {},
            "protocol": {
                "input": "仅RGB与标定；targets在预测完成后单独读取",
                "image_size": config["image_size"], "matching_image_size": config["matching_image_size"],
                "matching_source": "由同一原图独立缩放，未将主图放大",
                "depth_range": [near, far], "depth_range_units": units,
                "gt_mask": "GT有效 & 有限正深度 & 固定范围 & 参考图输入有效域；不依模型置信度筛选GT",
                "gt_resampling": "nearest-exact", "aggregation": "所有有效像素累计，非帧均值",
                "model_search_range": "从完整模型/本地配置保留，不读取测试标签设置；可在各模型原配置审计",
                "network": "离线；未运行训练、点云、占据或计时评测",
                "scope": "小样本入口自检，不能据此做完整测试集排名" if len(indices) < len(images) else "全清单深度评测",
            },
        }
        write_report(summary_path, report)
        for name in methods:
            model = factories[name]().to(device).eval()
            if hasattr(model, "reference_index") and model.reference_index != reference:
                raise ValueError(f"{name} 实际参考相机不等于配置")
            sums, rows = {}, []
            for index in indices:
                sample = images[index]
                inputs = move(collate_frames([sample])["inputs"], device)
                context = torch.backends.cudnn.flags(deterministic=True, benchmark=False, allow_tf32=False) if name == "fast_mvsnet" else nullcontext()
                with torch.inference_mode(), context:
                    prediction = model(inputs)
                    depth, valid, mapping = prediction_on_main_grid(
                        name, prediction, inputs, reference,
                        evaluation.get("low_resolution_convention"),
                    )
                labeled = labels[index]
                for key, value in sample["inputs"].items():
                    torch.testing.assert_close(value, labeled["inputs"][key], rtol=0, atol=0)
                if "depth" not in labeled["targets"]:
                    raise ValueError("该样本缺少深度标签，不能静默跳过")
                target = move(labeled["targets"], device)
                truth = target["depth"][reference]
                mask = target["depth_valid"][reference] & inputs["image_mask"][0, reference]
                mask &= inputs["view_mask"][0, reference]
                stats = depth_statistics(depth.double(), truth.double(), mask, valid, near, far)
                for key, value in stats.items():
                    sums[key] = sums.get(key, 0) + value
                rows.append({"index": index, "metadata": sample["metadata"], "resampling": mapping,
                             "metrics": finish_statistics(stats, units)})
                del prediction, inputs, target
            report["methods"][name] = {"frames": rows, "metrics": finish_statistics(sums, units)}
            write_report(summary_path, report)
            print(json.dumps({"dataset": settings["name"], "method": name, "samples": len(rows),
                              "geometry_units": units, "completed": True}, ensure_ascii=False), flush=True)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        report["completed"] = True
        write_report(summary_path, report)


def self_test():
    """只核验指标分母、单位声明和恒深度格点处理。"""
    truth = torch.full((2, 2), 10.0)
    prediction = torch.full_like(truth, 12.0)
    gt_valid = torch.ones_like(truth, dtype=torch.bool)
    pred_valid = gt_valid.clone()
    pred_valid[0, 0] = False
    statistics = depth_statistics(prediction, truth, gt_valid, pred_valid, 1.0, 20.0)
    for units in ("meters", "scene_unit"):
        result = finish_statistics(statistics, units)
        assert result["geometry_units"] == units
        assert result["MAE"] == 2 and result["RMSE"] == 2
        assert abs(result["AbsRel"] - 0.2) < 1e-6
        assert result["prediction_coverage"] == 0.75 and result["delta_1"] == 0.75
        assert not any(key.endswith("_m") for key in result)
    empty = depth_statistics(prediction, truth, gt_valid, torch.zeros_like(gt_valid), 1, 20)
    assert finish_statistics(empty, "scene_unit")["MAE"] is None
    assert finish_statistics(empty, "scene_unit")["delta_1"] == 0
    output = {"depth": torch.full((1, 1, 2, 2), 7.0), "reference_indices": [0]}
    inputs = {"rgb": torch.zeros(1, 2, 3, 4, 4)}
    depth, valid, _ = prediction_on_main_grid("MVSAnywhere_firstpass", output, inputs, 0, "half_pixel")
    assert depth.shape == (4, 4) and (depth == 7).all() and valid.all()
    inputs.update(
        K=torch.tensor([[[[4.0, 0, 1.5], [0, 4.0, 1.5], [0, 0, 1]]] * 2]),
        image_mask=torch.ones(1, 2, 4, 4, dtype=torch.bool),
        view_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    output["depth_K"] = torch.tensor([[[[2.0, 0, 0.5], [0, 2.0, 0.5], [0, 0, 1]]]])
    depth, valid, mapping = prediction_on_main_grid("MVSAnywhere_firstpass", output, inputs, 0, None)
    assert (depth == 7).all() and valid.all() and mapping["mapping"].startswith("depth_K")
    try:
        finish_statistics(statistics, "unspecified")
    except ValueError:
        pass
    else:
        raise AssertionError("未知单位应明确拒绝")
    print("公共深度评测自检通过：恒深度、缺失预测、固定GT分母、显式单位、低分辨率插值。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    elif arguments.config:
        run(json.loads(Path(arguments.config).read_text()))
    else:
        parser.error("需要 --config，或显式 --self-test")
