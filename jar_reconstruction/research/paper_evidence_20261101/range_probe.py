"""固定176组VAL的范围约束诊断；共同TRAIN裁剪只作独立后处理对照。"""

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import add_statistics, digest, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from research.near_mid_evaluation_20261028.metrics import depth_metrics, depth_sums, ground_truth, regions
from research.rulermvs_v6_20261101.evaluation import Models
from research.rulermvs_v6_20261101.speed_final import initialize


METHODS = ("v6_best", "mvsanywhere_refined")
REGIONS = ("full", "near", "mid", "far")


def clip_valid(depth, valid, low, high):
    """裁剪已有正有限预测，不把NaN、零深度或无效槽修复成有效预测。"""
    usable = valid & torch.isfinite(depth) & (depth > 0)
    return torch.where(usable, depth.clamp(low, high), depth)


def range_sums(depth, valid, truth, mask, low, high):
    usable = mask & valid & torch.isfinite(depth) & (depth > 0)
    lower = usable & (depth < low)
    upper = usable & (depth > high)
    outside = lower | upper
    clipped = clip_valid(depth, valid, low, high)
    error = (depth.double() - truth).abs()
    clipped_error = (clipped.double() - truth).abs()
    changed = usable & (clipped != depth)
    epsilon = 1e-5 * max(1., high)
    material = usable & ((depth < low - epsilon) | (depth > high + epsilon))
    gt_outside = mask & ((truth < low) | (truth > high))
    return {
        "gt_pixels": int(mask.sum()), "predicted_pixels": int(usable.sum()),
        "gt_outside_train_pixels": int(gt_outside.sum()),
        "pred_below_train_pixels": int(lower.sum()), "pred_above_train_pixels": int(upper.sum()),
        "pred_outside_train_pixels": int(outside.sum()),
        "pred_outside_train_with_tolerance_pixels": int(material.sum()),
        "raw_abs_sum": float(error[usable].sum()), "raw_sq_sum": float(error[usable].square().sum()),
        "outside_abs_sum": float(error[outside].sum()), "outside_sq_sum": float(error[outside].square().sum()),
        "clip_abs_reduction_sum": float((error[usable] - clipped_error[usable]).sum()),
        "clip_changed_pixels": int(changed.sum()),
        "clip_improved_pixels": int((changed & (clipped_error < error)).sum()),
        "clip_worsened_pixels": int((changed & (clipped_error > error)).sum()),
        "outside_prediction_gt_inside_pixels": int((outside & ~gt_outside).sum()),
        "outside_prediction_gt_outside_pixels": int((outside & gt_outside).sum()),
    }


def finish_range(values):
    result = dict(values)
    ratio = lambda a, b: values[a] / values[b] if values[b] else None
    result.update(
        gt_outside_train_fraction=ratio("gt_outside_train_pixels", "gt_pixels"),
        pred_outside_train_fraction=ratio("pred_outside_train_pixels", "predicted_pixels"),
        outside_abs_error_share=ratio("outside_abs_sum", "raw_abs_sum"),
        outside_sq_error_share=ratio("outside_sq_sum", "raw_sq_sum"),
        clip_MAE_reduction=ratio("clip_abs_reduction_sum", "predicted_pixels"),
    )
    return result


def prepare(settings):
    plan = json.loads(Path(settings["evaluation_plan"]).read_text())
    selected = json.loads(Path(settings["diagnostics"]).read_text())["plans"]
    configs = {}
    for phase in ("public", "carla_new"):
        filename = plan["groups"][phase + "_main"]["val_config"]
        config = json.loads(Path(filename).read_text())
        config["methods"] = {key: config["methods"][key] for key in METHODS}
        for domain in config["datasets"]:
            choice = selected[domain["name"]]
            if digest(domain["manifests"]["val"]) != choice["manifest_sha256"]:
                raise ValueError(f"{domain['name']}的固定VAL清单已改变")
            if not 0 < domain["near"] < domain["far"]:
                raise ValueError("TRAIN范围无效")
        configs[phase] = config
    return configs, selected


def forward_with_ranges(models, method, domain, inputs):
    calls = []
    handle = None
    if method == "mvsanywhere_refined" and int(inputs["view_mask"].sum()) >= 2:
        active = int(inputs["view_mask"][0].sum())
        native_domain = dict(domain, num_views=active) if not bool(inputs["view_mask"].all()) else domain
        native = models.get(method, native_domain).native_model

        def capture(module, args, kwargs):
            calls.append({key: torch.as_tensor(kwargs[key]).detach().cpu().flatten().tolist()
                          for key in ("min_depth", "max_depth")})

        handle = native.cost_volume.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.inference_mode():
            prediction = models.forward(method, domain, inputs)
            depth, valid = models.convert(prediction, inputs)
    finally:
        if handle is not None:
            handle.remove()
    return depth.cpu(), valid.cpu(), calls


def run(settings):
    configs, selected = prepare(settings)
    output = Path(settings["output"])
    if not output.resolve().is_relative_to(Path("/data")):
        raise ValueError("结果须经数据盘软链接保存")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent.resolve()).free < settings.get("minimum_free_gib", 200) * 2 ** 30:
        raise RuntimeError("数据盘空闲空间不足")
    initialize(settings)
    output.mkdir(parents=True)
    device = torch.device("cuda", settings.get("gpu", 0))
    report = {"completed": False, "split": "val", "test_accessed": False,
              "new_training": False, "settings": settings, "datasets": {}, "weights_sha256": {},
              "evaluation_plan_sha256": digest(settings["evaluation_plan"]),
              "diagnostics_sha256": digest(settings["diagnostics"]),
              "notes": [
                  "原主表未修改；本报告只检查原输出与共同TRAIN范围裁剪的差别。",
                  "MVSAnywhere每遍由sigmoid限制于当前范围，但官方范围细化可突破首遍TRAIN边界。",
                  "V6候选连续偏移、族内均值和凸上采样在固定TRAIN范围内。",
                  "裁剪不读取GT，原valid与非有限/非正预测保持，不能靠裁剪恢复有效率。",
                  "公共GT不默认裁剪到TRAIN；CARLA已有eval_range独立规定GT评价范围。",
                  "GT越过TRAIN时裁剪可能更差，因而保留该GT及其统计，不隐藏失败。",
                  "只沿用预登记11域各16VAL索引，无新TEST或超参数选择。",
              ]}
    for phase, config in configs.items():
        models = Models(config, device)
        report["weights_sha256"][phase] = {k: digest(v["weights"]) for k, v in config["methods"].items()}
        for domain in config["datasets"]:
            name, low, high = domain["name"], float(domain["near"]), float(domain["far"])
            choice = selected[name]
            dataset = build_dataset(dataset_config(domain, config), "val", load_targets=True)
            totals = {m: {"raw": {}, "clipped": {}, "range": {}} for m in METHODS}
            frames = []
            # 在inference_mode外构建网络；模型初始化不受标签或预测范围裁剪影响。
            for method in METHODS:
                models.get(method, domain)
            for index in choice["indices"]:
                batch = collate_frames([dataset[index]])
                inputs = move(batch["inputs"], device)
                truth, mask = ground_truth(batch, domain)
                masks = regions(truth, mask, domain)
                row = {"index": index, "valid_views": int(inputs["view_mask"].sum()), "methods": {}}
                for method in METHODS:
                    depth, valid, native_ranges = forward_with_ranges(models, method, domain, inputs)
                    clipped = clip_valid(depth, valid, low, high)
                    detail = {"native_cost_volume_ranges": native_ranges, "depth": {}, "range": {}}
                    for region in REGIONS:
                        before = depth_sums(depth, valid, truth, masks[region])
                        after = depth_sums(clipped, valid, truth, masks[region])
                        if before["predicted_pixels"] != after["predicted_pixels"]:
                            raise AssertionError("裁剪改变了有效预测分母")
                        counters = range_sums(depth, valid, truth, masks[region], low, high)
                        detail["depth"][region] = {"raw": depth_metrics(before), "clipped": depth_metrics(after)}
                        detail["range"][region] = finish_range(counters)
                        for key, values in (("raw", before), ("clipped", after), ("range", counters)):
                            add_statistics(totals[method][key].setdefault(region, {}), values)
                    row["methods"][method] = detail
                frames.append(row)
                del inputs, batch
            report["datasets"][name] = {
                "units": domain["geometry_units"], "train_range": [low, high],
                "gt_eval_range": domain.get("eval_range"), "depth_scale": domain["depth_scale"],
                "indices": choice["indices"], "manifest_sha256": choice["manifest_sha256"],
                "methods": {method: {
                    "raw": {r: depth_metrics(v) for r, v in counts["raw"].items()},
                    "clipped": {r: depth_metrics(v) for r, v in counts["clipped"].items()},
                    "range": {r: finish_range(v) for r, v in counts["range"].items()},
                } for method, counts in totals.items()}, "frames": frames,
            }
            write_json(output / "results.json", report)
            print(json.dumps({"completed_domain": name, "frames": len(frames)}, ensure_ascii=False), flush=True)
        models.loaded.clear()
        del models
        gc.collect()
        torch.cuda.empty_cache()
    report["completed"] = True
    report["source_sha256"] = {str(Path(__file__).relative_to(ROOT)): digest(__file__)}
    write_json(output / "results.json", report)


def self_test():
    depth = torch.tensor([1., 9., float("nan"), 0., 3., 7.]).reshape(1, 1, 1, -1)
    truth = torch.tensor([3., 6., 3., 5., 1., 9.], dtype=torch.float64).reshape_as(depth)
    mask = torch.ones_like(depth, dtype=torch.bool)
    clipped = clip_valid(depth, mask, 2., 8.)
    assert torch.isnan(clipped[..., 2]).all() and clipped[..., 3].item() == 0
    stats = range_sums(depth, mask, truth, mask, 2., 8.)
    assert stats["predicted_pixels"] == 4 and stats["gt_pixels"] == 6
    assert stats["pred_outside_train_pixels"] == 2 and stats["gt_outside_train_pixels"] == 2
    assert stats["clip_abs_reduction_sum"] == 2. and stats["clip_improved_pixels"] == 2
    other = range_sums(torch.tensor([9.]), torch.tensor([True]), torch.tensor([10.]),
                       torch.tensor([True]), 2., 8.)
    assert other["clip_abs_reduction_sum"] == -1 and other["clip_worsened_pixels"] == 1
    before = depth_sums(depth, mask, truth, mask)
    after = depth_sums(clipped, mask, truth, mask)
    assert before["predicted_pixels"] == after["predicted_pixels"] == 4
    assert before["gt_pixels"] == after["gt_pixels"] == 6
    print("CPU检查通过：范围统计、无效值保留、GT越界可恶化、共同分母不变")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--prepare", action="store_true", help="只核验固定配置/索引，不加载模型或运行GPU")
    parser.add_argument("--self-test", action="store_true", help="仅运行CPU统计正确性检查")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.config:
            parser.error("需要--config")
        settings = json.loads(Path(args.config).read_text())
        if args.prepare:
            configs, choices = prepare(settings)
            print(json.dumps({"prepared": True, "phases": list(configs),
                              "groups": len(choices), "frames": sum(len(v["indices"]) for v in choices.values()),
                              "gpu_run": False}, ensure_ascii=False))
        else:
            run(settings)
