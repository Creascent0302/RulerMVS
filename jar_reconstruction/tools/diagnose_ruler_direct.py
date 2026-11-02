"""仅DEV的直接解码器诊断：同格点覆盖/选面/连续偏移与固定权重组件敏感性。"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.carla_comparison import aggregate_depth, evaluate_depth, read_json
from jar.geometry import pixel_grid, sample_maps
from jar.models.metric_ruler_network import depth_cell_edges
from jar.models.ruler_direct import RulerDirectDecoder
from tools.cache_metric_ruler import load_record
from tools.cache_original_initializer import save_json


@contextmanager
def ablation(model, name):
    """只改变本次调用；不修改模块源码或checkpoint参数。"""
    geometry, scene, fine_offset = model.use_geometry, model.use_scene, model.use_fine_offset
    handles = []

    def geometry_off(module, args, kwargs):
        args, kwargs = list(args), dict(kwargs)
        position = 4 if module is model.scene else 5
        if len(args) > position:
            args[position] = False
        else:
            kwargs["use_geometry"] = False
        return tuple(args), kwargs

    def evidence_off(module, args):
        evidence = args[0]
        return (torch.cat((evidence[:, :-1], torch.zeros_like(evidence[:, -1:])), 1),)

    if name == "source_geometry_off":
        handles.append(model.source.register_forward_pre_hook(geometry_off, with_kwargs=True))
    elif name == "metric_channel_off":
        handles.append(model.evidence.register_forward_pre_hook(evidence_off))
    elif name == "scene_geometry_off":
        handles.append(model.scene.register_forward_pre_hook(geometry_off, with_kwargs=True))
    elif name == "all_geometry_off":
        model.use_geometry = False
    elif name == "scene_off":
        model.use_scene = False
    elif name == "fine_offset_off":
        model.use_fine_offset = False
    elif name != "full":
        raise ValueError("未知消融: " + name)
    try:
        yield
    finally:
        model.use_geometry, model.use_scene, model.use_fine_offset = geometry, scene, fine_offset
        for handle in handles:
            handle.remove()


def selected(stage, name):
    return stage[name].gather(1, stage["selected_index"])


def oracle(candidate, truth):
    index = (candidate - truth).abs().argmin(1, keepdim=True)
    return candidate.gather(1, index)


def same_grid_streams(result, stride):
    coarse, fine = result["coarse"], result["fine"]
    if stride == 8:
        lift = lambda value: value
        fine_grid = lambda value: value[..., ::2, ::2]
    elif stride == 4:
        lift = lambda value: value.repeat_interleave(2, -2).repeat_interleave(2, -1)
        fine_grid = lambda value: value
    else:
        raise ValueError("仅支持stride8/4")
    return {
        "coarse96_sampling": lift(coarse["sampling_depth"]),
        "coarse96_continuous": lift(coarse["candidate_depth"]),
        "coarse_selected_sampling": lift(selected(coarse, "sampling_depth")),
        "coarse_selected_continuous": lift(coarse["depth"]),
        # fine的每组三点中间项就是实际用于展开的粗层4个峰；无GT选择。
        "coarse_top4_continuous": fine_grid(fine["sampling_depth"][:, 1::3]),
        "fine12_sampling": fine_grid(fine["sampling_depth"]),
        "fine12_continuous": fine_grid(fine["candidate_depth"]),
        "fine_selected_sampling": fine_grid(selected(fine, "sampling_depth")),
        "fine_selected_continuous": fine_grid(fine["depth"]),
        "full_sampled": result["depth"][..., ::stride, ::stride],
    }


def score_streams(candidates, truth, valid, settings):
    result = {}
    for name, depth in candidates.items():
        if depth.shape[1] > 1:
            depth, name = oracle(depth, truth), name + "_oracle"
        if not bool(torch.isfinite(depth[valid]).all() and (depth[valid] > 0).all()):
            raise FloatingPointError("GT有效区域缺少有限正深度，不允许静默删除: " + name)
        result[name] = evaluate_depth(depth[0, 0], torch.ones_like(valid[0, 0]), truth[0, 0], valid[0, 0], settings)
    return result


def handoff_streams(result, truth, fallback):
    """整数中心4*j的固定插值，以及凸上采样真实可读邻域的GT oracle。

    GT只用于最后一个诊断oracle，不参与nearest/bilinear预测。
    """
    fine = result["fine"]
    depth, known = fine["depth"], fine["ref_valid"]
    batch, _, height, width = result["depth"].shape
    grid = pixel_grid(height, width, device=depth.device).reshape(1, -1, 2).expand(batch, -1, -1)
    streams = {"learned_convex": result["depth"]}
    for mode in ("nearest", "bilinear"):
        numerator = sample_maps(depth * known, grid, stride=4, mode=mode, padding_mode="border")
        denominator = sample_maps(known.float(), grid, stride=4, mode=mode, padding_mode="border")
        interpolated = torch.where(denominator > 0, numerator / denominator.clamp_min(1e-8), fallback)
        streams["fine_" + mode + "_aligned"] = interpolated.transpose(1, 2).reshape(batch, 1, height, width)
    neighbors = F.unfold(depth, 3, padding=1).reshape(batch, 9, *depth.shape[-2:])
    neighbor_valid = F.unfold(known.float(), 3, padding=1).reshape_as(neighbors) > 0.5
    neighbors = neighbors.repeat_interleave(4, -2).repeat_interleave(4, -1)
    neighbor_valid = neighbor_valid.repeat_interleave(4, -2).repeat_interleave(4, -1)
    error = (neighbors - truth).abs().masked_fill(~neighbor_valid, torch.inf)
    closest = neighbors.gather(1, error.argmin(1, keepdim=True))
    streams["fine_neighbor3x3_oracle"] = torch.where(neighbor_valid.any(1, keepdim=True), closest, fallback)
    return streams


def scalar_statistics(result, truth, valid, model):
    """返回每个有效GT像素的诊断量，累计时像素加权而非帧平均。"""
    values = {}
    for name, stride in (("coarse", 8), ("fine", 4)):
        stage = result[name]
        target, mask = truth[..., ::stride, ::stride], valid[..., ::stride, ::stride]
        raw = selected(stage, "sampling_depth")
        continuous = stage["depth"]
        before, after = (raw - target).abs(), (continuous - target).abs()
        best = (stage["candidate_depth"] - target).abs().amin(1, keepdim=True)
        sampling_best = (stage["sampling_depth"] - target).abs().amin(1, keepdim=True)
        shift = (continuous - raw).abs()
        if name == "coarse":
            edges = depth_cell_edges(model.depth_lattice)
            widths = (edges[1:] - edges[:-1])[None, :, None, None].expand_as(stage["sampling_depth"])
        else:
            grouped = stage["sampling_depth"].unflatten(1, (model.peak_count, 3))
            step = torch.maximum(grouped[:, :, 2] - grouped[:, :, 1], grouped[:, :, 1] - grouped[:, :, 0])
            widths = step[:, :, None].expand(-1, -1, 3, -1, -1).flatten(1, 2)
        width = widths.gather(1, stage["selected_index"]).clamp_min(1e-6)
        ordered = stage["sampling_depth"].sort(1).values
        duplicate = ((ordered[:, 1:] - ordered[:, :-1]).abs() < 1e-4).sum(1, keepdim=True).float()
        distance = (stage["candidate_depth"] - target).abs()
        near_optimal = ((distance - best).abs() < 0.01).sum(1, keepdim=True).float()
        fields = {
            "selected_offset_absolute_m": shift,
            "selected_offset_fraction_of_cell_width": shift / width,
            "offset_improvement_m": before - after,
            "offset_improves_1cm_fraction": (before - after > 0.01).float(),
            "offset_worsens_1cm_fraction": (after - before > 0.01).float(),
            "selected_minus_oracle_m": after - best,
            "sampling_oracle_error_m": sampling_best,
            "continuous_oracle_error_m": best,
            "duplicate_candidate_fraction": duplicate / max(1, ordered.shape[1] - 1),
            "near_optimal_candidate_count_1cm": near_optimal,
            "best_probability": stage["probability"].amax(1, keepdim=True),
            "predicted_support": selected(stage, "support").float(),
        }
        for threshold in (0.25, 0.5, 1., 2.):
            fields[f"oracle_outside_{threshold:g}m_fraction"] = (best > threshold).float()
            fields[f"error_on_oracle_outside_{threshold:g}m"] = after * (best > threshold)
        fields["selected_error_m"] = after
        for key, value in fields.items():
            values[name + "/" + key] = value[mask].double()
    return values


@torch.inference_mode()
def diagnose(model, records, settings, variant, detailed, upsampling=False):
    streams = {"same_grid_s8": [], "same_grid_s4": [], "full": [], "handoff_full": []}
    sums = {}
    with ablation(model, variant):
        for row in records:
            output = model(row["inputs"])
            truth, valid = row["targets"]["depth"], row["targets"]["valid"]
            streams["full"].append({"sample_id": row["sample_id"], "streams": score_streams(
                {"depth": output["depth"]}, truth, valid, settings)})
            if upsampling:
                choices = handoff_streams(output, truth, (model.near * model.far)**0.5)
                streams["handoff_full"].append({"sample_id": row["sample_id"], "streams": score_streams(
                    choices, truth, valid, settings)})
            if detailed:
                for stride in (8, 4):
                    candidates = same_grid_streams(output, stride)
                    scored = score_streams(candidates, truth[..., ::stride, ::stride], valid[..., ::stride, ::stride], settings)
                    streams[f"same_grid_s{stride}"].append({"sample_id": row["sample_id"], "streams": scored})
                for key, value in scalar_statistics(output, truth, valid, model).items():
                    total, count = sums.get(key, (0., 0))
                    sums[key] = (total + float(value.sum()), count + value.numel())
    statistics = {key: total / count if count else None for key, (total, count) in sums.items()}
    for name in ("coarse", "fine"):
        denominator = statistics.get(name + "/selected_error_m")
        if denominator:
            for threshold in (0.25, 0.5, 1., 2.):
                statistics[f"{name}/error_share_oracle_outside_{threshold:g}m"] = statistics[
                    f"{name}/error_on_oracle_outside_{threshold:g}m"] / denominator
    return {"metrics": {key: aggregate_depth(rows, settings) for key, rows in streams.items() if rows},
            "pixel_weighted_statistics": statistics}


def run(plan, device):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
    folders = {name: Path(path) for name, path in plan["experiments"].items()}
    for folder in folders.values():
        if not (folder / "summary.json").exists():
            raise FileNotFoundError(f"训练未完成，不能读取运行中best: {folder}")
    index = read_json(Path(plan["cache_root"]) / "index.json")
    settings = index["contract"]["protocol"]["evaluation"]
    records = []
    allowed = ("matching_features", "matching_scale", "raw_maps", "K", "T_cam_from_rig",
               "view_mask", "image_mask", "image_size", "reference_index")
    for path in index["sets"]["dev_complete"]:
        if not path.startswith("dev_complete/"):
            raise ValueError("诊断只允许DEV")
        row = load_record(plan["cache_root"], path, device)
        records.append({"sample_id": row["sample_id"], "inputs": {key: row["inputs"][key] for key in allowed},
                        "targets": {key: row["targets"][key] for key in ("depth", "valid")}})
    report = {"plan": plan, "no_test_read": True, "sample_ids": [row["sample_id"] for row in records],
              "notes": ["oracle读取GT事后选候选，仅诊断，不能部署或作为实际模型成绩。",
                        "same_grid_s8全部使用8*j同一GT，same_grid_s4把粗候选最近邻传播后再与4*j同一GT比较。",
                        "关模块属于固定权重敏感性，不等同重新训练后的结构净收益。",
                        "固定nearest/bilinear以整数中心4*j采样，非默认half-pixel interpolate；邻域oracle读取GT，只诊断。",
                        "unknown未用于删GT；所有实际评测像素要求有限正预测。"], "experiments": {}}
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    for name, folder in folders.items():
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        model = RulerDirectDecoder(checkpoint["plan"]["model"]).to(device).eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        variants = plan.get("ablations", {}).get(name, ["full"])
        results = {"step": checkpoint["step"], "checkpoint_sha256": hashlib.sha256((folder / "best.pt").read_bytes()).hexdigest(),
                   "variants": {}}
        for variant in variants:
            result = diagnose(model, records, settings, variant, detailed=variant == "full",
                              upsampling=plan.get("upsampling_diagnostics", False))
            results["variants"][variant] = result
            print(json.dumps({"experiment": name, "variant": variant,
                              "full": result["metrics"]["full"]["depth"]["all"]}), flush=True)
            report["experiments"][name] = results
            save_json(output / "partial.json", report)
        del model
    save_json(output / "summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    run(read_json(arguments.config), torch.device(arguments.device))
