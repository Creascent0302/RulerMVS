"""少量DEV帧的TVD逆深度表面运输：四面提议、精确匹配和独立GT oracle。"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.carla_comparison import aggregate_depth, comparison_dataset, evaluate_depth, gt_reference, read_json
from jar.data import collate_frames
from jar.geometry import image_texture, pixel_grid, project, sample_maps, unproject
from jar.pretrained_matching import sample_pretrained_matching
from jar.runtime import move, seed_all
from tools.carla_compare import file_sha256, make_model, synchronize


def minmod(left, right):
    return torch.where(left * right > 0, left.sign() * torch.minimum(left.abs(), right.abs()), 0)


def inverse_planes(depth):
    """TVD式限制梯度：仿射逆深度保持精确，跳变处不平均两侧斜率。"""
    rho = depth.clamp_min(1e-6).reciprocal()
    left = (rho - torch.cat((rho[:, :1], rho[:, :-1]), 1)) / 4
    right = (torch.cat((rho[:, 1:], rho[:, -1:]), 1) - rho) / 4
    up = (rho - torch.cat((rho[:1], rho[:-1]), 0)) / 4
    down = (torch.cat((rho[1:], rho[-1:]), 0) - rho) / 4
    gx, gy = minmod(left, right), minmod(up, down)
    gx[:, 0], gx[:, -1] = right[:, 0], left[:, -1]
    gy[0], gy[-1] = down[0], up[-1]
    return rho, gx, gy


def propose_surfaces(depth4, uv, near, far):
    rho, gx, gy = inverse_planes(depth4)
    x0 = (uv[:, 0] / 4).floor().long().clamp_max(rho.shape[1] - 1)
    y0 = (uv[:, 1] / 4).floor().long().clamp_max(rho.shape[0] - 1)
    x1, y1 = (x0 + 1).clamp_max(rho.shape[1] - 1), (y0 + 1).clamp_max(rho.shape[0] - 1)
    x = torch.stack((x0, x1, x0, x1), -1)
    y = torch.stack((y0, y0, y1, y1), -1)
    offset = uv[:, None] - 4 * torch.stack((x, y), -1)
    transported = rho[y, x] + gx[y, x] * offset[..., 0] + gy[y, x] * offset[..., 1]
    return transported.clamp(1 / far, 1 / near).reciprocal()


def choose_budget(inputs, full_depth, planes, reference, fraction):
    uv = pixel_grid(*full_depth.shape, device=full_depth.device).reshape(-1, 2)
    z = full_depth.flatten()
    intrinsic, transform = inputs["K"][0], inputs["T_cam_from_rig"][0]
    xyz, ray = unproject(uv, z, intrinsic[reference], transform[reference])
    _, _, valid, jacobian = project(
        xyz, intrinsic, transform, full_depth.shape, inputs["view_mask"][0], inputs["image_mask"][0]
    )
    valid[reference] = False
    derivative = (jacobian * ray[None, :, None]).sum(-1).norm(dim=-1)
    rho_derivative = derivative * z[None].square()
    observable = torch.where(valid, rho_derivative, 0).amax(0)
    rho = planes.reciprocal()
    span = rho.amax(-1) - rho.amin(-1)
    priority = span * observable
    allowed = inputs["image_mask"][0, reference].flatten() & inputs["view_mask"][0, reference]
    priority = torch.where(allowed & torch.isfinite(priority), priority, 0)
    count = min(math.floor(fraction * int(allowed.sum())), int((priority > 0).sum()))
    return uv, priority.topk(count, sorted=False).indices


def score_surfaces(inputs, features, uv, depths, reference, image_scale):
    """沿用等权Hero余弦，不新增学习评分器或改变有效源定义。"""
    views, queries, candidates = len(features), len(uv), depths.shape[1]
    intrinsic, transform = inputs["K"][0], inputs["T_cam_from_rig"][0]
    points, ray = unproject(
        uv[:, None].expand(-1, candidates, -1), depths, intrinsic[reference], transform[reference]
    )
    projected, _, valid, jacobian = project(
        points.reshape(-1, 3), intrinsic, transform, inputs["rgb"].shape[-2:],
        inputs["view_mask"][0], inputs["image_mask"][0],
    )
    reference_valid = valid[reference].clone()
    derivative = (jacobian * ray.reshape(1, -1, 1, 3)).sum(-1).norm(dim=-1)
    texture = inputs.get("_texture")
    texture = image_texture(inputs["rgb"][0]) if texture is None else texture[0]
    reference_texture = sample_maps(texture[reference:reference + 1], uv[None])[0, :, 0]
    source_texture = sample_maps(texture, projected)[..., 0]
    valid &= (derivative > 1e-5) & (source_texture > .002)
    valid &= (reference_texture[:, None].expand(-1, candidates).reshape(1, -1) > .002)
    valid &= reference_valid[None]
    valid[reference] = False
    valid = valid.reshape(views, queries, candidates)
    reference_feature = sample_pretrained_matching(features[reference:reference + 1], uv[None], image_scale)[0]
    source = sample_pretrained_matching(features, projected, image_scale).reshape(views, queries, candidates, -1)
    cosine = (F.normalize(source, dim=-1, eps=1e-6) * F.normalize(reference_feature, dim=-1, eps=1e-6)[None, :, None]).sum(-1)
    mean = torch.where(valid, cosine, 0).sum(0) / valid.sum(0).clamp_min(1)
    mean = mean.masked_fill(~valid.any(0), -1e4)
    return mean, cosine, valid


def run(experiment, device):
    output = Path(experiment["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "frames.jsonl").exists():
        raise FileExistsError("已有表面运输诊断，请使用新目录")
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    seed_all(experiment.get("seed", 29))
    model, config = make_model(experiment, experiment["weights"], device)
    protocol = read_json(experiment["protocol"])
    settings = protocol["evaluation"]
    reference = settings["reference_index"]
    dataset = comparison_dataset(protocol, "dev_complete")
    work = {}

    def capture(module, arguments):
        work.clear()
        work.update(arguments[0])

    handle = model.coarse.register_forward_pre_hook(capture)
    records, oracle_records, selected_records = [], [], []
    timing = []
    try:
        with torch.inference_mode(), (output / "frames.jsonl").open("w") as stream:
            for index in experiment["frame_indices"]:
                batch = move(collate_frames([dataset[index]]), device)
                synchronize(device)
                start = time.perf_counter()
                prediction = model(batch["inputs"])
                synchronize(device)
                network_ms = (time.perf_counter() - start) * 1000
                original = prediction["dense_depth"][0, reference]
                depth4 = next(stage["depth"][0, reference] for stage in prediction["dense"]["stages"] if stage["stride"] == 4)
                start = time.perf_counter()
                uv = pixel_grid(*original.shape, device=device).reshape(-1, 2)
                planes = propose_surfaces(depth4, uv, model.dense_decoder.near, model.dense_decoder.far)
                uv, indices = choose_budget(batch["inputs"], original, planes, reference, experiment["budget_fraction"])
                query_uv = uv[indices]
                hypotheses = torch.cat((original.flatten()[indices, None], planes[indices]), -1)
                means, source_cosines, source_masks = [], [], []
                image_scale = work.get("_pretrained_matching_scale")
                image_scale = None if image_scale is None else image_scale[0]
                for start_query in range(0, len(indices), 1024):
                    stop = start_query + 1024
                    mean, cosine, mask = score_surfaces(
                        work, work["_pretrained_matching_features"][0], query_uv[start_query:stop],
                        hypotheses[start_query:stop], reference, image_scale,
                    )
                    means.append(mean)
                    source_cosines.append(cosine)
                    source_masks.append(mask)
                mean = torch.cat(means)
                cosine, source_valid = torch.cat(source_cosines, 1), torch.cat(source_masks, 1)
                winner = mean.argmax(-1)
                selected = hypotheses.gather(1, winner[:, None])[:, 0]
                transported = original.clone().flatten()
                transported[indices] = selected
                transported = transported.reshape_as(original)
                two_positive = None
                paired_variant = None
                if experiment.get("two_positive_diagnostic", True):
                    common = source_valid & source_valid[:, :, :1]
                    positive = common & (cosine > cosine[:, :, :1])
                    two_positive = positive.sum(0).gather(1, winner[:, None])[:, 0] >= 2
                    paired_variant = original.clone().flatten()
                    paired_variant[indices] = torch.where(two_positive, selected, original.flatten()[indices])
                    paired_variant = paired_variant.reshape_as(original)
                synchronize(device)
                post_ms = (time.perf_counter() - start) * 1000

                # 所有选择已完成；以下GT仅评价与独立oracle，不参与预算或选面。
                truth, valid = gt_reference(batch["inputs"], batch["targets"], settings)
                target = truth.flatten()[indices]
                known = valid.flatten()[indices]
                baseline_error = (hypotheses[:, 0] - target).abs()
                candidate_error = (hypotheses - target[:, None]).abs()
                oracle_indices = candidate_error.argmin(-1)
                visible_error = candidate_error.masked_fill(~source_valid.any(0), torch.inf)
                visible_error[:, 0] = candidate_error[:, 0]
                visible_indices = visible_error.argmin(-1)
                oracles = {}
                for name, choice in (("GT_ORACLE_four_transported_surfaces", oracle_indices), ("GT_ORACLE_observable_surfaces", visible_indices)):
                    oracle = original.clone().flatten()
                    oracle[indices] = hypotheses.gather(1, choice[:, None])[:, 0]
                    oracles[name] = oracle.reshape_as(original)
                predictions = {"original_full": original, "mean_Hero_selected_surface": transported}
                if paired_variant is not None:
                    predictions["two_source_positive_diagnostic_only"] = paired_variant
                accepted = batch["inputs"]["image_mask"][0, reference]
                metrics = {name: evaluate_depth(depth, accepted, truth, valid, settings) for name, depth in predictions.items()}
                oracle_metrics = {name: evaluate_depth(depth, accepted, truth, valid, settings) for name, depth in oracles.items()}
                chosen_error = (selected - target).abs()
                oracle_error = candidate_error.gather(1, oracle_indices[:, None])[:, 0]
                gain = float((baseline_error[known] - chosen_error[known]).double().sum())
                possible = float((baseline_error[known] - oracle_error[known]).double().sum())
                details = {
                    "budget_pixels": len(indices), "budget_fraction_input": len(indices) / original.numel(),
                    "budget_gt_pixels": int(known.sum()), "changed_pixels": int((winner > 0).sum()),
                    "oracle_absolute_error_reduction_sum_m": possible,
                    "actual_absolute_error_reduction_sum_m": gain,
                    "oracle_gain_realized_fraction": gain / possible if possible else None,
                    "selected_mean_support": float(source_valid.sum(0).gather(1, winner[:, None]).float().mean()),
                }
                if two_positive is not None:
                    details["selected_two_positive_pixels"] = int(((winner > 0) & two_positive).sum())
                row = {"index": index, "predictions": metrics, "GT_ORACLE_diagnostic_only": oracle_metrics, "selection": details,
                       "timing": {"original_network_four_references_ms": network_ms, "postprocess_reference2_only_ms": post_ms}}
                records.append({"streams": metrics})
                oracle_records.append({"streams": oracle_metrics})
                selected_records.append(details)
                timing.append(row["timing"])
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                print(f"DEV[{index}] original={metrics['original_full']['all']['MAE_m']:.5f} "
                      f"transport={metrics['mean_Hero_selected_surface']['all']['MAE_m']:.5f} "
                      f"oracle={oracle_metrics['GT_ORACLE_four_transported_surfaces']['all']['MAE_m']:.5f}", flush=True)
                if index in experiment.get("save_frames", experiment["frame_indices"]):
                    torch.save({"index": index, "rgb": batch["inputs"]["rgb"][0, reference].cpu(), "truth": truth.cpu(),
                                "truth_valid": valid.cpu(), "selected_indices": indices.cpu(), "hypotheses": hypotheses.cpu(),
                                "mean_Hero": mean.cpu(), "source_cosines": cosine.cpu(), "source_valid": source_valid.cpu(),
                                "winner": winner.cpu(), "predictions": {k: v.cpu() for k, v in predictions.items()},
                                "GT_ORACLE_diagnostic_only": {k: v.cpu() for k, v in oracles.items()}}, output / f"frame_{index:02d}.pt")
    finally:
        handle.remove()
    possible = sum(row["oracle_absolute_error_reduction_sum_m"] for row in selected_records)
    gain = sum(row["actual_absolute_error_reduction_sum_m"] for row in selected_records)
    report = {
        "command": sys.argv, "experiment": experiment, "frames": len(records),
        "checkpoint_sha256": file_sha256(experiment["weights"]), "script_sha256": file_sha256(__file__),
        "predictions": aggregate_depth(records, settings), "GT_ORACLE_diagnostic_only": aggregate_depth(oracle_records, settings),
        "selection": {"oracle_absolute_error_reduction_sum_m": possible, "actual_absolute_error_reduction_sum_m": gain,
                      "oracle_gain_realized_fraction": gain / possible if possible else None},
        "timing": {key: sum(row[key] for row in timing) / len(timing) for key in timing[0]},
        "notes": [
            "实际DEV帧索引见experiment.frame_indices；不训练，冻结权重，普通forward后执行独立后处理。",
            "原图4*j锚点的局部逆深度平面采用minmod限斜率；每像素最多四个运输表面及原预测null，不跨表面求深度均值。",
            "预算只依预测四面差异及标定导数排序，最多输入像素10%；候选点始终精确投影。",
            "主选择器沿用有效源Hero等权余弦；两源正改进仅单列诊断，不预设其有效。",
            "GT oracle只在选择完成后读取GT，表示固定候选及预算的潜在收益，不是模型成绩或全局上限。",
            "后处理只计算参考2，原网络仍计算四参考；其耗时不能冒充完整场景流水线速度。",
            "未在stride4锚点出现的细物体不会凭此方案自动恢复；高余弦仍可能是遮挡或重复纹理误匹配。",
        ],
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(read_json(args.config), torch.device(args.device))


if __name__ == "__main__":
    main()
