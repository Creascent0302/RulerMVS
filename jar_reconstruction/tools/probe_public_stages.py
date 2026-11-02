"""仅在 TRAIN/VAL 定位候选选择和连续上采样的误差；GT 只用于事后诊断。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.geometry import pixel_grid, safe_masked_softmax, sample_maps
from jar.models.ruler_direct import RulerDirectDecoder
from jar.pretraining.comparison import write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.pretraining.model import PretrainingModel
from jar.runtime import move


def error_statistics(estimate, truth, valid):
    error = (estimate - truth).abs()[valid].double()
    relative = error / truth[valid].double()
    if not bool(torch.isfinite(error).all()):
        raise ValueError("诊断预测非有限，不能静默筛除")
    return {"count": len(error), "absolute_sum": float(error.sum()),
            "square_sum": float(error.square().sum()), "relative_sum": float(relative.sum())}


def summarize(value):
    if "absolute_sum" in value:
        count = max(value["count"], 1)
        return {**value, "mae": value["absolute_sum"] / count,
                "rmse": (value["square_sum"] / count) ** .5,
                "abs_rel": value["relative_sum"] / count}
    return {key: summarize(item) if isinstance(item, dict) else item for key, item in value.items()}


def sum_records(records):
    first = records[0]
    if isinstance(first, dict):
        return {key: sum_records([row[key] for row in records]) for key in first}
    return sum(records)


def stage_statistics(stage, truth, valid, scale):
    """比较硬标签、可达区间和连续候选，全部误差恢复为原数据单位。"""
    samples = stage["sampling_depth"] * scale
    candidates = stage["candidate_depth"] * scale
    available = stage["valid"]
    sample_error = (samples - truth).abs().masked_fill(~available, torch.inf)
    continuous_error = (candidates - truth).abs().masked_fill(~available, torch.inf)
    # 真实训练 CE 使用对数最近 sampling，而非绝对最近候选。
    label = (samples.clamp_min(1e-6).log() - truth.clamp_min(1e-6).log()).abs()
    label = label.masked_fill(~available, torch.inf).argmin(1, keepdim=True)
    oracle_index = continuous_error.argmin(1, keepdim=True)
    predicted_index = stage["selected_index"]
    lower = stage["lower"] * scale
    upper = stage["upper"] * scale
    inside = (truth >= lower) & (truth <= upper) & available
    interval_error = torch.maximum(lower - truth, truth - upper).clamp_min(0)
    interval_error = interval_error.masked_fill(~available, torch.inf).amin(1, keepdim=True)
    valid = valid & stage["ref_valid"]
    selected = candidates.gather(1, predicted_index)
    oracle = candidates.gather(1, oracle_index)
    labels = candidates.gather(1, label)
    output = {
        "prediction": error_statistics(stage["depth"] * scale, truth, valid),
        "winner": error_statistics(selected, truth, valid),
        "sampling_oracle": error_statistics(truth + sample_error.amin(1, keepdim=True), truth, valid),
        "continuous_oracle": error_statistics(oracle, truth, valid),
        "interval_oracle": error_statistics(truth + interval_error, truth, valid),
        "training_label_candidate": error_statistics(labels, truth, valid),
        "counts": {"valid": int(valid.sum()),
                   "any_interval_contains_gt": int((inside.any(1, keepdim=True) & valid).sum()),
                   "label_interval_contains_gt": int((inside.gather(1, label) & valid).sum()),
                   "label_vs_continuous_oracle_disagree": int(((label != oracle_index) & valid).sum()),
                   "winner_vs_continuous_oracle_disagree": int(((predicted_index != oracle_index) & valid).sum())},
    }
    for name, index in (("label", label), ("winner", predicted_index), ("oracle", oracle_index)):
        support = stage["support"].gather(1, index)
        output["counts"][name + "_no_source"] = int(((support == 0) & valid).sum())
    return output


def convex_diagnostics(fine, logits, scale):
    """重用同一上采样权重，比较深度/逆深度读出；不重新预测或读取GT。"""
    depth = fine["depth"] * scale
    batch, _, height, width = depth.shape
    legal = F.unfold(fine["ref_valid"].float(), 3, padding=1).reshape(batch, 1, 9, 1, 1, height, width) > .5
    weights = safe_masked_softmax(logits.reshape(batch, 1, 9, 4, 4, height, width).float(), legal, dim=2)
    neighbors = F.unfold(depth, 3, padding=1).reshape(batch, 1, 9, 1, 1, height, width)

    def pack(value):
        return value.permute(0, 1, 4, 2, 5, 3).reshape(batch, 1, height * 4, width * 4)

    inverse = pack((weights * neighbors.clamp_min(1e-8).reciprocal()).sum(2)).clamp_min(1e-8).reciprocal()
    low = neighbors.masked_fill(~legal, torch.inf).amin(2).expand(-1, -1, 4, 4, -1, -1)
    high = neighbors.masked_fill(~legal, -torch.inf).amax(2).expand(-1, -1, 4, 4, -1, -1)
    return inverse, pack(low), pack(high)


def depth_groups(truth, valid):
    """按每张图有效GT深度四分位分桶；仅用来解释误差分布。"""
    quantiles = torch.quantile(truth[valid], truth.new_tensor((.25, .5, .75)))
    rank = torch.bucketize(truth.contiguous(), quantiles)
    return {"quartile_" + str(index + 1): valid & (rank == index) for index in range(4)}


def edge_groups(truth, valid):
    safe = torch.where(valid, truth, 1).log()
    edge = torch.zeros_like(valid)
    for axis in (-1, -2):
        left, right = [slice(None)] * 4, [slice(None)] * 4
        left[axis], right[axis] = slice(1, None), slice(None, -1)
        left, right = tuple(left), tuple(right)
        jump = ((safe[left] - safe[right]).abs() > .1) & valid[left] & valid[right]
        edge[left] |= jump
        edge[right] |= jump
    edge = F.max_pool2d(edge.float(), 5, stride=1, padding=2) > .5
    return {"edge_5px": valid & edge, "smooth": valid & ~edge}


def fine_error_decomposition(fine, truth, valid, scale):
    """将细格绝对误差分解为供给下界、选峰损失、峰内定位损失。

    同一粗峰的三个连续候选构成一个深度区间；不同峰之间不连成区间。
    这是GT事后分析，不产生部署预测，不将跨表面平均解释为正确表面。
    """
    candidates = fine["candidate_depth"] * scale
    batch, count, height, width = candidates.shape
    if count % 3:
        raise ValueError("该分解要求每个粗峰对应三个细候选")
    groups = candidates.reshape(batch, count // 3, 3, height, width)
    available = fine["valid"].reshape_as(groups)
    low = groups.masked_fill(~available, torch.inf).amin(2)
    high = groups.masked_fill(~available, -torch.inf).amax(2)
    group_error = torch.maximum(low - truth, truth - high).clamp_min(0)
    group_error = group_error.masked_fill(~available.any(2), torch.inf)
    supply = group_error.amin(1, keepdim=True)
    selected = group_error.gather(1, fine["selected_index"] // 3)
    total = (fine["depth"] * scale - truth).abs()
    valid = valid & fine["ref_valid"]
    selection = selected - supply
    location = total - selected
    tolerance = 1e-5 * (1 + truth.abs() + (fine["depth"] * scale).abs())
    for value in (supply, selection, location):
        if not bool(torch.isfinite(value[valid]).all()):
            raise AssertionError("误差分解包含非有限值")
        if bool((value[valid] < -tolerance[valid]).any()):
            raise AssertionError("输出不在所选峰的连续候选凸包内，不能使用本分解")
    terms = {"supply_floor": supply, "wrong_peak": selection.clamp_min(0),
             "within_peak": location.clamp_min(0), "total": total}
    reconstruction = terms["supply_floor"] + terms["wrong_peak"] + terms["within_peak"]
    if bool(((reconstruction - total).abs()[valid] > tolerance[valid]).any()):
        raise AssertionError("误差分解加法恒等式不成立")
    result = {}
    for name, value in terms.items():
        error = value[valid].double()
        result[name] = {"count": len(error), "absolute_sum": float(error.sum()),
                        "square_sum": float(error.square().sum()),
                        "relative_sum": float((error / truth[valid].double()).sum())}
    return result


def diagnose(output, logits, batch, domain):
    inputs, targets = batch["inputs"], batch["targets"]
    truth = targets["depth"][:, :1].float()
    valid = targets["depth_valid"][:, :1] & inputs["image_mask"][:, :1]
    valid = valid & inputs["view_mask"][:, :1, None, None] & torch.isfinite(truth) & (truth > 0)
    scale = output["unit_scale"]
    full = output["depth"] * scale
    statistics = {"full": error_statistics(full, truth, valid)}
    for name, stride in (("coarse", 8), ("fine", 4)):
        stage = output[name]
        statistics[name] = stage_statistics(stage, truth[..., ::stride, ::stride], valid[..., ::stride, ::stride], scale)
    fine = output["fine"]
    statistics["fine_error_decomposition"] = fine_error_decomposition(
        fine, truth[..., ::4, ::4], valid[..., ::4, ::4], scale
    )
    statistics["fine_mode_mean"] = error_statistics(RulerDirectDecoder._mode_depth(fine) * scale,
                                                    truth[..., ::4, ::4], valid[..., ::4, ::4])
    grid = pixel_grid(*truth.shape[-2:], device=truth.device).reshape(1, -1, 2)
    for mode in ("nearest", "bilinear"):
        depth = sample_maps(fine["depth"] * scale, grid, stride=4, mode=mode, padding_mode="border")
        statistics["upsample_" + mode] = error_statistics(depth.transpose(1, 2).reshape_as(truth), truth, valid)
    inverse, low, high = convex_diagnostics(fine, logits, scale)
    statistics["upsample_inverse_same_weights"] = error_statistics(inverse, truth, valid)
    interval_error = torch.maximum(low - truth, truth - high).clamp_min(0)
    statistics["upsample_convex_oracle"] = error_statistics(truth + interval_error, truth, valid)
    groups = {**depth_groups(truth, valid), **edge_groups(truth, valid)}
    statistics["groups"] = {name: error_statistics(full, truth, mask) for name, mask in groups.items()}
    statistics["outside_range"] = {"count": int(valid.sum()),
                                  "outside": int((valid & ((truth < domain["near"]) | (truth > domain["far"]))).sum())}
    return statistics


@torch.no_grad()
def run(plan):
    if plan["split"] not in ("train", "val"):
        raise ValueError("结构诊断仅允许TRAIN/VAL，禁止TEST调参")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(plan["device"])
    torch.cuda.set_device(device)
    output = Path(plan["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("诊断输出已存在，不覆盖")
    output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(plan["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    config = saved["config"]
    model = PretrainingModel(config, initialize=False)
    model.load_state_dict(saved["model"], strict=True)
    model.requires_grad_(False).to(device).eval()
    del saved
    config = dict(config, datasets=[domain for domain in config["datasets"] if domain["name"] in plan["datasets"]])
    dataset = MixedDataset(config, plan["split"])
    records, final = [], {}
    capture = {}
    handle = model.decoder.upsample_mask.register_forward_hook(lambda module, inputs, result: capture.update(logits=result))
    write_json(output / "config.json", plan)
    try:
        for domain_index, domain in enumerate(dataset.domains):
            count = len(dataset.datasets[domain_index])
            indices = torch.linspace(0, count - 1, min(plan["samples_per_domain"], count)).long().tolist()
            rows = []
            for index in indices:
                batch = move(collate_mixed([dataset[(domain_index, index, -1)]]), device)
                prediction = model(batch["inputs"], domain, reference=0)
                statistics = diagnose(prediction, capture.pop("logits"), batch, domain)
                row = {"dataset": domain["name"], "index": index, "sample_id": batch.get("sample_id"),
                       "statistics": statistics}
                records.append(row)
                rows.append(statistics)
            final[domain["name"]] = {"frames": len(rows), "units": domain["geometry_units"],
                                     "statistics": summarize(sum_records(rows))}
            write_json(output / (domain["name"] + ".json"), {"summary": final[domain["name"]],
                                                              "rows": records[-len(rows):]})
            print(json.dumps({"dataset": domain["name"], "summary": final[domain["name"]]}, ensure_ascii=False), flush=True)
        interventions = {}
        for name, key, value in (("no_scale_routing", "scale_routing", False),
                                 ("cell_width_only", "fine_step_pixels", 0.)):
            previous = model.decoder.settings[key]
            model.decoder.settings[key] = value
            interventions[name] = {}
            for domain_index, domain in enumerate(dataset.domains):
                count = len(dataset.datasets[domain_index])
                indices = torch.linspace(0, count - 1, min(plan["samples_per_domain"], count)).long().tolist()
                rows = []
                for index in indices:
                    batch = move(collate_mixed([dataset[(domain_index, index, -1)]]), device)
                    prediction = model(batch["inputs"], domain, reference=0)
                    statistics = diagnose(prediction, capture.pop("logits"), batch, domain)
                    rows.append(statistics)
                interventions[name][domain["name"]] = summarize(sum_records(rows))
                write_json(output / (name + "_" + domain["name"] + ".json"),
                           {"statistics": interventions[name][domain["name"]], "rows": rows})
                print(json.dumps({"intervention": name, "dataset": domain["name"],
                                  "full": interventions[name][domain["name"]]["full"]}), flush=True)
            model.decoder.settings[key] = previous
    finally:
        handle.remove()
    result = {"config": plan, "fine_readout": model.decoder.fine_readout,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "notes": ["全部GT仅用于事后误差/oracle/分桶，不参与推理", "各stage在各自格点评价，不能直接把粗细MAE差当精修因果收益",
                        "逆深度上采样使用原深度网络学习的相同权重，是固定权重干预，不是重新训练后的上限",
                        "凸包oracle允许事后逐像素任选邻域混合，只是该上采样结构可表达的误差下界"],
              "domains": final, "interventions": interventions}
    write_json(output / "summary.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))
