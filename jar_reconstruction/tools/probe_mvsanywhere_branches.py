"""冻结权重的分支干预：区分单图先验、多视图证据和两者的融合。

只修改研究前向中的特征，不改官方源码、生产模型或 checkpoint。
置零/错位属于推理敏感性诊断，不能冒充重新训练后的结构消融。
"""

import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.carla_comparison import aggregate_depth, comparison_dataset, evaluate_depth, gt_reference, read_json
from jar.data import collate_frames
from jar.runtime import move, seed_all
from official_methods.methods.niantic_common import depth_intrinsics
from official_methods.postprocess import depth_on_image_grid
from tools.carla_compare import file_sha256, make_model


def alter_mono(features, mode, shape):
    """保持六组 (patch token, class token) 的类型、网格和通道。"""
    if mode == "keep":
        return features
    result = []
    for index, (tokens, cls) in enumerate(features):
        zero = mode == "zero" or (mode == "shallow_zero" and index < 2)
        zero |= mode == "deep_zero" and index >= 2
        if zero:
            result.append((torch.zeros_like(tokens), torch.zeros_like(cls)))
        elif mode == "roll":
            height, width = shape
            grid = tokens.reshape(len(tokens), height, width, -1)
            grid = torch.roll(grid, (height // 3, width // 3), (1, 2))
            result.append((grid.reshape_as(tokens), cls))
        elif mode in ("shallow_zero", "deep_zero"):
            result.append((tokens, cls))
        else:
            raise ValueError(f"未知单图特征干预：{mode}")
    return result


def alter_volume(volume, mode):
    if mode == "keep":
        return volume
    if mode == "zero":
        return torch.zeros_like(volume)
    if mode == "roll":
        height, width = volume.shape[-2:]
        return torch.roll(volume, (height // 3, width // 3), (-2, -1))
    raise ValueError(f"未知候选体干预：{mode}")


def replay(net, cur, src, volume, mono):
    """缓存同一帧的代价体后调用完整原生 forward，保留全部原生解码逻辑。"""
    current = dict(cur, cur_feats=mono)
    with patch.object(net.cost_volume, "forward", return_value=volume):
        return net(
            "test", current, dict(src), unbatched_matching_encoder_forward=True,
            return_mask=True, estimate_depth_range=False, num_refinement_steps=0,
        )


def build_volume(net, cur, src, reference_features, source_features):
    return net.cost_volume(
        cur_feats=reference_features, src_feats=source_features,
        src_extrinsics=src["cam_T_world_b44"] @ cur["world_T_cam_b44"][:, None],
        src_poses=cur["cam_T_world_b44"][:, None] @ src["world_T_cam_b44"],
        src_Ks=src["K_matching_b44"], cur_invK=cur["invK_matching_b44"],
        min_depth=cur["min_depth"], max_depth=cur["max_depth"], return_mask=True,
    )


def map_depth(depth, inputs, reference, scale=1.0):
    prediction = {
        "depth": depth, "depth_valid": torch.isfinite(depth) & (depth > 0),
        "depth_K": depth_intrinsics(inputs["K"][:, [reference]], scale),
        "reference_indices": [reference],
    }
    result = depth_on_image_grid(prediction, inputs)
    return result["depth"][0, 0]


def module_difference(module, original):
    current = module.state_dict()
    if current.keys() != original.keys():
        raise ValueError("权重比较的键不一致")
    squared, base, unequal, maximum = 0.0, 0.0, 0, 0.0
    for key, value in current.items():
        value, other = value.detach().cpu(), original[key]
        if value.shape != other.shape:
            raise ValueError(f"权重形状不同：{key}")
        unequal += int(not torch.equal(value, other))
        delta = value.float() - other.float()
        squared += float(delta.square().sum())
        base += float(other.float().square().sum())
        maximum = max(maximum, float(delta.abs().max()))
    return {"tensors": len(current), "different_tensors": unequal,
            "relative_l2": (squared / max(base, 1e-30)) ** 0.5, "max_abs": maximum}


def raw_components(net, path):
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = saved["state_dict"]
    mono = copy.deepcopy(net.encoder)
    matching = copy.deepcopy(net.matching_model)
    original = {}
    for name, module in (("encoder", mono), ("matching_model", matching)):
        prefix = name + "."
        selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        module.load_state_dict(selected, strict=True)
        original[name] = selected
    return mono.eval(), matching.eval(), original


def summarize(rows, settings):
    result = {grid: aggregate_depth([{ "streams": row[grid]} for row in rows], settings)
              for grid in ("anchors", "full_image")}
    intact = result["anchors"]["hero_full"]["all"]["MAE_m"]
    result["anchor_delta_from_full_m"] = {
        name: item["all"]["MAE_m"] - intact
        for name, item in result["anchors"].items()
    }
    return result


def own_depths(own, inputs, reference):
    cache = {}

    def capture(module, args, kwargs):
        cache["args"], cache["kwargs"] = args, kwargs

    handle = own.dense_decoder.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        prediction = own(inputs)
    finally:
        handle.remove()

    def collect(dense, prefix):
        return {
            prefix + "_learned": dense["surface_recovery"]["candidate_depth"][0, reference, 0],
            prefix + "_k9": dense["refinement"]["depth_before"][0, reference],
        }

    anchors = collect(prediction["dense"], "own")
    full = {"own_final": prediction["dense_depth"][0, reference]}
    context = cache["kwargs"]["context"]
    for name, transform in (
        ("own_mono_zero", lambda item: torch.zeros_like(item)),
        ("own_mono_roll", lambda item: torch.roll(item, (item.shape[-2] // 3, item.shape[-1] // 3), (-2, -1))),
    ):
        kwargs = dict(cache["kwargs"], context={key: transform(item) for key, item in context.items()})
        dense = own.dense_decoder(*cache["args"], **kwargs)
        anchors.update(collect(dense, name))
        full[name + "_final"] = dense["depth"][0, reference]
    return anchors, full


def run(plan, device):
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "frames.jsonl").exists():
        raise FileExistsError("已有实验记录，请使用新的输出目录")
    torch.set_num_threads(2)
    seed_all(plan.get("seed", 43))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    protocol = read_json(plan["protocol"])
    settings = protocol["evaluation"]
    reference = settings["reference_index"]
    adapter, cfg = make_model(plan["official"], plan["official"].get("weights", ""), device)
    if cfg["model"].get("range_refinement_steps", 1) != 0:
        raise ValueError("本实验只诊断首遍，须显式关闭范围重估")
    net = adapter.native_model
    raw_mono, raw_match, original = raw_components(net, plan["pretrained_weights"])
    audit = {
        "official_mono_vs_raw_hero": module_difference(net.encoder, original["encoder"]),
        "official_matching_vs_raw_hero": module_difference(net.matching_model, original["matching_model"]),
        "parameters": {name: sum(p.numel() for p in module.parameters()) for name, module in (
            ("mono_encoder", net.encoder), ("matching_encoder", net.matching_model),
            ("cost_mlp", net.cost_volume), ("cost_vit", net.cost_volume_net), ("dpt", net.depth_decoder))},
    }
    own = None
    if "own" in plan:
        own, own_cfg = make_model(plan["own"], plan["own"]["weights"], device)
        mono_state = {key[len("model."):]: value for key, value in original["encoder"].items()}
        audit["own_mono_vs_raw_hero"] = module_difference(own.encoder.foundation.model, mono_state)
        audit["own_matching_vs_raw_hero"] = module_difference(own.pretrained_matching.encoder, original["matching_model"])
        audit["own_foundation_gates"] = {key: float(value.tanh()) for key, value in own.encoder.foundation_gates.items()}
        audit["own_parameters"] = {name: sum(p.numel() for p in module.parameters()) for name, module in (
            ("encoder", own.encoder), ("coarse", own.coarse), ("dense_decoder", own.dense_decoder),
            ("candidate_field", own.candidate_field))}
        audit["own_feature_dim"] = own_cfg["model"]["feature_dim"]
        audit["own_decoder_hidden"] = own_cfg["model"]["dense_decoder"]["hidden_dim"]

    all_rows, parity = {}, []
    metadata = {"plan": plan, "weight_audit": audit, "script_sha256": file_sha256(__file__),
                "official_checkpoint_sha256": file_sha256(plan["official"].get("weights") or cfg["model"]["weights"])}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    with torch.inference_mode(), (output / "frames.jsonl").open("w") as log:
        for split, indices in plan["splits"].items():
            dataset = comparison_dataset(protocol, split)
            loader = DataLoader(Subset(dataset, indices), batch_size=1, num_workers=plan.get("num_workers", 2),
                                pin_memory=device.type == "cuda", collate_fn=collate_frames)
            rows = []
            for index, batch in zip(indices, loader):
                batch = move(batch, device)
                inputs = batch["inputs"]
                cur, src, _ = adapter.prepare_native(inputs, reference)
                volumes = []
                handle = net.cost_volume.register_forward_hook(lambda module, args, result: volumes.append(result))
                try:
                    full = adapter.call_native(cur, src, "test")
                finally:
                    handle.remove()
                volume = volumes[0]
                mono = cur["cur_feats"]
                shape = tuple(size // 16 for size in inputs["rgb"].shape[-2:])
                depth = {"hero_full": map_depth(full["depth_pred_s0_b1hw"], inputs, reference)}
                replayed = replay(net, cur, src, volume, mono)
                difference = float((full["depth_pred_s0_b1hw"] - replayed["depth_pred_s0_b1hw"]).abs().max())
                if difference > 1e-5:
                    raise AssertionError(f"原生完整forward与缓存重放不一致：{difference}")
                parity.append(difference)
                matching_cur, matching_src = cur["matching_cur_feats"], src["matching_src_feats"]
                zero_descriptors = build_volume(net, cur, src, torch.zeros_like(matching_cur), torch.zeros_like(matching_src))
                rolled_sources = build_volume(net, cur, src, matching_cur,
                                               torch.roll(matching_src, matching_src.shape[-1] // 3, -1))
                for name, mono_mode, volume_value, volume_mode in (
                    ("mono_zero", "zero", volume, "keep"),
                    ("resnet_zero", "keep", zero_descriptors, "keep"),
                    ("mono_and_resnet_zero", "zero", zero_descriptors, "keep"),
                    ("cost_zero", "keep", volume, "zero"),
                    ("mono_and_cost_zero", "zero", volume, "zero"),
                    ("mono_roll", "roll", volume, "keep"),
                    ("source_roll", "keep", rolled_sources, "keep"),
                    ("mono_shallow_zero", "shallow_zero", volume, "keep"),
                    ("mono_deep_zero", "deep_zero", volume, "keep"),
                ):
                    changed = (alter_volume(volume_value[0], volume_mode), *volume_value[1:])
                    result = replay(net, cur, src, changed, alter_mono(mono, mono_mode, shape))
                    depth[name] = map_depth(result["depth_pred_s0_b1hw"], inputs, reference)
                original_mono = mono if audit["official_mono_vs_raw_hero"]["different_tensors"] == 0 else raw_mono(cur["image_b3hw"])
                original_cur, original_src = matching_cur, matching_src
                if audit["official_matching_vs_raw_hero"]["different_tensors"] != 0:
                    images = torch.cat((cur["image_b3hw"][:, None], src["image_b3hw"]), dim=1)
                    features = torch.cat([raw_match(image) for image in images.flatten(0, 1).split(1)], dim=0)
                    features = features.unflatten(0, images.shape[:2])
                    original_cur, original_src = features[:, 0], features[:, 1:].contiguous()
                original_volume = build_volume(net, cur, src, original_cur, original_src)
                for name, image_features, candidates in (
                    ("raw_hero_mono_weights", original_mono, volume),
                    ("raw_hero_match_weights", mono, original_volume),
                    ("raw_hero_both_weights", original_mono, original_volume),
                ):
                    result = replay(net, cur, src, candidates, image_features)
                    depth[name] = map_depth(result["depth_pred_s0_b1hw"], inputs, reference)
                # 原始代价体最大分数只作阶段诊断，使用其真实特征中心映射到公共网格。
                depth["raw_cost_argmax"] = map_depth(volume[1][:, None], inputs, reference, 0.25)
                anchors = {name: item[::4, ::4] for name, item in depth.items()}
                if own is not None:
                    own_anchors, own_full = own_depths(own, inputs, reference)
                    anchors.update(own_anchors)
                    depth.update(own_full)
                # 所有模型前向与干预结束后才读取标签。
                truth, valid = gt_reference(inputs, batch["targets"], settings)
                accepted = inputs["image_mask"][0, reference] & inputs["view_mask"][0, reference]
                row = {
                    "split": split, "index": index,
                    "sample_id": batch.get("sample_id"),
                    "anchors": {name: evaluate_depth(item, accepted[::4, ::4], truth[::4, ::4], valid[::4, ::4], settings) for name, item in anchors.items()},
                    "full_image": {name: evaluate_depth(item, accepted, truth, valid, settings) for name, item in depth.items()},
                }
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                rows.append(row)
                if len(rows) == 1 or len(rows) % 10 == 0:
                    print(f"{split} {len(rows)}/{len(indices)}: " + ", ".join(
                        f"{key}={row['anchors'][key]['all']['MAE_m']:.4f}m"
                        for key in ("hero_full", "mono_zero", "resnet_zero", "cost_zero")), flush=True)
            all_rows[split] = summarize(rows, settings)
            summary = {**metadata, "results": all_rows, "cached_forward_max_abs_m": max(parity)}
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"完成：{output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(read_json(args.config), torch.device(args.device))


if __name__ == "__main__":
    main()
