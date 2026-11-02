"""CARLA跨轨迹标定及固定TRAIN几何可见性统计；不加载网络。"""

import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from jar.data import build_dataset, collate_frames
from jar.geometry import pixel_grid, project, sample_maps, unproject
from jar.pretraining.comparison import write_json
from jar.pretraining.data import dataset_config
from research.near_mid_evaluation_20261028.metrics import ground_truth, regions


def digest(value, decimals=None):
    array = np.asarray(value, dtype=np.float64)
    if decimals is not None:
        array = array.round(decimals)
    return hashlib.sha256(array.tobytes()).hexdigest()


def geometry(batch, domain, ref):
    inputs, targets = batch["inputs"], batch["targets"]
    truth_full, valid_full = ground_truth(batch, dict(domain, reference_index=ref))
    truth = truth_full[:, 0, ::8, ::8].float()
    valid = valid_full[:, 0, ::8, ::8]
    near = valid & (truth < 10)
    h, w = inputs["rgb"].shape[-2:]
    grid = pixel_grid(*truth.shape[-2:], stride=8).reshape(-1, 2)
    K, T = inputs["K"][0].float(), inputs["T_cam_from_rig"][0].float()
    safe_z = torch.where(valid, truth, 1).flatten()
    world, rays = unproject(grid, safe_z, K[ref], T[ref])
    uv, source_z, legal, J = project(world, K, T, (h, w), inputs["view_mask"][0], inputs["image_mask"][0])
    legal[ref] = False
    legal &= torch.linalg.vector_norm((J * rays[None, :, None]).sum(-1), dim=-1) > 1e-5
    observed = sample_maps(targets["depth"][0, :, None].float(), uv, mode="nearest")[..., 0]
    known = sample_maps(targets["depth_valid"][0, :, None].float(), uv, mode="nearest")[..., 0] > .5
    known &= legal & torch.isfinite(observed) & (observed > 0)
    visible = known & ((observed - source_z).abs() < .02 * source_z.abs().clamp_min(1e-3))
    support = legal.sum(0).reshape_as(truth)
    visible_count = visible.sum(0).reshape_as(truth)
    known_count = known.sum(0).reshape_as(truth)
    edge = regions(truth_full, valid_full, domain)["gradient_band"][:, 0, ::8, ::8]
    masks = {"edge": edge, "support0": support == 0, "support1": support == 1, "support2plus": support >= 2,
             "visible_any": visible_count > 0, "known_no_visible": (known_count > 0) & (visible_count == 0),
             "no_known_source_label": known_count == 0}
    return {"all_pixels": int(valid.sum()), "near_pixels": int(near.sum()),
            "near_depth_sum": float(truth[near].double().sum()),
            **{name + "_pixels": int((mask & near).sum()) for name, mask in masks.items()}}


def normalized(record):
    n = record["near_pixels"]
    return dict(record, near_fraction= n / max(record["all_pixels"], 1),
                mean_near_depth=record["near_depth_sum"] / max(n, 1),
                fractions={k.removesuffix("_pixels"): v / max(n, 1)
                           for k, v in record.items() if k.endswith("_pixels") and k not in ("all_pixels", "near_pixels")})


def add_comparison(report, output):
    """复用已执行的诊断，不读取 VAL confirmation 的预测表现。"""
    comparisons = {}
    for label, folder, subset in (
        ("VAL_development8", "diagnostics", "development"),
        ("TEST_fixed16_explanatory", "diagnostics_test", "groups"),
    ):
        path = output / folder / "summary.json"
        if not path.exists():
            continue
        source = json.loads(path.read_text())
        comparisons[label] = {}
        for name, item in source["domains"].items():
            if not name.startswith("CARLA_new_"):
                continue
            groups = item[subset]
            count = groups["near"]["pixels"]
            comparison = {
                "near_pixels": count,
                "mean_near_depth": groups["near"]["means"]["gt_depth"],
                "source": str(path),
            }
            for target, key in (
                ("edge", "near_edge"),
                ("visible_any", "near_visible_any"),
                ("known_no_visible", "near_known_but_no_visible"),
                ("no_known_source_label", "near_no_known_source_label"),
                ("support0", "near_geometric_support_0"),
                ("support1", "near_geometric_support_1"),
                ("support2plus", "near_geometric_support_2plus"),
            ):
                comparison[target + "_fraction"] = groups[key]["pixels"] / max(count, 1)
            comparisons[label][name] = comparison
    report["existing_val_test_geometry"] = comparisons
    report["qualitative_rgb_observations"] = {
        "scope": "人工查看各轨迹固定首中末上相机 RGB，共 12 张；以下仅描述示例，不是语义统计。",
        "captured_data0": "三例可见校园或住宅建筑、铺装道路及平整草地，伴有种植树木。",
        "captured_data1": "三例可见校园砖楼、道路或步道、围栏、种植树木，与训练轨迹建筑道路环境相似。",
        "captured_data2": "三例可见校园建筑、体育场地或球场、道路与围栏，近处较多平整铺装区域。",
        "captured_data3": "三例均可见较多树木灌丛、起伏或非铺装地面、强树荫，近景植被细碎。",
        "interpretation": "示例显示 TEST 场景类型变化；GT 边界与跨视图一致率的变化提供独立几何证据。不能仅凭这些示例估计场景语义比例。",
    }
    return report


def main():
    torch.set_num_threads(2)
    config = json.loads((ROOT / "research/v2_restart_20261029/configs/train_carla_new.json").read_text())
    domain = config["datasets"][0]
    output = ROOT / "runs/active/near_evidence_20261030"
    rows = {split: [json.loads(line) for line in Path(filename).read_text().splitlines()]
            for split, filename in domain["manifests"].items()}
    tracks = collections.defaultdict(list)
    for split, records in rows.items():
        for index, record in enumerate(records):
            tracks[record["sequence"]].append((split, index, record))
    cameras = collections.defaultdict(lambda: collections.defaultdict(list))
    first = {}
    for track, records in tracks.items():
        for _, _, record in records:
            for cam in record["cameras"]:
                key = cam["camera_id"]
                K, T = np.asarray(cam["K"]), np.asarray(cam["T_cam_from_rig"])
                cameras[track][key].append((K, T))
                first.setdefault(key, (K, T))
    calibration = {}
    for track, mapping in cameras.items():
        calibration[track] = {}
        for name, values in mapping.items():
            Ks = np.stack([a for a, b in values]); Ts = np.stack([b for a, b in values])
            centers = -np.einsum("nji,nj->ni", Ts[:, :3, :3], Ts[:, :3, 3])
            K0, T0 = first[name]
            C0 = -T0[:3, :3].T @ T0[:3, 3]
            calibration[track][name] = {
                "records": len(values), "K_exact_hashes": sorted(set(digest(x) for x in Ks)),
                "T_exact_unique_hash_count": len(set(digest(x) for x in Ts)),
                "T_rounded5_unique_hash_count": len(set(digest(x, 5) for x in Ts)),
                "intrinsic_first": Ks[0].tolist(), "rig_camera_center_mean": centers.mean(0).tolist(),
                "max_K_difference_to_track0": float(np.abs(Ks - K0).max()),
                "max_T_difference_to_track0": float(np.abs(Ts - T0).max()),
                "max_center_distance_to_track0_m": float(np.linalg.norm(centers - C0, axis=1).max()),
                "rotation_first": Ts[0, :3, :3].tolist(),
            }
    # 三张原RGB缩略图仅用于人工识别场景；不当作语义比例的证据。
    previews = {}
    for track, records in tracks.items():
        selected = [0, len(records) // 2, len(records) - 1]
        pictures, sources = [], []
        for local in selected:
            split, index, row = records[local]
            cam = next((c for c in row["cameras"] if c["camera_id"] == "stereo_up"), row["cameras"][0])
            source = Path(domain["root"]) / cam["rgb"]
            with Image.open(source) as image:
                pictures.append(image.convert("RGB").resize((448, 252)))
            sources.append({"split": split, "manifest_index": index, "frame_id": row["frame_id"], "camera": cam["camera_id"], "file": str(source)})
        strip = Image.new("RGB", (448 * 3, 252))
        for i, picture in enumerate(pictures): strip.paste(picture, (448 * i, 0))
        target = output / f"{track}_first_middle_last.jpg"
        strip.save(target, quality=90)
        previews[track] = {"sources": sources, "preview": str(target)}
    train = build_dataset(dataset_config(domain, config), "train", load_targets=True)
    statistics = {}
    camera_names = domain["camera"]["wide_ids"]
    for track in ("captured_data0", "captured_data2"):
        selected = tracks[track]
        positions = torch.linspace(0, len(selected) - 1, min(8, len(selected))).long().tolist()
        totals = {name: collections.defaultdict(float) for name in camera_names}
        frames = []
        for position in positions:
            _, index, row = selected[position]
            batch = collate_frames([train[index]])
            entries = {}
            for ref, name in enumerate(camera_names):
                record = geometry(batch, domain, ref)
                for key, value in record.items(): totals[name][key] += value
                entries[name] = normalized(record)
            frames.append({"manifest_index": index, "frame_id": row["frame_id"], "cameras": entries})
        statistics[track] = {"cameras": {name: normalized(value) for name, value in totals.items()}, "frames": frames}
    report = {"completed": True, "calibration": calibration, "previews": previews, "train_geometry": statistics,
              "protocol": "TRAIN每轨迹固定8帧、全部四参考；GT整数stride8，原2–80m有效区，near<10m，log-depth邻差>.1膨胀5px为边界，源GT一致阈值2%；不加载模型。",
              "limitations": "源GT一致性不是光度可匹配认证；首中末视觉观察只描述个例，不能当语义百分比。轨迹标定统计覆盖当前train/val/test清单所含RGB槽。"}
    write_json(output / "data_audit.json", add_comparison(report, output))
    print(json.dumps({"completed": True, "tracks": list(tracks), "train_frames": 16}, ensure_ascii=False))


if __name__ == "__main__": main()
