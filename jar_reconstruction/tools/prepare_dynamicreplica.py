"""将 DynamicReplica 原生同步双目验证子集转为共享接口。

保留原 PNG 和注释；仅另存无损 float32 深度与有效掩码。不跨时间拼四目，
不把动态前景 mask 当深度有效区域，不把原 validation 改称 train/test。
"""

import argparse
import gzip
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


CV_FROM_RIG = np.array([[0.0, -1, 0, 0], [0, 0, -1, 0],
                        [1, 0, 0, 0], [0, 0, 0, 1]])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_array(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = np.load(path, mmap_mode="r", allow_pickle=False)
        if old.dtype != value.dtype or not np.array_equal(old, value, equal_nan=True):
            raise ValueError(f"已转换文件与原始数据不一致：{path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def ensure_disk_reserve(root, upcoming_bytes, config):
    reserve = int(config.get("minimum_free_disk_gib", 300) * 1024 ** 3)
    available = shutil.disk_usage(root).free
    if available < reserve + upcoming_bytes:
        raise RuntimeError(f"数据盘剩余 {available / 1024 ** 3:.1f} GiB，不足以写入下一阶段并保留 {reserve / 1024 ** 3:.0f} GiB")


def extract_selected(root, raw, pairs, config):
    """只提取所选同步帧的 RGB、深度、前景标签，保留原包及已有原始文件。"""
    names = sorted({item[key]["path"] for pair in pairs for item in pair
                    for key in ("image", "depth", "mask")})
    with zipfile.ZipFile(root / "archives/valid_001.zip") as archive:
        members = {item.filename: item for item in archive.infolist()}
        missing = set(names) - set(members)
        if missing:
            raise ValueError(f"完整序列的原包缺少文件：{sorted(missing)[:5]}")
        ensure_disk_reserve(root, sum(members[name].file_size for name in names
                                     if not (raw / name).exists()), config)
        for name in names:
            destination = raw / name
            if not destination.resolve().is_relative_to(raw.resolve()) or members[name].is_dir():
                raise ValueError(f"不安全的 ZIP 路径：{name}")
            if destination.exists():
                if destination.stat().st_size != members[name].file_size:
                    raise ValueError(f"已有原始文件大小不匹配，不自动覆盖：{destination}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            with archive.open(name) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(destination)


def decode_depth(path):
    # 官方存储的是 float16 位模式；astype(float32) 直接转换整数会得到错误深度。
    with Image.open(path) as image:
        bits = np.asarray(image)
        if bits.ndim != 2 or not np.issubdtype(bits.dtype, np.integer):
            raise ValueError(f"不是单通道整数深度 PNG：{path}")
        if bits.min() < 0 or bits.max() > 65535:
            raise ValueError(f"深度位模式超过 uint16：{path}")
        return bits.astype(np.uint16).view(np.float16).astype(np.float32)


def camera_from_annotation(annotation):
    """等价于官方 PyTorch3D→OpenCV 转换，输出本项目整数像素中心的 K。"""
    camera = annotation["viewpoint"]
    height, width = annotation["image"]["size"]
    half_size = np.array([width, height], dtype=np.float64) / 2
    convention = camera["intrinsics_format"].lower()
    if convention == "ndc_isotropic":
        scale = half_size.min()
    elif convention == "ndc_norm_image_bounds":
        scale = half_size
    else:
        raise ValueError(f"不支持的 NDC 约定：{convention}")
    focal = np.asarray(camera["focal_length"], dtype=np.float64) * scale
    principal = half_size - np.asarray(camera["principal_point"], dtype=np.float64) * scale
    # 原 MVSAnywhere 用 (u+0.5,v+0.5)；本项目用 (u,v)，故同步移动主点。
    k = np.array([[focal[0], 0, principal[0] - 0.5],
                  [0, focal[1], principal[1] - 0.5], [0, 0, 1]])
    axes = np.diag([-1.0, -1.0, 1.0])
    e = np.eye(4)
    e[:3, :3] = axes @ np.asarray(camera["R"], dtype=np.float64).T
    e[:3, 3] = axes @ np.asarray(camera["T"], dtype=np.float64)
    if (not np.isfinite(k).all() or not np.isfinite(e).all() or focal.min() <= 0
            or not np.allclose(e[:3, :3] @ e[:3, :3].T, np.eye(3), atol=2e-5)
            or abs(np.linalg.det(e[:3, :3]) - 1) > 2e-5):
        raise ValueError(f"无效相机：{annotation['sequence_name']}/{annotation['frame_number']}")
    return k, e


def synchronized_pairs(annotations, sequences, expected_frames):
    """按真正的 sequence/frame/camera 建组，不用列表顺序或 GT 深度选择样本。"""
    grouped = {sequence: {} for sequence in sequences}
    for annotation in annotations:
        sequence = annotation["sequence_name"]
        if sequence not in grouped:
            continue
        frame = int(annotation["frame_number"])
        camera = annotation["camera_name"]
        if camera not in ("left", "right"):
            raise ValueError(f"未声明的相机：{sequence}/{camera}")
        pair = grouped[sequence].setdefault(frame, {})
        if camera in pair:
            raise ValueError(f"重复相机注释：{sequence}/{frame}/{camera}")
        pair[camera] = annotation
    pairs = []
    for sequence, frames in grouped.items():
        if sorted(frames) != list(range(expected_frames)):
            raise ValueError(f"所选完整序列缺帧：{sequence}")
        for frame, pair in sorted(frames.items()):
            if set(pair) != {"left", "right"}:
                raise ValueError(f"缺少同帧双目：{sequence}/{frame}")
            ordered = [pair["left"], pair["right"]]
            times = [float(item["frame_timestamp"]) for item in ordered]
            if not np.isfinite(times).all() or abs(times[0] - times[1]) > 1e-8:
                raise ValueError(f"左右相机不同步：{sequence}/{frame}")
            if ordered[0]["image"]["size"] != ordered[1]["image"]["size"]:
                raise ValueError(f"左右相机尺寸不同：{sequence}/{frame}")
            pairs.append(ordered)
    return pairs


def convert_frame(annotation, raw, converted, minimum_depth, maximum_depth):
    sequence, frame = annotation["sequence_name"], int(annotation["frame_number"])
    camera = annotation["camera_name"]
    size = tuple(annotation["image"]["size"])
    with Image.open(raw / annotation["image"]["path"]) as image:
        image.load()
        if image.mode not in ("RGB", "RGBA") or image.size != size[::-1]:
            raise ValueError(f"RGB 格式或尺寸错误：{sequence}/{frame}/{camera}")
        alpha_range = [int(np.asarray(image)[..., 3].min()), int(np.asarray(image)[..., 3].max())] if image.mode == "RGBA" else None
    if float(annotation["depth"]["scale_adjustment"]) != 1.0:
        raise ValueError("发现非单位 scale_adjustment；不能静默改变原始深度单位")
    depth = decode_depth(raw / annotation["depth"]["path"])
    with Image.open(raw / annotation["mask"]["path"]) as image:
        foreground = np.asarray(image)
    if depth.shape != size or foreground.shape != size or not np.isin(foreground, [0, 255]).all():
        raise ValueError(f"深度/前景标签形状或编码错误：{sequence}/{frame}/{camera}")
    valid = np.isfinite(depth) & (depth >= minimum_depth) & (depth < maximum_depth)
    depth_path = converted / sequence / camera / f"{frame:06d}.npy"
    mask_path = converted / sequence / camera / f"{frame:06d}_valid.npy"
    save_array(depth_path, depth)
    save_array(mask_path, valid)
    statistics = {
        "sequence": sequence, "frame": frame, "camera": camera,
        "valid_fraction": float(valid.mean()), "nonfinite_pixels": int((~np.isfinite(depth)).sum()),
        "nonpositive_pixels": int((np.isfinite(depth) & (depth <= 0)).sum()),
        "foreground_fraction": float((foreground > 0).mean()), "alpha_range": alpha_range,
        "valid_depth_quantiles_native_unit": np.quantile(depth[valid], [0, .5, .95, 1]).tolist() if valid.any() else None,
    }
    return depth_path, mask_path, statistics


def record_for_pair(pair, root, raw, converted, config):
    k, e = zip(*(camera_from_annotation(annotation) for annotation in pair))
    reference = config["reference_slot"]
    world_from_rig = np.linalg.inv(e[reference]) @ CV_FROM_RIG
    relative = e[1] @ np.linalg.inv(e[0])
    baseline = float(np.linalg.norm(relative[:3, 3]))
    if baseline < 1e-6:
        raise ValueError("双目相机中心重合")
    sequence = pair[0]["sequence_name"]
    frame = str(pair[0]["frame_number"])
    units = config["geometry_units"]
    record = {
        "sequence": "dynamicreplica_" + sequence, "source_sequence": sequence,
        "frame_id": frame, "timestamp": float(pair[0]["frame_timestamp"]),
        "source_split": "valid", "reference_slot": reference, "num_views": 2,
        "geometry_units": units, "sample_type": "synchronized_stereo",
        "synchronized_capture": True, "source_selection_uses_ground_truth": False,
        "cameras": [],
    }
    statistics = []
    for index, annotation in enumerate(pair):
        depth, valid, stats = convert_frame(annotation, raw, converted,
                                            config["minimum_depth"], config["maximum_depth"])
        statistics.append(stats)
        record["cameras"].append({
            "camera_id": annotation["camera_name"], "role": "wide", "frame_id": frame,
            "source_frame_id": frame, "timestamp": record["timestamp"],
            "rgb": (raw / annotation["image"]["path"]).relative_to(root).as_posix(),
            "K": k[index].tolist(), "T_cam_from_rig": (e[index] @ world_from_rig).tolist(),
            "targets": {
                "frame_id": frame, "depth_type": "optical_z", "depth_units": units,
                "depth_encoding": "metric_npy" if units == "meters" else "scene_unit_npy",
                "depth_scale": 1.0, "max_valid_depth": config["maximum_depth"],
                "depth": depth.relative_to(root).as_posix(),
                "depth_valid": valid.relative_to(root).as_posix(),
                "foreground_mask_png": (raw / annotation["mask"]["path"]).relative_to(root).as_posix(),
                "foreground_mask_definition": "动态前景标签；不是深度有效性或语义类别标签",
            },
        })
    return record, statistics, {
        "baseline_native_unit": baseline,
        "relative_rotation_error": float(np.abs(relative[:3, :3] - np.eye(3)).max()),
        "relative_vertical_translation_native_unit": float(abs(relative[1, 3])),
        "relative_forward_translation_native_unit": float(abs(relative[2, 3])),
    }


def diagnose_pair(record, root, stride):
    """双向 GT 重投影只用于诊断，绝不参与输入构造或样本筛选。"""
    reports = []
    for reference, source in ((0, 1), (1, 0)):
        cameras = record["cameras"]
        ref, src = cameras[reference], cameras[source]
        depth = np.load(root / ref["targets"]["depth"], allow_pickle=False)
        valid = np.load(root / ref["targets"]["depth_valid"], allow_pickle=False)
        other = np.load(root / src["targets"]["depth"], allow_pickle=False)
        other_valid = np.load(root / src["targets"]["depth_valid"], allow_pickle=False)
        height, width = depth.shape
        v, u = np.mgrid[0:height:stride, 0:width:stride]
        select = valid[v, u]
        pixels = np.stack([u[select], v[select], np.ones(select.sum())], axis=-1)
        points = (pixels @ np.linalg.inv(ref["K"]).T) * depth[v[select], u[select], None]
        transform = np.array(src["T_cam_from_rig"]) @ np.linalg.inv(ref["T_cam_from_rig"])
        camera = points @ transform[:3, :3].T + transform[:3, 3]
        projected = camera @ np.array(src["K"]).T
        uv = projected[:, :2] / np.maximum(camera[:, 2:3], 1e-8)
        near = np.rint(uv).astype(int)
        inside = ((camera[:, 2] > 0) & (near[:, 0] >= 0) & (near[:, 0] < width)
                  & (near[:, 1] >= 0) & (near[:, 1] < height))
        near, predicted, uv = near[inside], camera[inside, 2], uv[inside]
        good = other_valid[near[:, 1], near[:, 0]]
        measured = other[near[good, 1], near[good, 0]]
        predicted = predicted[good]
        relative_error = np.abs(measured - predicted) / np.maximum(predicted, 1e-8)
        reports.append({
            "sequence": record["source_sequence"], "frame": record["frame_id"],
            "reference": reference, "source": source, "sampled_valid": int(select.sum()),
            "projected_inside": int(inside.sum()), "valid_comparisons": int(good.sum()),
            "median_relative_depth_error": float(np.median(relative_error)) if len(relative_error) else None,
            "agreement_within_2_percent": float((relative_error < .02).mean()) if len(relative_error) else None,
            "occluded_by_nearer_surface_fraction": float((measured < predicted * .98).mean()) if len(relative_error) else None,
            "median_vertical_reprojection_pixels": float(np.median(np.abs(uv[:, 1] - pixels[inside, 1]))) if inside.any() else None,
        })
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    config = json.loads(Path(parser.parse_args().config).read_text())
    if config["reference_slot"] != 0 or config["geometry_units"] not in ("meters", "scene_unit"):
        raise ValueError("本适配协议固定左目为 reference_slot=0，单位必须明确")
    root = Path(config["root"]).resolve()
    raw, prepared = root / "raw/valid", root / "prepared"
    with gzip.open(raw / "frame_annotations_valid.jgz", "rt", encoding="utf-8") as stream:
        annotations = json.load(stream)
    pairs = synchronized_pairs(annotations, config["sequences"], config["expected_frames_per_sequence"])
    extract_selected(root, raw, pairs, config)
    manifest = prepared / "manifests/val.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(".jsonl.tmp")
    statistics, geometry, diagnostics = [], [], []
    diagnostic_ids = set(np.linspace(0, config["expected_frames_per_sequence"] - 1,
                                    config["diagnostic_frames_per_sequence"]).astype(int).tolist())
    with temporary.open("w", encoding="utf-8") as stream:
        for index, pair in enumerate(pairs):
            if pair[0]["frame_number"] == 0:
                height, width = pair[0]["image"]["size"]
                # float32 深度+bool有效性，两相机；按整序列保守预留，已有输出不重复占空间。
                next_sequence_bytes = config["expected_frames_per_sequence"] * 2 * height * width * 5
                ensure_disk_reserve(root, next_sequence_bytes, config)
            record, stats, camera = record_for_pair(pair, root, raw, prepared / "depth", config)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            statistics.extend(stats)
            geometry.append(camera)
            if pair[0]["frame_number"] in diagnostic_ids:
                diagnostics.extend(diagnose_pair(record, root, config["diagnostic_pixel_stride"]))
            if (index + 1) % 100 == 0:
                print(f"已核对并转换 {index + 1}/{len(pairs)} 组同步双目", flush=True)
    temporary.replace(manifest)
    write_json(prepared / "frame_audit.json", statistics)
    write_json(prepared / "geometry_diagnostics.json", diagnostics)
    finite_errors = [x["median_relative_depth_error"] for x in diagnostics if x["median_relative_depth_error"] is not None]
    summary = {
        "config": config, "source_annotation_records": len(annotations), "selected_records": len(statistics),
        "synchronized_pairs": len(pairs), "incomplete_selected_pairs": 0,
        "original_split": "valid", "public_validation_subset_only": True,
        "num_views": 2, "reference_slot": 0, "image_size_hw": [720, 1280],
        "geometry_units": config["geometry_units"], "units_provenance": config["units_provenance"],
        "source_selection_uses_ground_truth": False, "depth_type": "optical_z",
        "foreground_mask_is_depth_validity": False, "pixel_center_convention": "integer; original screen principal point minus 0.5",
        "baseline_native_unit_range": [min(x["baseline_native_unit"] for x in geometry), max(x["baseline_native_unit"] for x in geometry)],
        "max_relative_rotation_error": max(x["relative_rotation_error"] for x in geometry),
        "minimum_valid_fraction": min(x["valid_fraction"] for x in statistics),
        "mean_valid_fraction": float(np.mean([x["valid_fraction"] for x in statistics])),
        "diagnostic_directional_pairs": len(diagnostics),
        "diagnostic_median_relative_depth_error": float(np.median(finite_errors)) if finite_errors else None,
        "diagnostic_mean_agreement_within_2_percent": float(np.mean([x["agreement_within_2_percent"] for x in diagnostics if x["agreement_within_2_percent"] is not None])),
        "manifest": manifest.relative_to(root).as_posix(),
        "limitations": ["原 validation 的 4/20 条序列，不是完整验证集", "原数据只有同步双目，不能充作同步四目CARLA实验", "动态前景mask不是多类别语义或占据真值", "合成场景深度以float16存储，float32转换不会恢复已量化精度"],
    }
    write_json(prepared / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
