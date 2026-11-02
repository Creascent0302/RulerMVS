"""把 MatrixCity 静态街景转换为项目共享四视角 manifest。

仅配置文件控制：python tools/prepare_matrixcity.py --config config.json
配置包含 root、splits（每项的 rgb_directory/depth_directory/transforms），
以及可选 reference_slot、selection、geometry_groups_per_split。所有输入路径
相对 root；输出只写 root/processed 和 root/manifests，原始数据保持只读。

相机选择仅用标定与固定视锥探针。不同位置的静态图像不是同步四目采集，
manifest 明确记录这一差异；测试数据不借用训练数据的源图。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
from PIL import Image


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def paths_by_frame(directory, suffix):
    """帧名可有前导零；禁止同一 ID 对应多个文件。"""
    result = {}
    for path in sorted(directory.glob(f"*{suffix}")):
        if not path.stem.isdigit():
            raise ValueError(f"无法识别的帧文件名：{path}")
        index = int(path.stem)
        if index in result:
            raise ValueError(f"重复帧 ID：{directory} / {index}")
        result[index] = path
    return result


def load_poses(path):
    """沿用官方 MatrixCity 约定，并验证恢复后的旋转及公制平移。"""
    metadata = json.loads(path.read_text())
    angle = float(metadata["camera_angle_x"])
    if not 0 < angle < np.pi:
        raise ValueError("camera_angle_x 必须以弧度表示且位于 (0, pi)")
    basis = np.diag([1.0, -1.0, -1.0, 1.0])
    poses, statistics = {}, []
    for frame in metadata["frames"]:
        index = int(frame["frame_index"])
        if index in poses:
            raise ValueError(f"标定出现重复 frame_index：{index}")
        raw = np.asarray(frame["rot_mat"], dtype=np.float64)
        if raw.shape != (4, 4) or not np.isfinite(raw).all():
            raise ValueError(f"标定矩阵无效：{index}")
        if not np.allclose(raw[3], [0, 0, 0, 1], atol=1e-7):
            raise ValueError(f"非齐次相机矩阵：{index}")
        norms = np.linalg.norm(raw[:3, :3], axis=0)
        # 已核查的原始 MatrixCity rot_mat 把旋转整体缩小 100 倍，平移已为米。
        # 不把任意未知尺度自动归一化，以免掩盖错误格式。
        if not np.allclose(norms, 0.01, atol=1e-6):
            raise ValueError(f"rot_mat 旋转尺度不是官方 0.01 约定：{index}, {norms}")
        pose = raw.copy()
        pose[:3, :3] *= 100.0
        pose = basis @ pose @ basis
        if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5):
            raise ValueError(f"相机旋转不正交：{index}")
        determinant = float(np.linalg.det(pose[:3, :3]))
        if not np.isclose(determinant, 1.0, atol=1e-5):
            raise ValueError(f"相机旋转行列式异常：{index}, {determinant}")
        poses[index] = pose
        statistics.append({"frame_id": index, "raw_rotation_column_norms": norms.tolist(),
                           "corrected_rotation_determinant": determinant})
    return angle, poses, statistics


def project(points, transform, intrinsics):
    camera = points @ transform[:3, :3].T + transform[:3, 3]
    pixels = camera @ intrinsics.T
    uv = pixels[:, :2] / np.maximum(camera[:, 2:3], 1e-8)
    return uv, camera[:, 2]


def select_groups(poses, intrinsics, image_size, options):
    """按相机几何构造四视角，不读取 RGB 内容或深度标签。"""
    defaults = {"min_baseline_m": 0.05, "max_baseline_m": 120.0,
                "max_angle_degrees": 60.0, "min_overlap": 0.3,
                "probe_depths_m": [4, 8, 16, 32, 64, 128], "max_candidates": 128}
    settings = {**defaults, **options}
    if not (0 < settings["min_baseline_m"] < settings["max_baseline_m"]):
        raise ValueError("基线上下限无效")
    if not (0 <= settings["min_overlap"] <= 1 and settings["max_candidates"] >= 3):
        raise ValueError("视锥重叠门槛或候选数无效")
    depths = np.asarray(settings["probe_depths_m"], dtype=np.float64)
    if not np.isfinite(depths).all() or not (depths > 0).all():
        raise ValueError("固定视锥探针深度必须为有限正数")
    height, width = image_size
    u, v = np.meshgrid(np.linspace(0.1 * width, 0.9 * width - 1, 12),
                       np.linspace(0.1 * height, 0.9 * height - 1, 9))
    pixels = np.stack((u, v, np.ones_like(u)), -1).reshape(-1, 3)
    rays = pixels @ np.linalg.inv(intrinsics).T
    probes = (rays[None] * depths[:, None, None]).reshape(-1, 3)
    ids = sorted(poses)
    matrices = np.stack([poses[index] for index in ids])
    centers = matrices[:, :3, 3]
    inverse = np.linalg.inv(matrices)
    groups, skipped = [], []
    for reference_index, reference in enumerate(ids):
        pose = poses[reference]
        distances = np.linalg.norm(centers - pose[:3, 3], axis=1)
        relative_rotations = np.einsum("nij,jk->nik", inverse[:, :3, :3], pose[:3, :3])
        cosine = (np.trace(relative_rotations, axis1=1, axis2=2) - 1) / 2
        angles = np.arccos(np.clip(cosine, -1, 1))
        eligible = np.flatnonzero(
            (distances >= settings["min_baseline_m"])
            & (distances <= settings["max_baseline_m"])
            & (angles <= np.radians(settings["max_angle_degrees"]))
        )
        order = np.argsort(distances[eligible] + 2 * angles[eligible], kind="stable")
        candidates = eligible[order[:settings["max_candidates"]]]
        selected = []
        selected_centers = [centers[reference_index]]
        for candidate in candidates:
            center = centers[candidate]
            if any(np.linalg.norm(center - previous) < settings["min_baseline_m"]
                   for previous in selected_centers):
                continue
            uv, z = project(probes, inverse[candidate] @ pose, intrinsics)
            inside = ((z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
                      & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1))
            overlap = float(inside.mean())
            if overlap < settings["min_overlap"]:
                continue
            selected.append({"frame_id": ids[candidate], "baseline_m": float(distances[candidate]),
                             "angle_degrees": float(np.degrees(angles[candidate])),
                             "frustum_probe_overlap": overlap})
            selected_centers.append(center)
            if len(selected) == 3:
                break
        if len(selected) == 3:
            groups.append({"reference_frame_id": reference, "sources": selected})
        else:
            skipped.append({"frame_id": reference, "reason": "不足三个不同位置且视锥重叠的源视图",
                            "geometric_candidates": len(eligible), "accepted_sources": len(selected)})
    return groups, skipped, settings


def convert_frames(root, name, settings):
    rgb_folder = root / settings["rgb_directory"]
    depth_folder = root / settings["depth_directory"]
    angle, poses, pose_statistics = load_poses(root / settings["transforms"])
    audit_poses = poses
    if settings.get("transforms_origin"):
        original_angle, audit_poses, pose_statistics = load_poses(root / settings["transforms_origin"])
        if not np.isclose(original_angle, angle) or not set(poses).issubset(audit_poses):
            raise ValueError("官方过滤后与原始标定的内参/帧集合不一致")
        for index, pose in poses.items():
            if not np.allclose(pose, audit_poses[index], atol=1e-6):
                raise ValueError(f"过滤前后同帧位姿发生变化：{index}")
    rgbs = paths_by_frame(rgb_folder, ".png")
    depths = paths_by_frame(depth_folder, ".exr")
    ids = set(audit_poses)
    missing = {"missing_rgb": sorted(ids - rgbs.keys()), "missing_depth": sorted(ids - depths.keys()),
               "rgb_without_pose": sorted(rgbs.keys() - ids), "depth_without_pose": sorted(depths.keys() - ids)}
    if any(missing.values()):
        write_json(root / "processed" / name / "inventory_errors.json", missing)
        raise ValueError(f"{name} RGB/深度/标定帧未完整一一对应：{missing}")
    output = root / "processed" / name
    (output / "depth").mkdir(parents=True, exist_ok=True)
    rows, errors, paths = [], [], {}
    image_size = None
    for index in sorted(ids):
        try:
            with Image.open(rgbs[index]) as image:
                image.load()  # 完整解码，不仅检查文件头。
                if image.mode not in ("RGB", "RGBA"):
                    raise ValueError(f"图像模式应为 RGB/RGBA，实际为 {image.mode}")
                if image.mode == "RGBA" and image.getchannel("A").getextrema() != (255, 255):
                    raise ValueError("发现非不透明 RGBA，不能在没有有效图像掩码时静默丢弃 alpha")
                shape = image.size[::-1]
            if image_size is None:
                image_size = shape
            if shape != image_size:
                raise ValueError("同组原始图像尺寸不一致")
            raw = cv2.imread(str(depths[index]), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError("EXR 解码失败")
            if raw.ndim == 3:
                raw = raw[..., 0]
            if raw.shape != image_size:
                raise ValueError(f"EXR 与 RGB 尺寸不一致：{raw.shape} vs {image_size}")
            finite = np.isfinite(raw)
            saturated = finite & (raw >= 65000.0)
            valid = finite & (raw > 0) & ~saturated
            depth = np.asarray(raw / 100.0, dtype=np.float32)
            depth[~valid] = np.nan
            target = output / "depth" / f"{index:06d}.optical_z.npy"
            temporary = target.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, depth, allow_pickle=False)
            temporary.replace(target)
            paths[index] = {"rgb": rgbs[index].relative_to(root).as_posix(),
                            "depth": target.relative_to(root).as_posix(),
                            "source_filename": rgbs[index].name}
            rows.append({"frame_id": index, **paths[index], "image_size": list(image_size),
                         "in_official_filtered_transforms": index in poses,
                         "valid_depth_fraction": float(valid.mean()),
                         "saturated_65000cm_fraction": float(saturated.mean()),
                         "nonfinite_fraction": float((~finite).mean()),
                         "nonpositive_fraction": float((finite & (raw <= 0)).mean()),
                         "minimum_depth_m": float(depth[valid].min()) if valid.any() else None,
                         "maximum_depth_m": float(depth[valid].max()) if valid.any() else None})
        except (OSError, ValueError, cv2.error) as error:
            errors.append({"frame_id": index, "error": str(error)})
        if len(rows) % 100 == 0:
            print(f"{name}: 已完整核查并转换 {len(rows)}/{len(ids)} 帧", flush=True)
    (output / "frames.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write_json(output / "pose_audit.json", pose_statistics)
    write_json(output / "errors.json", errors)
    if errors or not rows:
        raise ValueError(f"{name} 转换失败，详见 {output / 'errors.json'}")
    height, width = image_size
    focal = 0.5 * width / np.tan(0.5 * angle)
    # 原生 Niantic 几何以 i+0.5 为像素中心，K 的主点为 W/2,H/2。
    # 项目共享几何以整数 i 为中心，因此先减 0.5；baseline 适配器再加回。
    intrinsics = np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]])
    camera_path = output / "cameras.json"
    write_json(camera_path, {"K": intrinsics.tolist(), "image_size": list(image_size),
                            "T_world_from_camera_cv": {str(key): value.tolist() for key, value in audit_poses.items()},
                            "rotation_scale_correction": 100, "translation_scale_correction": 1,
                            "depth_centimeters_to_meters": 0.01,
                            "pixel_center_convention": "integer; native principal point minus 0.5",
                            "source": settings["transforms"]})
    write_json(output / "official_filtered_out_frames.json", sorted(set(audit_poses) - set(poses)))
    return poses, intrinsics, image_size, paths, rows


def make_record(split, group, paths, poses, intrinsics, reference_slot):
    reference = group["reference_frame_id"]
    ordered = [source["frame_id"] for source in group["sources"]]
    ordered.insert(reference_slot, reference)
    cv_from_rig = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    world_from_rig = poses[reference] @ cv_from_rig
    frame_id = f"reference_{reference:06d}"
    record = {"sequence": f"matrixcity_small_city_street_outside_{split}", "frame_id": frame_id,
              "timestamp": None, "sync_basis": "frame_id", "sample_type": "static_multiview_tuple",
              "synchronized_capture": False, "reference_slot": reference_slot,
              "source_split": split, "reference_source_frame_id": reference,
              "selection_uses_ground_truth": False, "cameras": []}
    for slot, index in enumerate(ordered):
        record["cameras"].append({
            "camera_id": f"view_{slot}", "role": "wide", "frame_id": frame_id, "timestamp": None,
            "source_frame_id": index, "source_camera_id": "static_render_pose",
            "source_filename": paths[index]["source_filename"], "rgb": paths[index]["rgb"],
            "K": intrinsics.tolist(), "T_cam_from_rig": (np.linalg.inv(poses[index]) @ world_from_rig).tolist(),
            "targets": {"frame_id": frame_id, "depth": paths[index]["depth"], "depth_type": "optical_z",
                        "depth_encoding": "metric_npy", "depth_scale": 1.0, "max_valid_depth": 650.0},
        })
    return record


def geometry_audit(root, groups, paths, poses, intrinsics, count):
    """等间隔抽组核验 GT 重投影；不依据结果改组选图或剔除难帧。"""
    reports = []
    if not groups or count <= 0:
        return reports
    chosen = np.unique(np.linspace(0, len(groups) - 1, min(count, len(groups))).astype(int))
    for group_index in chosen:
        group = groups[group_index]
        reference = group["reference_frame_id"]
        depth = np.load(root / paths[reference]["depth"], allow_pickle=False)
        height, width = depth.shape
        v, u = np.mgrid[0:height:16, 0:width:16]
        z = depth[v, u].ravel()
        pixels = np.stack((u, v, np.ones_like(u)), -1).reshape(-1, 3)
        good = np.isfinite(z) & (z > 0)
        points = (pixels @ np.linalg.inv(intrinsics).T)[good] * z[good, None]
        for source in group["sources"]:
            frame = source["frame_id"]
            uv, predicted = project(points, np.linalg.inv(poses[frame]) @ poses[reference], intrinsics)
            inside = ((predicted > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
                      & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1))
            xy = np.rint(uv[inside]).astype(int)
            source_depth = np.load(root / paths[frame]["depth"], allow_pickle=False)
            sampled = source_depth[xy[:, 1], xy[:, 0]]
            valid = np.isfinite(sampled) & (sampled > 0)
            error = predicted[inside][valid] - sampled[valid]
            tolerance = np.maximum(0.1, 0.01 * sampled[valid])
            reports.append({
                "reference_frame_id": reference, "source_frame_id": frame,
                "reference_valid_probes": int(len(points)), "projected_inside": int(inside.sum()),
                "source_valid_probes": int(valid.sum()),
                "median_absolute_z_error_m": float(np.median(np.abs(error))) if len(error) else None,
                "p90_absolute_z_error_m": float(np.percentile(np.abs(error), 90)) if len(error) else None,
                "agreement_within_1percent_or_10cm": float((np.abs(error) <= tolerance).mean()) if len(error) else None,
                "behind_source_surface_fraction": float((error > tolerance).mean()) if len(error) else None,
                "in_front_of_source_surface_fraction": float((error < -tolerance).mean()) if len(error) else None,
            })
    return reports


def write_labeled_training_manifest(root):
    """仅派生训练监督清单；不改组选图，不依据 GT 过滤验证或测试集。"""
    root = Path(root).resolve()
    source = root / "manifests" / "train.jsonl"
    statistics_path = root / "processed" / "train" / "frames.jsonl"
    statistics = {row["frame_id"]: row for row in
                  (json.loads(line) for line in statistics_path.read_text().splitlines() if line.strip())}
    lines = [line for line in source.read_text().splitlines() if line.strip()]
    kept, excluded = [], []
    source_paths = set()
    for line in lines:
        record = json.loads(line)
        if record["source_split"] != "train" or record["official_source_split"] != "train":
            raise ValueError("监督派生清单只能读取训练分组")
        reference = record["reference_source_frame_id"]
        fraction = statistics[reference]["valid_depth_fraction"]
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError(f"参考帧有效深度统计无效：{reference}")
        if fraction == 0:
            excluded.append(reference)
            continue
        kept.append(line)
        source_paths.update(camera["rgb"] for camera in record["cameras"])
    overlap_counts = {}
    for split in ("val", "test"):
        path = root / "manifests" / f"{split}.jsonl"
        if path.exists():
            other_paths = {camera["rgb"] for line in path.read_text().splitlines() if line.strip()
                           for camera in json.loads(line)["cameras"]}
            overlap_counts[split] = len(source_paths & other_paths)
            if overlap_counts[split]:
                raise ValueError(f"train_labeled 与 {split} 共享原始图像")
    target = root / "manifests" / "train_labeled.jsonl"
    temporary = target.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(line + "\n" for line in kept))
    temporary.replace(target)
    audit = {
        "source_manifest": source.relative_to(root).as_posix(),
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "frame_statistics": statistics_path.relative_to(root).as_posix(),
        "output_manifest": target.relative_to(root).as_posix(),
        "original_train_tuples": len(lines), "labeled_train_tuples": len(kept),
        "excluded_no_valid_reference_depth_count": len(excluded),
        "excluded_reference_frame_ids": excluded,
        "rule": "仅剔除参考原始深度有效像素比例为0的训练组；不设任务深度范围、不改变既有四视角",
        "uses_ground_truth": True, "purpose": "仅为深度监督训练提供有标签参考",
        "reference_or_source_reselection": False, "original_train_val_test_modified": False,
        "shared_rgb_paths_with_other_splits": overlap_counts,
        "scope": "val/test保持原清单，不依据标签挑测试样本；train_labeled可继续使用原train的源视图",
    }
    write_json(root / "processed" / "train" / "labeled_manifest_audit.json", audit)
    print(f"train_labeled: {len(kept)} 组，剔除 {len(excluded)} 个无有效深度参考；原清单未修改", flush=True)
    return audit


def prepare(config):
    root = Path(config["root"]).resolve()
    reference_slot = int(config.get("reference_slot", 2))
    if not 0 <= reference_slot < 4:
        raise ValueError("reference_slot 必须在 0–3")
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": "MatrixCity small_city street/outside", "config": config, "splits": {},
               "sample_type": "static_multiview_tuple", "synchronized_capture": False,
               "rig_axes": "x前/y左/z上，参考相机中心定义局部rig原点",
               "depth_type": "optical_z", "depth_units": "meters",
               "limitations": ["不同静态渲染位姿，不是同步物理四目或动态驾驶数据。",
                               "官方train/test属于同一城市，不能称为新城市泛化。",
                               "天空无独立语义标签；仅报告非有限/非正/65000厘米以上深度无效区域。"]}
    validation_fraction = float(config.get("validation_fraction", 0.0))
    validation_axis = int(config.get("validation_axis", 0))
    if not (0 <= validation_fraction < 0.5 and validation_axis in (0, 1, 2)):
        raise ValueError("validation_fraction 应在 [0,0.5)，validation_axis 应为 0/1/2")
    if validation_fraction and "val" in config["splits"]:
        raise ValueError("已有显式val，不能再从train自动划分")
    summary["total_frames_checked_and_converted"] = 0
    used_rgb_paths = set()
    for name, settings in config["splits"].items():
        poses, intrinsics, image_size, paths, frame_rows = convert_frames(root, name, settings)
        rgb_paths = {str((root / value["rgb"]).resolve()) for value in paths.values()}
        if used_rgb_paths & rgb_paths:
            raise ValueError("不同split引用了相同的RGB文件")
        used_rgb_paths |= rgb_paths
        summary["total_frames_checked_and_converted"] += len(paths)
        partitions = {name: poses}
        if name == "train" and validation_fraction:
            coordinates = np.array([pose[validation_axis, 3] for pose in poses.values()])
            threshold = float(np.quantile(coordinates, 1 - validation_fraction))
            partitions = {
                "train": {key: pose for key, pose in poses.items() if pose[validation_axis, 3] < threshold},
                "val": {key: pose for key, pose in poses.items() if pose[validation_axis, 3] >= threshold},
            }
            summary["validation_split"] = {
                "method": "原始官方train按相机中心坐标固定空间半区划分；同一位置全部朝向归同一组",
                "axis_in_converted_world": validation_axis, "validation_fraction_requested": validation_fraction,
                "threshold_m": threshold, "validation_side": ">= threshold",
                "shared_frame_ids_between_train_and_val": [], "uses_ground_truth": False,
                "scope": "同城空间留出，不保证视野无交叠，也不是新地图泛化",
            }
        for output_name, pool in partitions.items():
            groups, skipped, selection = select_groups(pool, intrinsics, image_size, config.get("selection", {}))
            if not groups:
                write_json(root / "processed" / output_name / "unmatched_references.json", skipped)
                raise ValueError(f"{output_name} 没有可用四视角组，不自动放宽选择或跨split借图")
            records = [make_record(output_name, group, paths, pool, intrinsics, reference_slot) for group in groups]
            for record in records:
                record["official_source_split"] = name
            manifest = manifests / f"{output_name}.jsonl"
            manifest.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
            write_json(root / "processed" / output_name / "tuple_selection.json", groups)
            write_json(root / "processed" / output_name / "unmatched_references.json", skipped)
            diagnostics = geometry_audit(root, groups, paths, pool, intrinsics,
                                         int(config.get("geometry_groups_per_split", 8)))
            write_json(root / "processed" / output_name / "geometry_audit.json", diagnostics)
            pool_rows = [row for row in frame_rows if row["frame_id"] in pool]
            summary["splits"][output_name] = {
                "source_conversion_split": name, "available_input_frames": len(pool), "tuples": len(groups),
                "frames_checked_and_converted_in_original_split": len(paths),
                "official_filtered_pose_count_in_original_split": len(poses),
                "frames_preserved_but_excluded_by_official_transforms": len(paths) - len(poses),
                "references_without_three_sources": len(skipped), "manifest": manifest.relative_to(root).as_posix(),
                "image_size": list(image_size), "K": intrinsics.tolist(), "selection": selection,
                "selection_uses_ground_truth": False, "geometry_pairs_checked": len(diagnostics),
                "mean_valid_depth_fraction": float(np.mean([row["valid_depth_fraction"] for row in pool_rows])),
                "mean_saturated_depth_fraction": float(np.mean([row["saturated_65000cm_fraction"] for row in pool_rows])),
                "all_rgb_exr_pose_frames_accounted_for": True,
            }
            print(f"{output_name}: {len(pool)} 个可选参考，{len(groups)} 个四视角组，{len(skipped)} 个参考无法组装", flush=True)
    if "val" not in summary["splits"]:
        (manifests / "val.jsonl").write_text("")
        summary["validation_split"] = "未建立；val.jsonl故意留空，不可直接用于模型选择"
    write_json(root / "processed" / "preparation_summary.json", summary)
    if "train" in summary["splits"]:
        write_labeled_training_manifest(root)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="转换配置 JSON")
    arguments = parser.parse_args()
    configuration = json.loads(Path(arguments.config).read_text())
    if configuration.get("derive_labeled_only", False):
        write_labeled_training_manifest(configuration["root"])
    else:
        prepare(configuration)
