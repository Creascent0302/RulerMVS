"""将一个标准针孔 Hypersim 场景接入公共接口，仅作静态四视角接入测试。

选四图只读取标定和文件名。原深度在选组完成后才用于标签转换和几何核验。
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from prepare_hypersim_planar import optical_axis_factor


def read_h5(path):
    with h5py.File(path, "r") as stream:
        return stream["dataset"][:]


def load_cameras(root, metadata, scene, camera_name="cam_00", allow_uniform_scale=False):
    folder = root / "raw" / scene
    with open(metadata, newline="", encoding="utf-8") as stream:
        row = next(row for row in csv.DictReader(stream) if row["scene_name"] == scene)
    height = int(float(row["settings_output_img_height"]))
    width = int(float(row["settings_output_img_width"]))
    matrix = np.array([[float(row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)]
                       for i in range(3)])
    if not np.allclose(matrix, np.diag(np.diag(matrix)), atol=1e-10):
        raise ValueError("本样例转换器仅支持标准针孔，不将倾斜相机静默近似为针孔")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or matrix[2, 2] >= 0:
        raise ValueError("Hypersim 相机坐标约定不符")
    intrinsics = np.array([
        [-width * matrix[2, 2] / (2 * matrix[0, 0]), 0, (width - 1) / 2],
        [0, -height * matrix[2, 2] / (2 * matrix[1, 1]), (height - 1) / 2],
        [0, 0, 1],
    ])
    with open(folder / "_detail/metadata_scene.csv", newline="", encoding="utf-8") as stream:
        values = {row["parameter_name"]: row["parameter_value"] for row in csv.DictReader(stream)}
    scale = float(values["meters_per_asset_unit"])
    camera = folder / "_detail" / camera_name
    indices = read_h5(camera / "camera_keyframe_frame_indices.hdf5").reshape(-1)
    positions = read_h5(camera / "camera_keyframe_positions.hdf5")
    rotations = read_h5(camera / "camera_keyframe_orientations.hdf5")
    lookup = {int(frame): index for index, frame in enumerate(indices)}
    frames = sorted(int(path.name.split(".")[1]) for path in
                    (folder / "images" / f"scene_{camera_name}_final_preview").glob("frame.*.tonemap.jpg"))
    poses = {}
    for frame in frames:
        index = lookup[frame]
        pose = np.eye(4)
        basis = rotations[index]
        if allow_uniform_scale:
            # 正均匀缩放不改变针孔射线方向；仅去除经严格验证的共同尺度。
            # 不修正非均匀缩放、剪切或反射，不改变平移与原始距离标签。
            basis_scale = np.linalg.norm(basis) / np.sqrt(3)
            if not np.isfinite(basis_scale) or basis_scale <= 0:
                raise ValueError(f"无效相机基底尺度：{scene}/{camera_name}/{frame}")
            normalized = basis / basis_scale
            if not np.allclose(normalized.T @ normalized, np.eye(3), atol=1e-5):
                raise ValueError(f"相机基底不是旋转或正均匀缩放：{scene}/{camera_name}/{frame}")
            if not np.isclose(basis_scale, 1, atol=1e-5):
                basis = normalized
        pose[:3, :3] = basis @ np.diag([1, -1, -1])
        pose[:3, 3] = positions[index] * scale
        if not np.isfinite(pose).all() or not np.allclose(
            pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5
        ) or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5):
            raise ValueError(f"无效相机位姿：{frame}")
        poses[frame] = pose
    return folder, (height, width), matrix, intrinsics, scale, poses


def project(points, transform, intrinsics):
    camera = points @ transform[:3, :3].T + transform[:3, 3]
    pixels = camera @ intrinsics.T
    uv = pixels[:, :2] / np.maximum(camera[:, 2:3], 1e-8)
    return uv, camera[:, 2]


def select_tuple(poses, intrinsics, image_size, reference_slot):
    """固定米制视锥探针估计重叠，随后按位姿距离选择相近的真实视图。"""
    height, width = image_size
    u, v = np.meshgrid(np.linspace(0.1 * width, 0.9 * width - 1, 12),
                       np.linspace(0.1 * height, 0.9 * height - 1, 9))
    rays = np.stack((u, v, np.ones_like(u)), -1).reshape(-1, 3) @ np.linalg.inv(intrinsics).T
    probes = (rays[None] * np.array([1, 2, 4, 8, 16])[:, None, None]).reshape(-1, 3)
    groups = []
    for reference, pose in poses.items():
        sources = []
        for frame, other in poses.items():
            if frame == reference:
                continue
            relative = np.linalg.inv(other) @ pose
            baseline = float(np.linalg.norm(relative[:3, 3]))
            angle = float(np.arccos(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1)))
            uv, z = project(probes, relative, intrinsics)
            inside = ((z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
                      & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1))
            overlap = float(inside.mean())
            if baseline >= 0.01 and angle <= np.pi / 3 and overlap >= 0.35:
                sources.append({"frame": frame, "baseline_m": baseline,
                                "angle_degrees": float(np.degrees(angle)),
                                "frustum_probe_overlap": overlap,
                                "pose_cost": baseline + 2 * angle})
        sources.sort(key=lambda value: (value["pose_cost"], value["frame"]))
        if len(sources) >= 3:
            chosen = sources[:3]
            groups.append((sum(value["pose_cost"] for value in chosen), reference, chosen))
    if not groups:
        raise ValueError("没有满足非零基线、相似朝向和视锥重叠的四个真实视图")
    _, reference, sources = min(groups, key=lambda value: (value[0], value[1]))
    ordered = [value["frame"] for value in sources]
    ordered.insert(reference_slot, reference)
    return ordered, {"reference_frame": reference, "sources": sources,
                     "uses_ground_truth": False, "probe_depths_m": [1, 2, 4, 8, 16],
                     "minimum_baseline_m": 0.01, "minimum_probe_overlap": 0.35,
                     "maximum_angle_degrees": 60,
                     "rule": "每参考按平移米数+2×旋转弧度选三近邻，再取总位姿代价最小的组合"}


def check_depth_geometry(depths, ordered, poses, intrinsics):
    """选组后独立核验；报告遮挡与不一致比例，不以这些数值重新选组。"""
    reference = ordered[0]
    depth = depths[reference]
    height, width = depth.shape
    v, u = np.mgrid[0:height:8, 0:width:8]
    z = depth[v, u].ravel()
    pixels = np.stack((u, v, np.ones_like(u)), -1).reshape(-1, 3)
    good = np.isfinite(z) & (z > 0)
    points = (pixels @ np.linalg.inv(intrinsics).T)[good] * z[good, None]
    reports = []
    for frame in ordered[1:]:
        uv, projected = project(points, np.linalg.inv(poses[frame]) @ poses[reference], intrinsics)
        inside = ((projected > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
                  & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1))
        xy = np.rint(uv[inside]).astype(int)
        target = depths[frame][xy[:, 1], xy[:, 0]]
        valid = np.isfinite(target) & (target > 0)
        expected, target = projected[inside][valid], target[valid]
        error = expected - target
        tolerance = np.maximum(0.02, 0.01 * target)
        reports.append({
            "reference_frame": reference, "source_frame": frame,
            "reference_valid_probes": int(len(points)), "projected_inside": int(inside.sum()),
            "source_valid_probes": int(valid.sum()),
            "absolute_error_median_m": float(np.median(np.abs(error))) if len(error) else None,
            "absolute_error_p90_m": float(np.percentile(np.abs(error), 90)) if len(error) else None,
            "depth_agreement_fraction": float((np.abs(error) <= tolerance).mean()) if len(error) else None,
            "behind_source_surface_fraction": float((error > tolerance).mean()) if len(error) else None,
            "in_front_of_source_surface_fraction": float((error < -tolerance).mean()) if len(error) else None,
        })
    return reports


def prepare(root, metadata, scene, reference_slot):
    root = Path(root).resolve()
    folder, image_size, matrix, intrinsics, scale, poses = load_cameras(root, metadata, scene)
    ordered, selection = select_tuple(poses, intrinsics, image_size, reference_slot)
    # 从此处才开始读取GT；它不能影响上面的四视图选择。
    factor = optical_axis_factor(matrix, *image_size)
    v, u = np.mgrid[0:image_size[0]:8, 0:image_size[1]:8]
    rays = np.stack((u, v, np.ones_like(u)), -1) @ np.linalg.inv(intrinsics).T
    factor_difference = float(np.abs(1 / np.linalg.norm(rays, axis=-1) - factor[::8, ::8]).max())
    if factor_difference > 1e-5:
        raise ValueError("K反投影射线与原始距离转换射线不一致")
    depths, depth_paths, depth_statistics = {}, {}, {}
    output = root / "processed" / scene / "cam_00"
    output.mkdir(parents=True, exist_ok=True)
    for frame in poses:
        name = f"frame.{frame:04d}"
        rgb = folder / "images/scene_cam_00_final_preview" / f"{name}.tonemap.jpg"
        with Image.open(rgb) as image:
            if image.size[::-1] != image_size:
                raise ValueError(f"RGB尺寸与标定不同：{rgb}")
        distance = read_h5(folder / "images/scene_cam_00_geometry_hdf5" / f"{name}.depth_meters.hdf5")
        if distance.shape != image_size:
            raise ValueError(f"深度尺寸与标定不同：{frame}")
        depth = np.asarray(distance * factor, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        depth[~valid] = np.nan
        if not valid.any():
            raise ValueError(f"该帧无有效深度：{frame}")
        target = output / f"{name}.optical_z.npy"
        np.save(target, depth, allow_pickle=False)
        depths[frame], depth_paths[frame] = depth, target.relative_to(root).as_posix()
        depth_statistics[str(frame)] = {"valid_fraction": float(valid.mean()),
                                        "minimum_m": float(depth[valid].min()),
                                        "maximum_m": float(depth[valid].max())}
    cv_to_rig = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    cv_from_rig = np.eye(4)
    cv_from_rig[:3, :3] = cv_to_rig.T
    world_from_rig = poses[ordered[reference_slot]] @ cv_from_rig
    record = {"sequence": f"hypersim_{scene}_cam_00", "frame_id": "tuple_0000",
              "timestamp": None, "sync_basis": "frame_id",
              "sample_type": "static_multiview_tuple", "synchronized_capture": False,
              "reference_slot": reference_slot, "cameras": []}
    for slot, frame in enumerate(ordered):
        record["cameras"].append({
            "camera_id": f"view_{slot}", "role": "wide", "frame_id": record["frame_id"],
            "timestamp": None, "source_frame_id": f"{frame:04d}", "source_camera_id": "cam_00",
            "rgb": f"raw/{scene}/images/scene_cam_00_final_preview/frame.{frame:04d}.tonemap.jpg",
            "K": intrinsics.tolist(),
            "T_cam_from_rig": (np.linalg.inv(poses[frame]) @ world_from_rig).tolist(),
            "targets": {"frame_id": record["frame_id"], "depth": depth_paths[frame],
                        "depth_type": "optical_z", "depth_encoding": "metric_npy",
                        "depth_scale": 1.0, "max_valid_depth": 1000.0},
        })
    manifest = root / "manifests/smoke.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    diagnostic_order = [ordered[reference_slot]] + [frame for slot, frame in enumerate(ordered) if slot != reference_slot]
    summary = {"scene": scene, "sample_type": "static_multiview_tuple",
               "synchronized_capture": False,
               "official_training_scene": True if scene == "ai_011_010" else None,
               "purpose": "下载、几何和模型接入测试；不是未见场景泛化评测",
               "manifest": str(manifest), "image_size": list(image_size),
               "converted_frames": sorted(poses), "selected_frames_by_slot": ordered,
               "reference_slot": reference_slot, "selection": selection,
               "meters_per_asset_unit": scale, "K": intrinsics.tolist(),
               "rig_axes": "x前/y左/z上；原点与朝向由参考视图定义",
               "depth_definition": "原米制射线距离乘单位射线在光轴上的投影，输出米制Z-depth",
               "intrinsics_note": "使用W/H像素中心射线；不同于原生Hypersim loader用W-1/H-1估计焦距",
               "all_pose_rotations_orthonormal": True,
               "ray_factor_max_difference": factor_difference,
               "depth_statistics": depth_statistics,
               "post_selection_geometry": check_depth_geometry(depths, diagnostic_order, poses, intrinsics),
               "geometry_note": "最近邻源深度采样；遮挡和边界会产生不一致，不据此筛选输入或修改标签"}
    path = root / "sample_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="含 raw/场景名 的数据盘目录")
    parser.add_argument("--metadata", required=True, help="官方 metadata_camera_parameters.csv")
    parser.add_argument("--scene", default="ai_011_010")
    parser.add_argument("--reference-slot", type=int, choices=range(4), default=2)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.metadata, args.scene, args.reference_slot),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
