"""检查 TartanAir V1 原始文件，生成公共接口的静态四视角清单。

只接受 JSON 配置；原始 RGB、深度和分割保持不变，不需要模型或网络。
四个观测来自同一轨迹不同帧，不能称为同步四目或未见场景测试集。
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SIZE = (480, 640)
OFFICIAL_K = np.array([[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1.0]])
# 官方网络以半整数为像素中心；公共接口使用整数中心，主点同步减去 0.5。
INTRINSICS = OFFICIAL_K.copy()
INTRINSICS[:2, 2] -= 0.5
NED_TO_CV = np.array([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]])
CV_FROM_RIG = np.array([[0.0, -1, 0, 0], [0, 0, -1, 0],
                        [1, 0, 0, 0], [0, 0, 0, 1]])


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def quaternion_rotation(quaternion):
    """pose_left.txt 使用 xyzw；归一化只修正文本保存的舍入误差。"""
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-3:
        raise ValueError(f"位姿四元数不是单位四元数：范数 {norm}")
    x, y, z, w = quaternion / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_trajectory(folder, maximum_depth):
    """逐帧解码检查文件；标签内容不会参与后续四视图选择。"""
    rows = np.loadtxt(folder / "pose_left.txt", ndmin=2)
    if rows.shape[1] != 7 or not np.isfinite(rows).all():
        raise ValueError(f"位姿必须是有限的 [帧数,7] 矩阵：{folder}")
    frames = sorted(int(path.name[:6]) for path in (folder / "image_left").glob("??????_left.png"))
    expected = list(range(len(rows)))
    if frames != expected:
        raise ValueError(f"RGB 帧号与位姿行不一一对应：{folder}")
    for directory, suffix in [("depth_left", "_left_depth.npy"), ("seg_left", "_left_seg.npy")]:
        ids = sorted(int(path.name[:6]) for path in (folder / directory).glob("??????" + suffix))
        if ids != expected:
            raise ValueError(f"{directory} 帧号与 RGB 不一致：{folder}")
    poses, statistics = {}, []
    for frame, row in enumerate(rows):
        name = f"{frame:06d}_left"
        with Image.open(folder / "image_left" / (name + ".png")) as image:
            image.load()
            if image.mode != "RGB" or image.size[::-1] != IMAGE_SIZE:
                raise ValueError(f"RGB 格式/尺寸错误：{folder}/{name}")
        depth = np.load(folder / "depth_left" / (name + "_depth.npy"), allow_pickle=False)
        segmentation = np.load(folder / "seg_left" / (name + "_seg.npy"), allow_pickle=False)
        if depth.shape != IMAGE_SIZE or depth.dtype != np.float32:
            raise ValueError(f"深度应为 480×640 float32 米制数组：{folder}/{name}")
        if segmentation.shape != IMAGE_SIZE or not np.issubdtype(segmentation.dtype, np.integer):
            raise ValueError(f"分割应为 480×640 整数数组：{folder}/{name}")
        finite = np.isfinite(depth)
        valid = finite & (depth > 0) & (depth < maximum_depth)
        statistics.append({
            "frame": frame, "valid_depth_pixels": int(valid.sum()),
            "nonfinite_depth_pixels": int((~finite).sum()),
            "nonpositive_depth_pixels": int((finite & (depth <= 0)).sum()),
            "out_of_range_depth_pixels": int((finite & (depth >= maximum_depth)).sum()),
            "valid_depth_min_m": float(depth[valid].min()) if valid.any() else None,
            "valid_depth_max_m": float(depth[valid].max()) if valid.any() else None,
        })
        pose = np.eye(4)
        pose[:3, :3] = NED_TO_CV @ quaternion_rotation(row[3:]) @ NED_TO_CV.T
        pose[:3, 3] = NED_TO_CV @ row[:3]
        if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-8):
            raise ValueError(f"旋转矩阵不正交：{folder}/{name}")
        poses[frame] = pose
    return poses, statistics


def fixed_probes(settings):
    height, width = IMAGE_SIZE
    u, v = np.meshgrid(np.linspace(0.1 * width, 0.9 * width - 1, 12),
                       np.linspace(0.1 * height, 0.9 * height - 1, 9))
    rays = np.stack((u, v, np.ones_like(u)), -1).reshape(-1, 3) @ np.linalg.inv(INTRINSICS).T
    depths = np.asarray(settings["probe_depths_m"], dtype=float)
    if depths.ndim != 1 or len(depths) == 0 or not np.isfinite(depths).all() or (depths <= 0).any():
        raise ValueError("probe_depths_m 须为固定正深度列表，不能来自当前标签")
    return (rays[None] * depths[:, None, None]).reshape(-1, 3)


def choose_tuples(poses, settings):
    """只使用帧号、标定和配置；每个合格参考帧至多生成一个四视图样本。"""
    probes = fixed_probes(settings)
    ids = np.array(sorted(poses))
    inverses = {frame: np.linalg.inv(pose) for frame, pose in poses.items()}
    selected, rejected = [], []
    height, width = IMAGE_SIZE
    for reference in ids:
        neighbors = ids[(np.abs(ids - reference) <= settings["frame_window"]) & (ids != reference)]
        candidates = []
        for frame in neighbors:
            relative = inverses[frame] @ poses[reference]
            baseline = float(np.linalg.norm(relative[:3, 3]))
            angle = float(np.arccos(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1)))
            if not settings["minimum_baseline_m"] <= baseline <= settings["maximum_baseline_m"]:
                continue
            if np.degrees(angle) > settings["maximum_angle_degrees"]:
                continue
            camera = probes @ relative[:3, :3].T + relative[:3, 3]
            pixels = camera @ INTRINSICS.T
            uv = pixels[:, :2] / np.maximum(camera[:, 2:3], 1e-8)
            inside = ((camera[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
                      & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1))
            overlap = float(inside.mean())
            if overlap < settings["minimum_probe_overlap"]:
                continue
            cost = abs(np.log(baseline / settings["preferred_baseline_m"])) + 2 * angle + 1 - overlap
            candidates.append({"frame": int(frame), "baseline_m": baseline,
                               "angle_degrees": float(np.degrees(angle)),
                               "frustum_probe_overlap": overlap, "pose_cost": float(cost)})
        candidates.sort(key=lambda value: (value["pose_cost"], value["frame"]))
        chosen = []
        for candidate in candidates:
            center = poses[candidate["frame"]][:3, 3]
            # 参考-源基线已检查；还须排除位置几乎相同的源-源组合。
            if any(np.linalg.norm(center - poses[source["frame"]][:3, 3])
                   < settings["minimum_baseline_m"] for source in chosen):
                continue
            chosen.append(candidate)
            if len(chosen) == 3:
                break
        if len(chosen) < 3:
            rejected.append({"reference": int(reference), "eligible_sources": len(candidates),
                             "separated_sources": len(chosen)})
            continue
        selected.append({"reference": int(reference), "sources": chosen})
    return selected, rejected


def make_record(root, folder, trajectory, group, poses, reference_slot, maximum_depth):
    reference = group["reference"]
    ordered = [source["frame"] for source in group["sources"]]
    ordered.insert(reference_slot, reference)
    world_from_rig = poses[reference] @ CV_FROM_RIG
    tuple_id = f"reference_{reference:06d}"
    record = {"sequence": f"tartanair_{trajectory.replace('/', '_')}", "frame_id": tuple_id,
              "timestamp": None, "sync_basis": "frame_id", "reference_slot": reference_slot,
              "sample_type": "static_multiview_tuple", "synchronized_capture": False,
              "source_selection_uses_ground_truth": False, "cameras": []}
    for slot, frame in enumerate(ordered):
        name = f"{frame:06d}_left"
        record["cameras"].append({
            "camera_id": f"view_{slot}", "role": "wide", "frame_id": tuple_id,
            "timestamp": None, "source_frame_id": f"{frame:06d}", "source_camera_id": "left",
            "rgb": (folder / "image_left" / (name + ".png")).relative_to(root).as_posix(),
            "K": INTRINSICS.tolist(),
            "T_cam_from_rig": (np.linalg.inv(poses[frame]) @ world_from_rig).tolist(),
            "targets": {"frame_id": tuple_id, "depth_type": "optical_z",
                        "depth_encoding": "metric_npy", "depth_scale": 1.0,
                        "max_valid_depth": maximum_depth,
                        "depth": (folder / "depth_left" / (name + "_depth.npy")).relative_to(root).as_posix()},
            "native_segmentation": (folder / "seg_left" / (name + "_seg.npy")).relative_to(root).as_posix(),
        })
    return record


def geometry_diagnostics(folder, poses, groups, maximum_depth, samples):
    """选组完成后均匀抽样核验标签投影；不根据诊断指标删除难例。"""
    reports = []
    if not groups or samples <= 0:
        return reports
    height, width = IMAGE_SIZE
    v, u = np.mgrid[0:height:8, 0:width:8]
    rays = np.stack((u, v, np.ones_like(u)), -1).reshape(-1, 3) @ np.linalg.inv(INTRINSICS).T
    for index in np.unique(np.linspace(0, len(groups) - 1, min(samples, len(groups))).astype(int)):
        group = groups[index]
        reference = group["reference"]
        depth = np.load(folder / "depth_left" / f"{reference:06d}_left_depth.npy", allow_pickle=False)
        z = depth[v, u].ravel()
        good = np.isfinite(z) & (z > 0) & (z < maximum_depth)
        points = rays[good] * z[good, None]
        for source in group["sources"]:
            frame = source["frame"]
            target = np.load(folder / "depth_left" / f"{frame:06d}_left_depth.npy", allow_pickle=False)
            relative = np.linalg.inv(poses[frame]) @ poses[reference]
            camera = points @ relative[:3, :3].T + relative[:3, 3]
            projected = camera @ INTRINSICS.T
            uv = projected[:, :2] / np.maximum(camera[:, 2:3], 1e-8)
            inside = ((camera[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
                      & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1))
            xy = np.rint(uv[inside]).astype(int)
            measured = target[xy[:, 1], xy[:, 0]]
            valid = np.isfinite(measured) & (measured > 0) & (measured < maximum_depth)
            measured = measured[valid]
            error = camera[inside, 2][valid] - measured
            tolerance = np.maximum(0.05, 0.02 * measured)
            reports.append({"reference": reference, "source": frame,
                            "reference_valid_probes": int(good.sum()),
                            "projected_inside": int(inside.sum()), "valid_pairs": int(valid.sum()),
                            "absolute_error_median_m": float(np.median(np.abs(error))) if len(error) else None,
                            "absolute_error_p90_m": float(np.percentile(np.abs(error), 90)) if len(error) else None,
                            "agreement_fraction": float((np.abs(error) <= tolerance).mean()) if len(error) else None,
                            "behind_source_surface_fraction": float((error > tolerance).mean()) if len(error) else None,
                            "in_front_of_source_surface_fraction": float((error < -tolerance).mean()) if len(error) else None})
    return reports


def prepare(config):
    root = Path(config["root"]).resolve()
    raw = (root / config.get("raw_directory", "raw")).resolve()
    output = (root / config.get("output_directory", "prepared")).resolve()
    if not output.is_relative_to(root) or output == raw or output.is_relative_to(raw):
        raise ValueError("输出须在数据集根目录下，并与 raw 原始数据目录分开")
    splits = config["splits"]
    if set(splits) != {"train", "val", "test"} or any(not value for value in splits.values()):
        raise ValueError("splits 须显式设置非空 train/val/test 整轨迹清单")
    trajectories = [trajectory for split in splits.values() for trajectory in split]
    if len(trajectories) != len(set(trajectories)):
        raise ValueError("同一轨迹不能跨 train/val/test")
    available = {path.parent.relative_to(raw).as_posix() for path in raw.glob("*/*/P*/pose_left.txt")}
    if set(trajectories) != available:
        raise ValueError(f"splits 须完整覆盖已下载轨迹；未分配={available - set(trajectories)}，缺失={set(trajectories) - available}")
    settings = {"frame_window": 20, "minimum_baseline_m": 0.1, "maximum_baseline_m": 4.0,
                "preferred_baseline_m": 1.0, "maximum_angle_degrees": 30.0,
                "minimum_probe_overlap": 0.5, "probe_depths_m": [2, 5, 10, 20, 40, 80]}
    settings.update(config.get("selection", {}))
    if (settings["frame_window"] < 1 or not 0 < settings["minimum_baseline_m"] < settings["maximum_baseline_m"]
            or settings["preferred_baseline_m"] <= 0 or not 0 <= settings["minimum_probe_overlap"] <= 1):
        raise ValueError("无效的位姿选择配置")
    reference_slot = int(config.get("reference_slot", 2))
    maximum_depth = float(config.get("maximum_valid_depth_m", 1000.0))
    if reference_slot not in range(4) or not np.isfinite(maximum_depth) or maximum_depth <= 0:
        raise ValueError("无效的参考位置或最大有效深度")
    records = {split: [] for split in splits}
    report = {"dataset": "TartanAir V1", "config": config, "selection": settings,
              "synchronized_capture": False, "sample_type": "static_multiview_tuple",
              "selection_uses_ground_truth": False, "depth_definition": "米制光轴 Z-depth，原 float32 不缩放",
              "K_integer_pixel_centers": INTRINSICS.tolist(), "official_K_half_integer_centers": OFFICIAL_K.tolist(),
              "calibration_basis": "官方 TartanAirDataset 的 K 配合 BackprojectDepth(u+0.5,v+0.5)；公共 K 主点减 0.5 后射线一致",
              "pose_basis": "tx ty tz qx qy qz qw；NED->CV采用官方 C@T@C.T；参考CV视图再定义 x前/y左/z上 rig",
              "invalid_depth_policy": "仅有限且0<Z<maximum_valid_depth_m为监督；由公共loader掩码，无效/天空远值不当作free",
              "segmentation_policy": "逐帧校验并保留原标签；未知跨场景类别映射，不推测天空ID、不作为统一语义监督",
              "evaluation_limit": "本地划分仅按轨迹隔离；TartanAir为MVSAnywhere原训练数据来源，不能称原模型未见场景评测",
              "geometry_diagnostic_policy": "组选择后抽样，最近邻源深度，容差max(0.05m,2%源深度)，包含遮挡/边缘不一致，不据此删样本",
              "trajectories": {}}
    for split, names in splits.items():
        for trajectory in sorted(names):
            folder = raw / trajectory
            print(f"检查 {trajectory}", flush=True)
            poses, statistics = load_trajectory(folder, maximum_depth)
            groups, rejected = choose_tuples(poses, settings)
            records[split].extend(make_record(root, folder, trajectory, group, poses,
                                              reference_slot, maximum_depth) for group in groups)
            diagnostics = geometry_diagnostics(folder, poses, groups, maximum_depth,
                                               int(config.get("geometry_samples_per_trajectory", 6)))
            audit = {"split": split, "input_frames": len(poses), "valid_four_view_tuples": len(groups),
                     "unusable_reference_count": len(rejected), "rejected_references": rejected,
                     "frame_audit": statistics, "selected_groups": groups, "geometry_diagnostics": diagnostics}
            audit_path = output / "audit" / (trajectory.replace("/", "_") + ".json")
            save_json(audit_path, audit)
            report["trajectories"][trajectory] = {key: audit[key] for key in
                                                ("split", "input_frames", "valid_four_view_tuples", "unusable_reference_count")}
            report["trajectories"][trajectory]["audit"] = str(audit_path)
            print(f"  {len(poses)}帧，{len(groups)}组，{len(rejected)}个参考无法组成四视图", flush=True)
    report["manifests"] = {}
    for split, values in records.items():
        if not values:
            raise ValueError(f"{split} 没有可用四视图，不生成伪数据")
        path = output / "manifests" / (split + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for record in values:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(path)
        report["manifests"][split] = {"path": str(path), "samples": len(values)}
    save_json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="所有处理参数由此 JSON 文件控制")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = json.load(stream)
    print(json.dumps(prepare(config), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
