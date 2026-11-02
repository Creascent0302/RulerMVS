"""用同帧雷达射线检验深度类型；只读原数据，不自动拟合或修改训练标定。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .carla_capture import carla_pose_matrix


def recover_radar_points(records, max_range=100.0):
    """显式假设：前三列与射线方向平行，第四列为距离。

    这是当前采集文件的经验解释，不是 CARLA 原始 radar.raw_data 的格式。
    前向雷达要求射线 x > 0，所以用第一列的符号消除径向速度正负号。
    停车时前三列全零，方向已丢失，必须剔除；不能将其补成正前方射线。
    """
    records = np.asarray(records, dtype=np.float64)
    if records.ndim != 2 or records.shape[1] != 4:
        raise ValueError("雷达数组必须为 [M,4]")
    norm = np.linalg.norm(records[:, :3], axis=1)
    valid = (
        np.isfinite(records).all(axis=1)
        & (norm > 1e-6)
        & (np.abs(records[:, 0]) > 1e-6)
        & (records[:, 3] > 0)
        & (records[:, 3] < max_range)
    )
    selected = records[valid]
    direction = selected[:, :3] * np.sign(selected[:, :1]) / norm[valid, None]
    return direction * selected[:, 3:4], valid


def radar_to_camera(points, radar_pose, camera_pose):
    """CARLA 雷达局部坐标 -> CARLA 相机局部坐标 -> CV 相机坐标，单位米。"""
    relative = np.linalg.inv(carla_pose_matrix(camera_pose)) @ carla_pose_matrix(radar_pose)
    local = points @ relative[:3, :3].T + relative[:3, 3]
    return np.stack([local[:, 1], -local[:, 2], local[:, 0]], axis=1)


def bilinear_sample(image, uv):
    """整数坐标为数组像素中心；调用前需排除图外点，不跨遮挡寻找最佳匹配。"""
    x0, y0 = np.floor(uv).astype(np.int64).T
    dx, dy = (uv - np.floor(uv)).T
    return (
        image[y0, x0] * (1 - dx) * (1 - dy)
        + image[y0, x0 + 1] * dx * (1 - dy)
        + image[y0 + 1, x0] * (1 - dx) * dy
        + image[y0 + 1, x0 + 1] * dx * dy
    )


def matched_samples(depth, camera_points, intrinsic, pixel_offset=0.0):
    """两种深度假设共用采样点和掩码；不按匹配误差筛点。

    排除 5x5 范围内深度变化超过 2.5% 的位置，降低栅格和遮挡边界影响。
    光学与碰撞几何不一定相同，剩余离群点仍计入统计，不冒充精确 GT。
    """
    z = camera_points[:, 2]
    positive = np.isfinite(camera_points).all(axis=1) & (z > 0)
    points = camera_points[positive]
    uv = points[:, :2] / points[:, 2:3]
    uv = uv * [intrinsic["fx"], intrinsic["fy"]]
    uv += [intrinsic["cx"] + pixel_offset, intrinsic["cy"] + pixel_offset]
    height, width = depth.shape
    inside = (uv[:, 0] >= 3) & (uv[:, 0] < width - 4) & (uv[:, 1] >= 3) & (uv[:, 1] < height - 4)
    uv, points = uv[inside], points[inside]
    patch = np.stack(
        [bilinear_sample(depth, uv + [dx, dy]) for dy in (-2, 0, 2) for dx in (-2, 0, 2)]
    )
    observed = patch[4]
    smooth = (patch.max(axis=0) - patch.min(axis=0)) < 0.025 * observed
    valid = np.isfinite(patch).all(axis=0) & (observed > 1) & (observed < 150) & smooth
    points = points[valid]
    return np.stack([observed[valid], points[:, 2], np.linalg.norm(points, axis=1)], axis=1)


def comparison(samples):
    if len(samples) == 0:
        return {"count": 0, "hypotheses": {}}
    observed = samples[:, 0]
    result = {"count": len(samples), "hypotheses": {}}
    for column, name in [(1, "optical_z"), (2, "ray_range")]:
        expected = samples[:, column]
        error = np.abs(observed - expected)
        relative = error / expected
        # 只报告诊断拟合值；不把它偷偷写入标签缩放或内参。
        fitted_scale = np.median(expected / observed)
        result["hypotheses"][name] = {
            "median_absolute_error_m": float(np.median(error)),
            "median_relative_error": float(np.median(relative)),
            "p90_relative_error": float(np.quantile(relative, 0.9)),
            "fraction_within_2_percent": float(np.mean(relative < 0.02)),
            "diagnostic_fitted_scale_not_applied": float(fitted_scale),
            "fitted_scale_median_relative_error": float(
                np.median(np.abs(fitted_scale * observed - expected) / expected)
            ),
        }
    return result


def audit(root, sequence):
    directory = Path(root) / sequence
    calibration_path = directory / "calibration.json"
    index_path = directory / "sync_index.json"
    calibration = json.loads(calibration_path.read_text())
    index = json.loads(index_path.read_text())
    radar_cal, depth_cal = calibration["radar"], calibration["depth"]
    collected = {"manifest_K": [], "pixel_center_minus_half": []}
    per_frame = []
    counts = {
        "paired_frames": 0,
        "radar_returns": 0,
        "zero_direction_returns": 0,
        "frames_without_recoverable_direction": 0,
        "recoverable_returns": 0,
        "direction_within_declared_fov": 0,
        "raw_polar_columns_within_declared_fov": 0,
    }
    extrema, saturated = [], []
    half_h, half_v = np.radians([radar_cal["fov_h"] / 2, radar_cal["fov_v"] / 2])
    for frame_id, files in sorted(index.items(), key=lambda item: int(item[0])):
        if not {"depth", "radar"} <= files.keys():
            continue
        records = np.load(directory / files["radar"], allow_pickle=False)
        depth = np.load(directory / files["depth"], allow_pickle=False)
        counts["paired_frames"] += 1
        counts["radar_returns"] += len(records)
        counts["zero_direction_returns"] += int((records[:, :3] == 0).all(axis=1).sum())
        counts["raw_polar_columns_within_declared_fov"] += int(
            ((np.abs(records[:, 1]) <= half_h) & (np.abs(records[:, 2]) <= half_v)).sum()
        )
        finite_depth = depth[np.isfinite(depth)]
        if finite_depth.size:
            extrema.append([float(finite_depth.min()), float(finite_depth.max())])
            saturated.append(float(np.mean(finite_depth >= 1000)))
        points, _ = recover_radar_points(records, radar_cal["range_m"])
        counts["recoverable_returns"] += len(points)
        if len(points) == 0:
            counts["frames_without_recoverable_direction"] += 1
            continue
        azimuth = np.arctan2(points[:, 1], points[:, 0])
        altitude = np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1))
        counts["direction_within_declared_fov"] += int(
            ((np.abs(azimuth) <= half_h) & (np.abs(altitude) <= half_v)).sum()
        )
        camera_points = radar_to_camera(points, radar_cal["extrinsic"], depth_cal["extrinsic"])
        frame = {"frame_id": int(frame_id)}
        for name, offset in [("manifest_K", 0.0), ("pixel_center_minus_half", -0.5)]:
            samples = matched_samples(depth, camera_points, depth_cal["intrinsic"], offset)
            collected[name].append(samples)
            frame[name] = comparison(samples)
        per_frame.append(frame)
    samples_by_convention = {
        name: np.concatenate(values) if values else np.empty((0, 3))
        for name, values in collected.items()
    }
    comparisons = {name: comparison(samples) for name, samples in samples_by_convention.items()}
    angular = []
    samples = samples_by_convention["manifest_K"]
    angles = np.degrees(np.arccos(np.clip(samples[:, 1] / samples[:, 2], -1, 1)))
    for lower, upper in [(0, 10), (10, 20), (20, 30), (30, 45), (45, 90)]:
        subset = samples[(angles >= lower) & (angles < upper)]
        angular.append({"angle_degrees": [lower, upper], **comparison(subset)})
    return {
        "sequence": sequence,
        "model_predictions_used": False,
        "raw_data_modified": False,
        "radar_format_assumption": "first_three_parallel_to_ray_last_column_range_m",
        "radar_format_confirmed_by_capture_script": False,
        "capture_decode_formula_verified": False,
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        "counts": counts,
        "depth_min": min((v[0] for v in extrema), default=None),
        "depth_max": max((v[1] for v in extrema), default=None),
        "mean_fraction_depth_at_least_1000": float(np.mean(saturated)) if saturated else None,
        "comparisons": comparisons,
        "by_camera_ray_angle": angular,
        "per_frame": per_frame,
        "limitations": [
            "这是传感器数据合同检查，不是模型精度评测或独立验证集。",
            "前三列为零的雷达返回无法恢复方向，不参加逐像素对照。",
            "半像素偏移仅用于敏感性检查，没有修改训练用 K。",
            "无法仅凭 float32 NPY 唯一反推出原 RGB24 解码代码。",
            "没有按匹配误差删除离群点，雷达碰撞几何与相机可见表面可能不同。",
            "诊断拟合比例不自动应用；保留 depth_scale=1，饱和值 >=1000 不监督。",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--sequence", default="captured_data (1)")
    parser.add_argument("--output", default="data/carla_wide4/depth_audit.json")
    args = parser.parse_args()
    report = audit(args.root, args.sequence)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("counts", "comparisons")}, indent=2))


if __name__ == "__main__":
    main()
