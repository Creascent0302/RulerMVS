"""将作者发布的 GTAV_540 转为公共四视图接口，保留未经公制认证的场景单位。

只读取 JSON 配置。原始数据不改动；EXR 转 float32 NPY，标定直接来自逐帧 JSON。
所有视图选择只用相机，不用深度；原数据不是同步四相机采集。
"""

import argparse
import json
import os
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
from PIL import Image


CV_FROM_RIG = np.array([[0.0, -1, 0, 0], [0, 0, -1, 0],
                        [1, 0, 0, 0], [0, 0, 0, 1]])


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_scene(folder, output, maximum_depth):
    frames = sorted(path.stem for path in (folder / "images").glob("*.png"))
    if not frames or frames != sorted(path.stem for path in (folder / "poses").glob("*.json")):
        raise ValueError(f"RGB 与位姿不一一对应：{folder}")
    if frames != sorted(path.stem for path in (folder / "depths").glob("*.exr")):
        raise ValueError(f"RGB 与深度不一一对应：{folder}")
    intrinsics, extrinsics, statistics = [], [], []
    output.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        with Image.open(folder / "images" / (frame + ".png")) as image:
            image.load()
            if image.mode != "RGB" or image.size != (960, 540):
                raise ValueError(f"不是原始 GTAV_540 RGB：{folder}/{frame}")
        camera = json.loads((folder / "poses" / (frame + ".json")).read_text())
        k = np.array([[camera["f_x"], 0, camera["c_x"]],
                      [0, camera["f_y"], camera["c_y"]], [0, 0, 1]], dtype=np.float64)
        e = np.asarray(camera["extrinsic"], dtype=np.float64)
        if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
            raise ValueError(f"无效内参：{folder}/{frame}")
        if (e.shape != (4, 4) or not np.isfinite(e).all()
                or not np.allclose(e[3], [0, 0, 0, 1], atol=1e-8)
                or not np.allclose(e[:3, :3] @ e[:3, :3].T, np.eye(3), atol=2e-5)):
            raise ValueError(f"无效外参：{folder}/{frame}")
        depth = cv2.imread(str(folder / "depths" / (frame + ".exr")), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.shape != (540, 960) or depth.dtype != np.float32:
            raise ValueError(f"EXR 解码格式不符：{folder}/{frame}")
        finite = np.isfinite(depth)
        valid = finite & (depth > 0) & (depth < maximum_depth)
        target = output / (frame + ".npy")
        if target.exists():
            old = np.load(target, allow_pickle=False, mmap_mode="r")
            if old.dtype != depth.dtype or not np.array_equal(old, depth, equal_nan=True):
                raise ValueError(f"已有转换与原始 EXR 不一致：{target}")
        else:
            temporary = target.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, depth, allow_pickle=False)
            temporary.replace(target)
        statistics.append({"frame": frame, "valid_fraction": float(valid.mean()),
                           "infinite_pixels": int(np.isinf(depth).sum()),
                           "nan_pixels": int(np.isnan(depth).sum()),
                           "nonpositive_pixels": int((finite & (depth <= 0)).sum()),
                           "positive_depth_quantiles_scene_unit": np.quantile(depth[valid], [0, .5, .95, 1]).tolist() if valid.any() else None,
                           "original_rotation_determinant": float(np.linalg.det(e[:3, :3])),
                           "original_rotation_orthogonality_error": float(np.abs(e[:3, :3] @ e[:3, :3].T - np.eye(3)).max())})
        intrinsics.append(k)
        extrinsics.append(e)
    extrinsics = np.stack(extrinsics)
    determinants = np.linalg.det(extrinsics[:, :3, :3])
    reflection = np.eye(4)
    if (determinants < 0).all():
        # 统一反射世界坐标基；不翻图，不改变相对外参 E_i @ inv(E_j)。
        reflection[2, 2] = -1
        extrinsics = extrinsics @ reflection
    elif not (determinants > 0).all():
        raise ValueError(f"同场景混用不同手性外参：{folder}")
    return frames, np.stack(intrinsics), extrinsics, statistics, reflection


def choose_groups(intrinsics, extrinsics, settings):
    poses = np.linalg.inv(extrinsics)
    centers = poses[:, :3, 3]
    step_lengths = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    positive = step_lengths[step_lengths > 1e-8]
    if not len(positive):
        return [], [{"reference_index": i, "reason": "pure_rotation"} for i in range(len(poses))], None
    spacing = float(np.median(positive))
    u, v = np.meshgrid(np.linspace(95, 864, 8), np.linspace(53, 486, 6))
    pixels = np.stack((u, v, np.ones_like(u)), -1).reshape(-1, 3)
    probe_depths = spacing * np.asarray(settings["probe_depth_in_pose_steps"], dtype=float)
    groups, rejected = [], []
    for reference in range(len(poses)):
        ids = np.arange(max(0, reference - settings["frame_window"]),
                        min(len(poses), reference + settings["frame_window"] + 1))
        ids = ids[ids != reference]
        relative = extrinsics[ids] @ poses[reference]
        baselines = np.linalg.norm(relative[:, :3, 3], axis=1)
        angles = np.arccos(np.clip((np.trace(relative[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1))
        rays = pixels @ np.linalg.inv(intrinsics[reference]).T
        probes = (probe_depths[:, None, None] * rays[None]).reshape(-1, 3)
        camera = np.einsum("cij,pj->cpi", relative[:, :3, :3], probes) + relative[:, None, :3, 3]
        projected = np.einsum("cij,cpj->cpi", intrinsics[ids], camera)
        uv = projected[..., :2] / np.maximum(camera[..., 2:3], 1e-8)
        inside = ((camera[..., 2] > 0) & (uv[..., 0] >= 0) & (uv[..., 0] <= 959)
                  & (uv[..., 1] >= 0) & (uv[..., 1] <= 539))
        overlap = inside.mean(axis=1)
        minimum = spacing * settings["minimum_baseline_in_pose_steps"]
        eligible = ((baselines >= minimum) & (baselines <= spacing * settings["maximum_baseline_in_pose_steps"])
                    & (np.degrees(angles) <= settings["maximum_angle_degrees"])
                    & (overlap >= settings["minimum_probe_overlap"]))
        candidates = np.flatnonzero(eligible)
        preferred = spacing * settings["preferred_baseline_in_pose_steps"]
        costs = np.abs(np.log(np.maximum(baselines, 1e-8) / preferred)) + 2 * angles + 1 - overlap
        # 同分时固定按帧号决策，避免浮点舍入让单位换算改变对称候选的顺序。
        ordered = sorted(candidates, key=lambda index: (round(float(costs[index]), 12), ids[index]))
        chosen = []
        for index in ordered:
            frame = int(ids[index])
            if any(np.linalg.norm(centers[frame] - centers[source["index"]]) < minimum for source in chosen):
                continue
            chosen.append({"index": frame, "baseline_scene_unit": float(baselines[index]),
                           "angle_degrees": float(np.degrees(angles[index])),
                           "probe_overlap": float(overlap[index])})
            if len(chosen) == 3:
                break
        if len(chosen) == 3:
            groups.append({"reference_index": reference, "sources": chosen})
        else:
            rejected.append({"reference_index": reference, "eligible_sources": len(ordered),
                             "separated_sources": len(chosen)})
    return groups, rejected, spacing


def record_for_group(root, folder, converted, scene, frames, k, e, group, slot, maximum_depth):
    reference = group["reference_index"]
    ordered = [source["index"] for source in group["sources"]]
    ordered.insert(slot, reference)
    world_from_rig = np.linalg.inv(e[reference]) @ CV_FROM_RIG
    tuple_id = "reference_" + frames[reference]
    record = {"sequence": "mvssynth_" + scene, "frame_id": tuple_id, "timestamp": None,
              "sync_basis": "frame_id", "reference_slot": slot, "geometry_units": "scene_unit",
              "sample_type": "static_multiview_tuple", "synchronized_capture": False,
              "source_selection_uses_ground_truth": False, "cameras": []}
    for view_slot, index in enumerate(ordered):
        frame = frames[index]
        record["cameras"].append({
            "camera_id": f"view_{view_slot}", "role": "wide", "frame_id": tuple_id,
            "timestamp": None, "source_frame_id": frame, "source_camera_id": "original_camera",
            "rgb": (folder / "images" / (frame + ".png")).relative_to(root).as_posix(),
            "K": k[index].tolist(), "T_cam_from_rig": (e[index] @ world_from_rig).tolist(),
            "targets": {"frame_id": tuple_id, "depth_type": "optical_z", "depth_units": "scene_unit",
                        "depth_encoding": "scene_unit_npy", "depth_scale": 1.0,
                        "max_valid_depth": maximum_depth,
                        "depth": (converted / (frame + ".npy")).relative_to(root).as_posix()}})
    return record


def diagnose(converted, frames, k, e, groups, maximum_depth, samples):
    reports = []
    if not groups:
        return reports
    v, u = np.mgrid[0:540:8, 0:960:8]
    pixels = np.stack((u, v, np.ones_like(u)), -1).reshape(-1, 3)
    for index in np.unique(np.linspace(0, len(groups) - 1, min(samples, len(groups))).astype(int)):
        group = groups[index]
        reference = group["reference_index"]
        z = np.load(converted / (frames[reference] + ".npy"), allow_pickle=False)[v, u].ravel()
        good = np.isfinite(z) & (z > 0) & (z < maximum_depth)
        points = (pixels @ np.linalg.inv(k[reference]).T)[good] * z[good, None]
        for source in group["sources"]:
            frame = source["index"]
            depth = np.load(converted / (frames[frame] + ".npy"), allow_pickle=False)
            relative = e[frame] @ np.linalg.inv(e[reference])
            camera = points @ relative[:3, :3].T + relative[:3, 3]
            projected = camera @ k[frame].T
            uv = projected[:, :2] / np.maximum(camera[:, 2:3], 1e-8)
            inside = ((camera[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= 959)
                      & (uv[:, 1] >= 0) & (uv[:, 1] <= 539))
            xy = np.rint(uv[inside]).astype(int)
            measured = depth[xy[:, 1], xy[:, 0]]
            valid = np.isfinite(measured) & (measured > 0) & (measured < maximum_depth)
            measured = measured[valid]
            error = camera[inside, 2][valid] - measured
            relative_error = np.abs(error) / np.maximum(measured, 1e-8)
            reports.append({"reference": frames[reference], "source": frames[frame],
                            "reference_valid_probes": int(good.sum()), "projected_inside": int(inside.sum()),
                            "valid_pairs": int(valid.sum()),
                            "median_absolute_error_scene_unit": float(np.median(np.abs(error))) if len(error) else None,
                            "median_relative_error": float(np.median(relative_error)) if len(error) else None,
                            "agreement_within_two_percent": float((relative_error <= .02).mean()) if len(error) else None,
                            "behind_source_surface_fraction": float((error > .02 * measured).mean()) if len(error) else None,
                            "in_front_of_source_surface_fraction": float((error < -.02 * measured).mean()) if len(error) else None})
    return reports


def prepare(config):
    root = Path(config["root"]).resolve()
    raw = root / config.get("raw_directory", "raw/GTAV_540")
    output = root / config.get("output_directory", "prepared")
    if config.get("geometry_units") != "scene_unit":
        raise ValueError("未经公制认证的数据必须显式配置 geometry_units=scene_unit")
    scenes = sorted(path.name for path in raw.iterdir() if path.is_dir() and path.name.isdigit())
    expected_counts = json.loads((raw / "num_images.json").read_text())
    if scenes != [f"{index:04d}" for index in range(len(expected_counts))]:
        raise ValueError("场景目录与作者 num_images.json 不一致")
    split_sizes = config["split_scene_counts"]
    if set(split_sizes) != {"train", "val", "test"} or sum(split_sizes.values()) != len(scenes):
        raise ValueError("按场景划分的数量必须完整覆盖下载场景")
    order = np.random.default_rng(int(config["split_seed"])).permutation(scenes).tolist()
    settings = config["selection"]
    maximum_depth = float(config.get("maximum_valid_depth_scene_unit", 1e9))
    slot = int(config.get("reference_slot", 2))
    if slot not in range(4) or not np.isfinite(maximum_depth) or maximum_depth <= 0:
        raise ValueError("无效参考索引或深度上限")
    report = {"dataset": "MVS-Synth GTAV_540", "config": config, "geometry_units": "scene_unit",
              "metric_scale_verified": False, "metric_evaluation_allowed": False,
              "image_size_hw": [540, 960], "depth_conversion": "half EXR解码float32；不缩放，不取倒数，保留inf",
              "intrinsics": "直接使用逐帧JSON原K；作者DeepMVS以整数像素中心投影，不采用MVSAnywhere loader的810宽补偿",
              "pose": "原world-to-camera；统一世界坐标反射纠正det=-1，所有相对投影不变",
              "source_selection": "只用标定；探针和基线按同场景相机位移中位数定义，未使用GT估范围",
              "supervision_mask": "所有有限正深度；不按GT百分位删除最远5%，不将inf当free",
              "evaluation_warning": "本地整场景隔离不消除MVSAnywhere原训练暴露；本数据不能参与米制CARLA指标混评",
              "synchronized_capture": False, "splits": {}, "scenes": {}}
    offset = 0
    for split, count in split_sizes.items():
        assigned = sorted(order[offset:offset + count])
        offset += count
        path = output / "manifests" / (split + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".jsonl.tmp")
        samples = 0
        with temporary.open("w", encoding="utf-8") as stream:
            for scene in assigned:
                folder = raw / scene
                converted = output / "depth_scene_unit" / scene
                frames, k, e, statistics, reflection = load_scene(folder, converted, maximum_depth)
                if frames != [f"{index:04d}" for index in range(expected_counts[int(scene)])]:
                    raise ValueError(f"帧号与作者 num_images.json 不一致：{scene}")
                groups, rejected, spacing = choose_groups(k, e, settings)
                for group in groups:
                    record = record_for_group(root, folder, converted, scene, frames, k, e, group, slot, maximum_depth)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                audit = {"scene": scene, "split": split, "frames": len(frames), "tuples": len(groups),
                         "rejected_references": rejected, "pose_step_scene_unit": spacing,
                         "world_reflection": reflection.tolist(), "frame_audit": statistics,
                         "groups": groups, "geometry_diagnostics": diagnose(converted, frames, k, e, groups, maximum_depth,
                                                                             int(config.get("geometry_samples_per_scene", 3)))}
                save_json(output / "audit" / (scene + ".json"), audit)
                report["scenes"][scene] = {"split": split, "frames": len(frames), "tuples": len(groups),
                                          "rejected_references": len(rejected), "pose_step_scene_unit": spacing}
                samples += len(groups)
                print(f"{split}/{scene}: {len(frames)}帧，{len(groups)}组，拒绝{len(rejected)}参考", flush=True)
        if samples == 0:
            raise ValueError(f"{split}没有可用四视图，保留审计而不伪造样本")
        temporary.replace(path)
        report["splits"][split] = {"scenes": assigned, "samples": samples, "manifest": str(path)}
    save_json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(json.loads(Path(args.config).read_text())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
