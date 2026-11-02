"""转换官方 BlendedMVS 基础集；保留原生场景尺度，不声称公制标定。

所有设置来自 --config JSON。原 RGB/PFM/cam 文件只读，输出写入数据盘根目录的
processed 和 manifests。四视图由官方 pair.txt 与非零基线选择，不读取深度挑源。
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_pfm(path):
    """PFM按文件字节序读取，行方向由底到顶翻回图像方向；不猜测米制尺度。"""
    with path.open("rb") as stream:
        if stream.readline().strip() != b"Pf":
            raise ValueError(f"深度必须是单通道Pf：{path}")
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        if abs(scale) != 1.0:
            raise ValueError(f"非单位PFM标度尚未支持，不能静默改变原标定：{path}, {scale}")
        values = np.fromfile(stream, dtype="<f4" if scale < 0 else ">f4")
    if values.size != width * height:
        raise ValueError(f"PFM数据长度错误：{path}")
    return np.flipud(values.reshape(height, width)).astype(np.float32)


def read_camera(path, principal_point_offset):
    lines = path.read_text().splitlines()
    if lines[0].strip() != "extrinsic" or lines[6].strip() != "intrinsic":
        raise ValueError(f"未知cam文件布局：{path}")
    extrinsic = np.fromstring(" ".join(lines[1:5]), sep=" ").reshape(4, 4)
    intrinsic = np.fromstring(" ".join(lines[7:10]), sep=" ").reshape(3, 3)
    if not np.isfinite(extrinsic).all() or not np.isfinite(intrinsic).all():
        raise ValueError(f"非有限标定：{path}")
    if not np.allclose(extrinsic[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError(f"非齐次外参：{path}")
    rotation = extrinsic[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-4
    ):
        raise ValueError(f"相机外参不是正常旋转：{path}")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or not np.allclose(intrinsic[2], [0, 0, 1]):
        raise ValueError(f"针孔内参无效：{path}")
    intrinsic[:2, 2] += principal_point_offset
    # cam末行深度搜索范围可能来自数据重建，故不把它加入推理输入。
    return intrinsic, np.linalg.inv(extrinsic)


def read_pairs(path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    count = int(lines[0])
    if len(lines) != 1 + 2 * count:
        raise ValueError(f"pair.txt行数不匹配：{path}")
    pairs = {}
    for index in range(count):
        reference = int(lines[1 + 2 * index])
        fields = lines[2 + 2 * index].split()
        if len(fields) != 1 + 2 * int(fields[0]):
            raise ValueError(f"pair.txt源数量不匹配：{path}, {reference}")
        if reference in pairs:
            raise ValueError(f"pair.txt重复参考：{path}, {reference}")
        sources = [(int(fields[i]), float(fields[i + 1])) for i in range(1, len(fields), 2)]
        if not all(np.isfinite(score) for _, score in sources):
            raise ValueError(f"pair.txt含非有限分数：{path}")
        pairs[reference] = sources
    return pairs


def choose_groups(pairs, poses):
    groups, skipped = [], []
    for reference, sources in pairs.items():
        if reference not in poses:
            raise ValueError(f"pair参考不在完整相机集合中：{reference}")
        selected = []
        centers = [poses[reference][:3, 3]]
        for source, score in sources:
            if source not in poses:
                raise ValueError(f"pair源不在完整相机集合中：{source}")
            center = poses[source][:3, 3]
            if any(np.linalg.norm(center - previous) <= 1e-6 for previous in centers):
                continue
            selected.append({"frame_id": source, "official_pair_score": score})
            centers.append(center)
            if len(selected) == 3:
                break
        if len(selected) == 3:
            groups.append({"reference_frame_id": reference, "sources": selected})
        else:
            skipped.append({"frame_id": reference, "available_distinct_sources": len(selected)})
    return groups, skipped


def convert_scene(root, directory, principal_point_offset, minimum_free_gib=0):
    scene = directory.name
    cameras = {int(path.name.split("_")[0]): path for path in (directory / "cams").glob("*_cam.txt")}
    images = {int(path.stem): path for path in (directory / "blended_images").glob("*.jpg") if path.stem.isdigit()}
    depths = {int(path.stem): path for path in (directory / "rendered_depth_maps").glob("*.pfm") if path.stem.isdigit()}
    if not cameras or set(cameras) != set(images) or set(cameras) != set(depths):
        raise ValueError(f"{scene} RGB、PFM和标定帧集合不完全对应")
    output = root / "processed" / scene
    (output / "depth").mkdir(parents=True, exist_ok=True)
    poses, intrinsics, paths, statistics = {}, {}, {}, []
    for frame in sorted(cameras):
        if shutil.disk_usage(root).free < minimum_free_gib * 1024 ** 3 + depths[frame].stat().st_size:
            raise RuntimeError("剩余空间将低于配置保留量，停止转换并保留已完成数据")
        intrinsic, pose = read_camera(cameras[frame], principal_point_offset)
        with Image.open(images[frame]) as image:
            image.load()
            if image.mode != "RGB":
                raise ValueError(f"RGB图像模式错误：{images[frame]}")
            image_size = image.size[::-1]
        depth = read_pfm(depths[frame])
        if depth.shape != image_size:
            raise ValueError(f"RGB与深度尺寸不一致：{scene}/{frame}")
        valid = np.isfinite(depth) & (depth > 0)
        depth[~valid] = np.nan
        target = output / "depth" / f"{frame:08d}.optical_z.npy"
        temporary = target.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, depth, allow_pickle=False)
        temporary.replace(target)
        poses[frame], intrinsics[frame] = pose, intrinsic
        paths[frame] = {"rgb": images[frame].relative_to(root).as_posix(),
                        "depth": target.relative_to(root).as_posix()}
        statistics.append({"frame_id": frame, "image_size": list(image_size),
                           "valid_depth_fraction": float(valid.mean()),
                           "minimum_native_depth": float(depth[valid].min()) if valid.any() else None,
                           "maximum_native_depth": float(depth[valid].max()) if valid.any() else None})
    (output / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in statistics))
    groups, skipped = choose_groups(read_pairs(directory / "cams" / "pair.txt"), poses)
    write_json(output / "tuple_selection.json", groups)
    write_json(output / "unmatched_references.json", skipped)
    return poses, intrinsics, paths, groups, statistics, skipped


def make_record(scene, split, group, paths, poses, intrinsics, reference_slot):
    reference = group["reference_frame_id"]
    ordered = [source["frame_id"] for source in group["sources"]]
    ordered.insert(reference_slot, reference)
    cv_from_rig = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    world_from_rig = poses[reference] @ cv_from_rig
    frame_id = f"reference_{reference:08d}"
    record = {"sequence": f"blendedmvs_{scene}", "frame_id": frame_id, "timestamp": None,
              "sync_basis": "frame_id", "sample_type": "static_multiview_tuple", "synchronized_capture": False,
              "reference_slot": reference_slot, "source_split": split, "source_scene": scene,
              "reference_source_frame_id": reference, "selection_uses_ground_truth_depth": False,
              "geometry_units": "scene_unit", "metric_scale_known": False, "cameras": []}
    for slot, frame in enumerate(ordered):
        record["cameras"].append({
            "camera_id": f"view_{slot}", "role": "wide", "frame_id": frame_id, "timestamp": None,
            "source_frame_id": frame, "source_camera_id": f"{scene}/{frame:08d}",
            "rgb": paths[frame]["rgb"], "K": intrinsics[frame].tolist(),
            "T_cam_from_rig": (np.linalg.inv(poses[frame]) @ world_from_rig).tolist(),
            "targets": {"frame_id": frame_id, "depth": paths[frame]["depth"], "depth_type": "optical_z",
                        "depth_encoding": "scene_unit_npy", "depth_scale": 1.0,
                        "depth_units": "scene_unit", "max_valid_depth": 1e30},
        })
    return record


def audit_geometry(root, groups, paths, poses, intrinsics, count):
    """事后GT核查，不修改输入组；误差单位为原生场景单位。"""
    reports = []
    if not groups or not count:
        return reports
    selected = np.unique(np.linspace(0, len(groups) - 1, min(count, len(groups))).astype(int))
    for index in selected:
        group = groups[index]
        reference = group["reference_frame_id"]
        depth = np.load(root / paths[reference]["depth"], allow_pickle=False)
        height, width = depth.shape
        v, u = np.mgrid[0:height:12, 0:width:12]
        z = depth[v, u].ravel()
        pixels = np.stack((u, v, np.ones_like(u)), -1).reshape(-1, 3)
        good = np.isfinite(z) & (z > 0)
        points = (pixels @ np.linalg.inv(intrinsics[reference]).T)[good] * z[good, None]
        for source in group["sources"]:
            frame = source["frame_id"]
            transform = np.linalg.inv(poses[frame]) @ poses[reference]
            camera = points @ transform[:3, :3].T + transform[:3, 3]
            projected = camera @ intrinsics[frame].T
            uv = projected[:, :2] / np.maximum(camera[:, 2:3], 1e-10)
            other = np.load(root / paths[frame]["depth"], allow_pickle=False)
            inside = ((camera[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= other.shape[1] - 1)
                      & (uv[:, 1] >= 0) & (uv[:, 1] <= other.shape[0] - 1))
            xy = np.rint(uv[inside]).astype(int)
            sampled = other[xy[:, 1], xy[:, 0]]
            valid = np.isfinite(sampled) & (sampled > 0)
            error = camera[inside, 2][valid] - sampled[valid]
            relative = np.abs(error) / np.maximum(sampled[valid], 1e-10)
            reports.append({"reference_frame_id": reference, "source_frame_id": frame,
                            "source_valid_probes": int(valid.sum()),
                            "median_absolute_error_native_units": float(np.median(np.abs(error))) if len(error) else None,
                            "median_relative_error": float(np.median(relative)) if len(error) else None,
                            "fraction_within_one_percent": float((relative <= 0.01).mean()) if len(error) else None})
    return reports


def prepare(config):
    root = Path(config["root"]).resolve()
    raw = root / config.get("raw_directory", "raw/BlendedMVS")
    reference_slot = int(config.get("reference_slot", 2))
    if not 0 <= reference_slot <= 3:
        raise ValueError("reference_slot 必须为0–3")
    train = set(Path(config["official_training_list"]).read_text().split())
    test = set(Path(config["official_validation_list"]).read_text().split())
    scenes = {path.parent.parent.name: path.parent.parent for path in raw.glob("*/cams/pair.txt")}
    if train & test or set(scenes) != train | test:
        raise ValueError("原始场景集合与官方106训练/7验证清单不一致")
    fraction = float(config.get("validation_fraction", 0.1))
    if not 0 < fraction < 0.5:
        raise ValueError("validation_fraction必须在0与0.5之间")
    ordered = sorted(train, key=lambda scene: hashlib.sha256((str(config.get("split_seed", 47)) + scene).encode()).hexdigest())
    val = set(ordered[:max(1, int(np.ceil(len(train) * fraction)))])
    train -= val
    scene_splits = {scene: name for name, pool in (("train", train), ("val", val), ("test", test)) for scene in pool}
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    results = {"config": config, "dataset": "official BlendedMVS v1.0.0 low resolution base set",
               "geometry_units": "scene_unit", "metric_scale_known": False,
               "depth_scale_applied": 1.0, "pfm_vertical_flip": True,
               "camera_convention": "cam文件为CV世界到相机外参；参考局部rig为x前/y左/z上",
               "pixel_center_convention": "整数中心；默认保留MVSFormer++原始K，不静默缩放",
               "scene_splits": {name: sorted(pool) for name, pool in (("train", train), ("val", val), ("test", test))},
               "test_definition": "官方7个validation场景在本项目中保留为test；额外val只从官方train按整场景留出",
               "limitations": ["原生场景尺度不保证米制，不能与CARLA米制MAE/RMSE混比。",
                               "不是同步物理四目，部分数据覆盖建筑/物体而非驾驶场景。"],
               "scenes": {}, "totals": {}}
    rows = {name: [] for name in ("train", "val", "test")}
    labeled = []
    for scene in sorted(scenes):
        poses, intrinsic, paths, groups, statistics, skipped = convert_scene(
            root, scenes[scene], float(config.get("principal_point_offset", 0.0)),
            float(config.get("minimum_free_gib", 0)))
        split = scene_splits[scene]
        valid = {row["frame_id"]: row["valid_depth_fraction"] for row in statistics}
        for group in groups:
            record = make_record(scene, split, group, paths, poses, intrinsic, reference_slot)
            rows[split].append(record)
            if split == "train" and valid[group["reference_frame_id"]] > 0:
                labeled.append(record)
        diagnostic = audit_geometry(root, groups, paths, poses, intrinsic, int(config.get("geometry_groups_per_scene", 1)))
        write_json(root / "processed" / scene / "geometry_audit.json", diagnostic)
        results["scenes"][scene] = {"split": split, "frames_checked_and_converted": len(paths),
                                    "tuples": len(groups), "unmatched_references": len(skipped),
                                    "zero_valid_depth_frames": sum(row["valid_depth_fraction"] == 0 for row in statistics)}
        print(f"{scene}: {split}, {len(paths)}帧, {len(groups)}个四视角组", flush=True)
    for name, records in rows.items():
        (manifests / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        results["totals"][name] = {"scenes": len(results["scene_splits"][name]), "tuples": len(records)}
    (manifests / "train_labeled.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labeled))
    results["totals"]["train_labeled_tuples"] = len(labeled)
    results["labeled_rule"] = "仅从既有train组排除完全无有效参考深度的组；val/test不筛选且源图不重选"
    write_json(root / "processed" / "preparation_summary.json", results)
    print("BlendedMVS原生单位数据接口转换完成；未把尺度强行解释为米", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    prepare(json.loads(Path(args.config).read_text()))
