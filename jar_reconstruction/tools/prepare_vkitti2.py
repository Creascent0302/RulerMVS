"""转换 Virtual KITTI 2.0.3；同步双目与动态跨时多视图协议严格分开。

双目 JSONL 可由 CarlaFrameDataset 读取；公共训练/推理入口须显式设置
num_views=2。跨时四帧只写 MVSAnywhere 原生 tuple 文本与独立元数据，
不伪装成同步 CARLA 帧。配置文件决定路径、Scene 级划分和处理线程数。
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


CV_FROM_RIG = np.array([
    [0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1],
], dtype=np.float64)
CATEGORIES = [
    "Terrain", "Sky", "Tree", "Vegetation", "Building", "Road", "GuardRail",
    "TrafficSign", "TrafficLight", "Pole", "Misc", "Truck", "Car", "Van",
]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def save_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = np.load(path, mmap_mode="r", allow_pickle=False)
        if existing.dtype != array.dtype or not np.array_equal(existing, array):
            raise ValueError(f"已有转换结果与原始数据不一致，保留文件并停止：{path}")
        return
    temporary = path.with_suffix(".npy.partial")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def load_calibration(folder):
    intrinsics = np.loadtxt(folder / "intrinsic.txt", skiprows=1)
    extrinsics = np.loadtxt(folder / "extrinsic.txt", skiprows=1)
    if intrinsics.shape[1] != 6 or extrinsics.shape[1] != 18:
        raise ValueError(f"标定列数不符合 Virtual KITTI 2.0.3：{folder}")
    if not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all():
        raise ValueError(f"标定中存在 NaN/Inf：{folder}")
    matrices = {}
    for row in extrinsics:
        key = tuple(row[:2].astype(int))
        if key in matrices:
            raise ValueError(f"重复外参：{folder} {key}")
        matrix = row[2:].reshape(4, 4)
        rotation = matrix[:3, :3]
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7):
            raise ValueError(f"外参不是齐次矩阵：{folder} {key}")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2e-6):
            raise ValueError(f"旋转不正交：{folder} {key}")
        if not np.isclose(np.linalg.det(rotation), 1, atol=2e-6):
            raise ValueError(f"旋转方向异常：{folder} {key}")
        matrices[key] = matrix
    cameras = {}
    for row in intrinsics:
        frame, camera = row[:2].astype(int)
        key = (frame, camera)
        if camera not in (0, 1) or key not in matrices or key in cameras:
            raise ValueError(f"相机或内外参索引异常：{folder} {key}")
        fx, fy, cx, cy = row[2:]
        if min(fx, fy) <= 0:
            raise ValueError("相机焦距必须为正")
        cameras[key] = (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), matrices[key])
    if set(cameras) != set(matrices):
        raise ValueError(f"内外参索引不一致：{folder}")
    frames = sorted({frame for frame, _ in cameras})
    if set(cameras) != {(frame, camera) for frame in frames for camera in (0, 1)}:
        raise ValueError(f"标定没有完整同步双目：{folder}")
    return cameras, frames


def colors(folder):
    result = {}
    for line in (folder / "colors.txt").read_text().splitlines()[1:]:
        name, red, green, blue = line.split()
        code = (int(red) << 16) | (int(green) << 8) | int(blue)
        result[code] = -1 if name == "Undefined" else CATEGORIES.index(name)
    return result


def convert_camera(root, sequence, frame, camera, palette):
    folder = root / "raw" / sequence / "frames"
    suffix = f"{frame:05d}"
    image_path = folder / "rgb" / f"Camera_{camera}" / f"rgb_{suffix}.jpg"
    depth_path = folder / "depth" / f"Camera_{camera}" / f"depth_{suffix}.png"
    class_path = folder / "classSegmentation" / f"Camera_{camera}" / f"classgt_{suffix}.png"
    with Image.open(image_path) as reader:
        reader.load()
        if reader.mode != "RGB" or reader.size != (1242, 375):
            raise ValueError(f"RGB格式或尺寸不符合官方定义：{image_path}")
    with Image.open(depth_path) as reader:
        depth = np.asarray(reader)
    with Image.open(class_path) as reader:
        labels = np.asarray(reader.convert("RGB"), dtype=np.uint32)
    if depth.shape != (375, 1242) or labels.shape != (375, 1242, 3):
        raise ValueError(f"RGB/depth/semantic尺寸不一致：{sequence}/{frame}/{camera}")
    if depth.dtype.kind not in "ui" or depth.min() < 0 or depth.max() > 65535:
        raise ValueError(f"深度不是无符号厘米编码：{depth_path}")
    valid = (depth > 0) & (depth < 65535)
    centimeters = np.where(valid, depth, 0).astype(np.uint16)
    code = (labels[..., 0] << 16) | (labels[..., 1] << 8) | labels[..., 2]
    unique, inverse = np.unique(code, return_inverse=True)
    if any(int(value) not in palette for value in unique):
        raise ValueError(f"出现 colors.txt 未定义的 RGB 类别：{class_path}")
    lookup = np.array([palette[int(value)] for value in unique], dtype=np.int8)
    semantic = lookup[inverse].reshape(depth.shape)
    destination = root / "prepared" / sequence / f"Camera_{camera}"
    converted_depth = destination / f"depth_cm_{suffix}.npy"
    converted_semantic = destination / f"semantic_{suffix}.npy"
    save_npy(converted_depth, centimeters)
    save_npy(converted_semantic, semantic)
    return {
        "frame": frame, "camera": camera,
        "rgb": str(image_path.relative_to(root)),
        "depth": str(converted_depth.relative_to(root)),
        "semantic": str(converted_semantic.relative_to(root)),
        "valid_fraction": float(valid.mean()),
        "saturated_fraction": float((depth == 65535).mean()),
    }


def camera_record(source, intrinsics, transform, frame):
    return {
        "name": f"Camera_{source['camera']}", "frame_id": str(frame), "timestamp": None,
        "rgb": source["rgb"], "K": intrinsics.tolist(), "T_cam_from_rig": transform.tolist(),
        "targets": {
            "frame_id": str(frame), "depth": source["depth"], "depth_type": "optical_z",
            "depth_encoding": "metric_npy", "depth_scale": 0.01,
            "max_valid_depth": 655.35, "semantic": source["semantic"],
        },
    }


def synchronous_record(sequence, frame, cameras, records):
    # rig 原点为本时刻左目光心，x前/y左/z上；无需修改官方世界坐标。
    world_from_rig = np.linalg.inv(cameras[(frame, 0)][1]) @ CV_FROM_RIG
    result = []
    for camera in (0, 1):
        intrinsics, extrinsics = cameras[(frame, camera)]
        result.append(camera_record(records[(frame, camera)], intrinsics,
                                    extrinsics @ world_from_rig, frame))
    return {
        "sequence": "VKITTI2/" + sequence, "frame_id": str(frame), "timestamp": None,
        "sync_basis": "frame_id", "protocol": "synchronized_stereo2",
        "synchronized": True, "dynamic_scene": True, "cameras": result,
    }


def temporal_sources(frame, frames, cameras, settings):
    """只按帧邻域及标定选择不同源帧，不读取标签筛选动态物体。"""
    reference_pose = np.linalg.inv(cameras[(frame, 0)][1])
    candidates = []
    for other in frames:
        if other == frame or abs(other - frame) > settings.get("maximum_frame_gap", 4):
            continue
        pose = np.linalg.inv(cameras[(other, 0)][1])
        baseline = np.linalg.norm(pose[:3, 3] - reference_pose[:3, 3])
        direction_cosine = np.dot(pose[:3, 2], reference_pose[:3, 2])
        if not settings.get("min_baseline_m", 0.05) < baseline < settings.get("max_baseline_m", 8.0):
            continue
        if direction_cosine < np.cos(np.deg2rad(settings.get("max_axis_angle_deg", 20.0))):
            continue
        candidates.append((abs(other - frame), baseline, other))
    return [other for _, _, other in sorted(candidates)[:3]]


def geometry_check(root, record):
    """离线GT诊断双目光轴深度/投影；不以结果增删样本。"""
    left, right = record["cameras"]
    depth = np.load(root / left["targets"]["depth"], allow_pickle=False).astype(np.float64) * 0.01
    source = np.load(root / right["targets"]["depth"], allow_pickle=False).astype(np.float64) * 0.01
    y, x = np.mgrid[8:depth.shape[0]:16, 8:depth.shape[1]:16]
    z = depth[y, x]
    rays = np.stack([x, y, np.ones_like(x)], -1) @ np.linalg.inv(left["K"]).T
    points = rays * z[..., None]
    relative = np.array(right["T_cam_from_rig"]) @ np.linalg.inv(left["T_cam_from_rig"])
    transformed = points @ relative[:3, :3].T + relative[:3, 3]
    projected = transformed @ np.asarray(right["K"]).T
    uv = projected[..., :2] / np.maximum(projected[..., 2:], 1e-8)
    px = np.rint(uv[..., 0]).astype(int)
    py = np.rint(uv[..., 1]).astype(int)
    valid = (z > 0) & (transformed[..., 2] > 0) & (px >= 0) & (px < depth.shape[1]) & (py >= 0) & (py < depth.shape[0])
    measured = source[py[valid], px[valid]]
    predicted = transformed[..., 2][valid]
    visible = measured > 0
    error = np.abs(measured[visible] - predicted[visible])
    threshold = np.maximum(0.05, 0.02 * predicted[visible])
    baseline = float(np.linalg.norm(relative[:3, 3]))
    if not np.isclose(baseline, 0.532725, atol=2e-4):
        raise ValueError(f"官方双目基线不符：{baseline}")
    return {
        "frame_id": record["frame_id"], "baseline_m": baseline,
        "projected_valid_samples": int(visible.sum()),
        "within_5cm_or_2percent": float((error <= threshold).mean()) if len(error) else None,
        "median_absolute_depth_error_m": float(np.median(error)) if len(error) else None,
        "note": "包含最近邻采样、遮挡及边缘影响，仅离线诊断，不用来筛选样本。",
    }


def verify_official_checksums(root, download):
    """将社区镜像归档再与官方发布的 MD5 对照；SHA256已由下载器检查。"""
    expected = {}
    for line in (root / "logs" / "official_md5_checksums.txt").read_text().splitlines():
        digest, name = line.split()
        expected[name] = digest
    results = []
    for item in download["files"]:
        path = Path(item["path"])
        digest = hashlib.md5()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected[path.name]:
            raise ValueError(f"与官方MD5不一致：{path.name}")
        results.append({"archive": path.name, "md5": actual, "official_md5_matched": True})
    save_json(root / "logs" / "official_md5_validation.json", results)
    return results


def run(config):
    root = Path(config["root"]).resolve()
    output = root / "prepared"
    assignments = {}
    for split, scenes in config["splits"].items():
        for scene in scenes:
            if scene in assignments:
                raise ValueError(f"同Scene跨split泄露：{scene}")
            assignments[scene] = split
    present = {path.name for path in (root / "raw").glob("Scene*") if path.is_dir()}
    if present != set(assignments):
        raise ValueError(f"Scene配置与原始数据不一致：{present} vs {set(assignments)}")
    download = json.loads((root / "download_evidence.json").read_text())
    if not download.get("completed") or not all(item["source_sha256_matched"] for item in download["files"]):
        raise ValueError("先完成全部下载与源SHA256校验，再运行转换")
    official_checks = verify_official_checksums(root, download)
    image_count = sum(len(np.loadtxt(path, skiprows=1)) for path in (root / "raw").glob("Scene*/*/intrinsic.txt"))
    existing = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    existing_arrays = sum(path.stat().st_size for path in output.rglob("*.npy"))
    projected = existing - existing_arrays + image_count * (375 * 1242 * 3 + 256)
    if projected > config.get("storage_budget_gib", 170) * 1024 ** 3:
        raise ValueError(f"预计占用超出预算：{projected / 1024 ** 3:.2f} GiB")
    summary = {
        "dataset": "Virtual KITTI 2.0.3", "config": config, "sequences": {}, "completed": False,
        "projected_storage_gib": projected / 1024 ** 3,
        "official_archive_checks": official_checks,
        "depth_storage": "uint16厘米NPY，65535及0置0；loader通过depth_scale=0.01转float32米制光轴Z",
        "semantic_ids": {str(index): name for index, name in enumerate(CATEGORIES)},
        "undefined_semantic_id": -1,
        "coordinate_system": "rig原点为同帧左目光心，x前/y左/z上；CV相机x右/y下/z前。T_cam_from_rig。",
        "limitations": [
            "同步双目不是同步四目；公共入口必须显式设置num_views=2，禁止复制图像补齐。",
            "temporal4为动态场景非同步单相机四帧，无运动补偿，不能宣称严格静态MVS一致。",
            "MVSAnywhere官方VKITTI加载器同样只读取Camera_0并取跨时帧；未发现动态补偿。",
            "提供完整原始classSegmentation及int8语义标签，标签不作为推理输入。",
            "所有同Scene的天气/视角变体归同split。原始预训练可能见过这些Scene，不称未见泛化测试。",
        ],
    }
    streams, completed_paths = {}, []
    for split in config["splits"]:
        paths = {
            "stereo": output / "manifests" / "synchronized2" / f"{split}.jsonl",
            "temporal": output / "temporal4" / f"{split}_four_view.txt",
        }
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            streams[(split, name)] = temporary.open("w")
            completed_paths.append((temporary, path))
    try:
        for scene in sorted(assignments):
            split = assignments[scene]
            for folder in sorted((root / "raw" / scene).iterdir()):
                if not folder.is_dir():
                    continue
                sequence = str(folder.relative_to(root / "raw"))
                cameras, frames = load_calibration(folder)
                palette = colors(folder)
                expected = {f"rgb_{frame:05d}.jpg" for frame in frames}
                for camera in (0, 1):
                    files = {path.name for path in (folder / "frames" / "rgb" / f"Camera_{camera}").glob("*.jpg")}
                    if files != expected:
                        raise ValueError(f"RGB帧与标定帧不一致：{sequence}/Camera_{camera}")
                tasks = [(frame, camera) for frame in frames for camera in (0, 1)]
                with ThreadPoolExecutor(max_workers=config.get("workers", 4)) as executor:
                    converted = executor.map(lambda pair: convert_camera(root, sequence, *pair, palette), tasks)
                    records = {(item["frame"], item["camera"]): item for item in converted}
                diagnostic_frames = {frames[index] for index in np.linspace(0, len(frames) - 1, 3).astype(int)}
                diagnostics, temporal_count = [], 0
                for frame in frames:
                    record = synchronous_record(sequence, frame, cameras, records)
                    streams[(split, "stereo")].write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    if frame in diagnostic_frames:
                        diagnostics.append(geometry_check(root, record))
                    sources = temporal_sources(frame, frames, cameras, config.get("temporal4", {}))
                    if len(sources) == 3:
                        streams[(split, "temporal")].write(" ".join([sequence, str(frame), *map(str, sources)]) + "\n")
                        temporal_count += 1
                for stream in streams.values():
                    stream.flush()
                audit = {
                    "split": split, "stereo_samples": len(frames), "camera_images": len(records),
                    "temporal4_samples": temporal_count, "geometry": diagnostics,
                    "mean_valid_depth_fraction": float(np.mean([item["valid_fraction"] for item in records.values()])),
                    "mean_saturated_depth_fraction": float(np.mean([item["saturated_fraction"] for item in records.values()])),
                }
                summary["sequences"][sequence] = audit
                save_json(output / "summary.json", summary)
                print(json.dumps({"sequence": sequence, **audit}, ensure_ascii=False), flush=True)
    finally:
        for stream in streams.values():
            stream.close()
    for temporary, final in completed_paths:
        temporary.replace(final)
    summary["completed"] = True
    summary["camera_images"] = sum(item["camera_images"] for item in summary["sequences"].values())
    summary["synchronized2_samples"] = sum(item["stereo_samples"] for item in summary["sequences"].values())
    summary["temporal4_samples"] = sum(item["temporal4_samples"] for item in summary["sequences"].values())
    save_json(output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))
