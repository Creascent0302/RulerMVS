"""整理新采集的四条 CARLA 轨迹，保留缺相机帧与每台相机独立深度。

原始文件只读；只写 JSONL 清单和核查结果，不复制图像/深度。
TRAIN=0+2、VAL=1、TEST=3 在查看模型误差前固定；不读取旧采集目录。
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jar.carla_capture import WIDE_IDS, camera_calibration


SPLITS = {
    "train": ("captured_data0", "captured_data2"),
    "val": ("captured_data1",),
    "test": ("captured_data3",),
}
SENSORS = (*WIDE_IDS, *(name + "_depth" for name in WIDE_IDS), "radar")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_sequence(root, sequence):
    directory = root / sequence
    calibration = json.loads((directory / "calibration.json").read_text())
    index = json.loads((directory / "sync_index.json").read_text())
    declared = json.loads((directory / "matched_frames.json").read_text())
    cameras = {name: camera_calibration(calibration[name]) for name in WIDE_IDS}
    for name, camera in cameras.items():
        depth_camera = camera_calibration(calibration[name + "_depth"])
        for key in ("K", "T_cam_from_rig", "image_size"):
            if not np.allclose(camera[key], depth_camera[key], atol=1e-6):
                raise ValueError(f"{sequence}/{name}: RGB 与对应深度不在同一像素格")
        transform = np.asarray(camera["T_cam_from_rig"])
        pose = calibration[name]["extrinsic"]
        expected_center = [pose["x"], -pose["y"], pose["z"]]
        if not np.allclose(np.linalg.inv(transform)[:3, 3], expected_center, atol=1e-5):
            raise ValueError("相机中心的 CARLA -> 右手 rig 转换错误")
        if not np.allclose(transform[:3, :3] @ transform[:3, :3].T, np.eye(3), atol=1e-5):
            raise ValueError("相机旋转矩阵不是正交矩阵")
        if not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5):
            raise ValueError("内部旋转没有保持右手坐标系")

    rows = []
    indexed_paths = set()
    sensor_counts = Counter()
    depth_headers = 0
    rgb_headers = 0
    sampled_nonfinite = 0
    sampled_empty_depths = 0
    total_bytes = 0
    numeric_frames = set()
    for raw_frame, entries in sorted(index.items(), key=lambda item: int(item[0])):
        frame = int(raw_frame)
        if frame in numeric_frames:
            raise ValueError("同步索引包含重复数值帧 ID")
        numeric_frames.add(frame)
        actual = {}
        for sensor, relative in entries.items():
            if sensor not in SENSORS:
                raise ValueError(f"新数据白名单之外的传感器: {sensor}")
            path = directory / relative
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("索引路径越出轨迹目录")
            expected_sensor = sensor.removesuffix("_depth")
            if path.stem != f"{expected_sensor}_{frame:08d}" or not path.is_file():
                raise ValueError(f"帧号或索引文件错误: {path}")
            indexed_paths.add(str(path.relative_to(directory)))
            sensor_counts[sensor] += 1
            actual[sensor] = str(path.relative_to(root))
            total_bytes += path.stat().st_size
            if sensor.endswith("_depth"):
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if array.dtype != np.float32 or array.shape != (1080, 1920):
                    raise ValueError(f"深度格式错误: {path}")
                if array.offset + array.nbytes != path.stat().st_size:
                    raise ValueError(f"深度文件长度错误: {path}")
                sampled = array[::32, ::32]
                sampled_nonfinite += int((~np.isfinite(sampled)).sum())
                sampled_empty_depths += int(not np.any(np.isfinite(sampled) & (sampled > 0) & (sampled < 999.9)))
                depth_headers += 1
            elif sensor in WIDE_IDS:
                with Image.open(path) as image:
                    if image.mode != "RGB" or image.size != (1920, 1080):
                        raise ValueError(f"RGB 格式错误: {path}")
                rgb_headers += 1

        present = [name for name in WIDE_IDS if name in actual]
        labelled = [name for name in present if name + "_depth" in actual]
        supervised = len(present) >= 2 and bool(labelled)
        record = {
            "sequence": sequence,
            "frame_id": frame,
            "timestamp": None,
            "sync_basis": "frame_id",
            "timestamp_source": "not_recorded",
            "geometry_units": "meters",
            "sample_type": "synchronized_rig",
            "synchronized_capture": True,
            "valid_camera_count": len(present),
            "missing_camera_ids": [name for name in WIDE_IDS if name not in present],
            "camera_completeness": "complete" if len(present) == 4 else "partial" if present else "no_rgb",
            "depth_supervised_camera_ids": labelled,
            "four_rgb_four_depth": len(present) == 4 and len(labelled) == 4,
            "supervised_training_candidate": supervised,
            "upper_reference_supervised": supervised and "stereo_up" in labelled,
            "radar_available": "radar" in actual,
            "radar_used_as_input_or_target": False,
            "cameras": [],
        }
        for name in present:
            camera = {
                "camera_id": name,
                "role": "wide",
                "frame_id": frame,
                "timestamp": None,
                "rgb": actual[name],
                **cameras[name],
            }
            if name in labelled:
                camera["targets"] = {
                    "frame_id": frame,
                    "depth": actual[name + "_depth"],
                    "depth_type": "optical_z",
                    "depth_encoding": "metric_npy",
                    "depth_units": "meters",
                    "depth_scale": 1.0,
                    "max_valid_depth": 999.9,
                    "source_sensor": name + "_depth",
                    "depth_evidence": "research_memory 126 + geometry_check.json; 未取得实采脚本版本哈希",
                }
            record["cameras"].append(camera)
        rows.append(record)

    physical = {
        str(path.relative_to(directory))
        for kind in ("camera", "depth", "radar")
        for path in (directory / kind).rglob("*")
        if path.is_file()
    }
    if physical != indexed_paths:
        raise ValueError(f"{sequence}: 磁盘文件与同步索引不完全对应")
    full_sensor_frames = {row["frame_id"] for row in rows if row["four_rgb_four_depth"] and row["radar_available"]}
    if full_sensor_frames != set(declared["matched_frames"]):
        raise ValueError("九传感器匹配清单与同步索引不一致")
    report = {
        "frames": len(rows),
        "supervised": sum(row["supervised_training_candidate"] for row in rows),
        "upper_reference": sum(row["upper_reference_supervised"] for row in rows),
        "four_rgb": sum(len(row["cameras"]) == 4 for row in rows),
        "four_rgb_four_depth": sum(row["four_rgb_four_depth"] for row in rows),
        "camera_count_histogram": dict(Counter(len(row["cameras"]) for row in rows)),
        "sensor_counts": dict(sensor_counts),
        "depth_headers_checked": depth_headers,
        "rgb_headers_checked": rgb_headers,
        "depth_sparse_nonfinite_values": sampled_nonfinite,
        "depth_sparse_empty_files": sampled_empty_depths,
        "indexed_bytes": total_bytes,
        "json_sha256": {name: sha256(directory / name) for name in ("calibration.json", "sync_index.json", "matched_frames.json")},
        "missing_indexed_files": 0,
        "unindexed_files": 0,
        "calibration": cameras,
    }
    return rows, report


def geometry_check(root, rows):
    """data3 首中末完整帧，比较 Z/range 两种解释；不按匹配残差筛点。"""
    complete = [row for row in rows if row["four_rgb_four_depth"]]
    chosen = [complete[0], complete[len(complete) // 2], complete[-1]]
    residuals = {mode: [] for mode in ("optical_z", "ray_range")}
    smooth_residuals = {mode: [] for mode in residuals}
    eligible_count = 0
    frame_results = []
    for row in chosen:
        cameras = {camera["camera_id"]: camera for camera in row["cameras"]}
        depths = {name: np.load(root / camera["targets"]["depth"], mmap_mode="r") for name, camera in cameras.items()}
        per_frame = {mode: [] for mode in residuals}
        for reference, source in itertools.permutations(WIDE_IDS, 2):
            raw = depths[reference]
            yy, xx = np.meshgrid(np.arange(8, raw.shape[0] - 1, 16), np.arange(8, raw.shape[1] - 1, 16), indexing="ij")
            x, y = xx.ravel(), yy.ravel()
            pixels = np.stack((x, y, np.ones_like(x)), -1).astype(np.float64)
            K_ref = np.asarray(cameras[reference]["K"])
            K_src = np.asarray(cameras[source]["K"])
            relative = np.asarray(cameras[source]["T_cam_from_rig"]) @ np.linalg.inv(np.asarray(cameras[reference]["T_cam_from_rig"]))
            rays = pixels @ np.linalg.inv(K_ref).T
            raw_z = raw[y, x].astype(np.float64)
            ray_norm = np.linalg.norm(rays, axis=-1)
            eligible = np.isfinite(raw_z) & (raw_z >= 2) & (raw_z <= 80) & (raw_z / ray_norm >= 2)
            eligible_count += int(eligible.sum())
            neighbors = np.stack((raw[y, x], raw[y, x + 1], raw[y + 1, x], raw[y + 1, x + 1]), -1)
            ref_smooth = (neighbors.max(-1) - neighbors.min(-1)) / np.maximum(neighbors.min(-1), 1e-6) < .01
            records = {}
            for mode in residuals:
                z = raw_z if mode == "optical_z" else raw_z / ray_norm
                points = (rays * z[:, None]) @ relative[:3, :3].T + relative[:3, 3]
                uv = (points @ K_src.T)[:, :2] / np.maximum(points[:, 2:3], 1e-8)
                target = depths[source]
                inside = (uv[:, 0] >= 0) & (uv[:, 0] < target.shape[1] - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < target.shape[0] - 1)
                ix = np.clip(np.floor(uv[:, 0]).astype(int), 0, target.shape[1] - 2)
                iy = np.clip(np.floor(uv[:, 1]).astype(int), 0, target.shape[0] - 2)
                corners = np.stack((target[iy, ix], target[iy, ix + 1], target[iy + 1, ix], target[iy + 1, ix + 1]), -1)
                wx, wy = np.clip(uv[:, 0] - ix, 0, 1), np.clip(uv[:, 1] - iy, 0, 1)
                weights = np.stack(((1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy), -1)
                observed = (weights * corners).sum(-1)
                if mode == "ray_range":
                    source_rays = np.concatenate((uv, np.ones((len(uv), 1))), -1) @ np.linalg.inv(K_src).T
                    observed = observed / np.linalg.norm(source_rays, axis=-1)
                legal = eligible & inside & (points[:, 2] > 0) & np.isfinite(corners).all(-1) & (corners > 0).all(-1) & (corners < 999.9).all(-1) & (observed >= 2) & (observed <= 80)
                smooth = ref_smooth & ((corners.max(-1) - corners.min(-1)) / np.maximum(corners.min(-1), 1e-6) < .01)
                error = np.abs(points[:, 2] - observed) / np.maximum(observed, 1e-8)
                records[mode] = (legal, smooth, error)
            common = records["optical_z"][0] & records["ray_range"][0]
            common_smooth = common & records["optical_z"][1] & records["ray_range"][1]
            for mode, (_, _, error) in records.items():
                residuals[mode].append(error[common])
                per_frame[mode].append(error[common])
                smooth_residuals[mode].append(error[common_smooth])
        frame_results.append({"frame_id": row["frame_id"], "common_relative_median": {mode: float(np.median(np.concatenate(parts))) for mode, parts in per_frame.items()}})
    report = {"frames": frame_results, "eligible_reference_count": eligible_count, "protocol": "2–80m共同可投影点；16像素间距；平滑子集仅使用原深度2x2跳变<1%；遮挡未按误差删除", "results": {}}
    for mode in residuals:
        common = np.concatenate(residuals[mode])
        smooth = np.concatenate(smooth_residuals[mode])
        report["results"][mode] = {
            "common_count": len(common),
            "common_relative_median": float(np.median(common)),
            "smooth_count": len(smooth),
            "smooth_relative_median": float(np.median(smooth)),
            "smooth_within_one_percent": float((smooth < .01).mean()),
        }
    return report


def verify_loader(output):
    """真实新数据 CPU 核查：四独立标签、缺相机、轮换参考和推理标签隔离。"""
    from unittest.mock import patch
    import torch
    from torch.nn import functional as F
    from jar.data import build_dataset
    from jar.pretraining.data import MixedDataset, collate_mixed, dataset_config

    torch.set_num_threads(2)
    config = json.loads((output / "data_config.json").read_text())
    config.update(seed=20261028, augmentation=False)
    domain = config["datasets"][0]
    checks = []
    for split in SPLITS:
        data = MixedDataset(config, split)
        frames = data.datasets[0].frames
        chosen = {0, len(frames) - 1}
        for complete in (True, False):
            candidate = next((i for i, row in enumerate(frames) if (len(row["cameras"]) == 4) == complete), None)
            if candidate is not None:
                chosen.add(candidate)
        for index in sorted(chosen):
            original = data.datasets[0][index]
            rotated = data[(0, index, 0)]
            for sample in (original, rotated):
                inputs, targets = sample["inputs"], sample["targets"]
                assert inputs["rgb"].shape == (4, 3, 256, 448)
                assert inputs["matching_rgb"].shape == (4, 3, 512, 896)
                assert not inputs["rgb"][~inputs["view_mask"]].any()
                assert not targets["depth_valid"][~inputs["view_mask"]].any()
                lookup = {camera["camera_id"]: camera for camera in frames[index]["cameras"]}
                for slot, name in enumerate(inputs["camera_ids"]):
                    if name not in lookup:
                        continue
                    camera = lookup[name]
                    torch.testing.assert_close(inputs["T_cam_from_rig"][slot], torch.tensor(camera["T_cam_from_rig"]), rtol=0, atol=0)
                    expected_K = torch.tensor(camera["K"])
                    expected_K[0] *= 448 / 1920
                    expected_K[1] *= 256 / 1080
                    expected_K[0, 2] += .5 * 448 / 1920 - .5
                    expected_K[1, 2] += .5 * 256 / 1080 - .5
                    torch.testing.assert_close(inputs["K"][slot], expected_K)
                    if "targets" not in camera:
                        assert not targets["depth_valid"][slot].any()
                        continue
                    raw = torch.from_numpy(np.load(Path(domain["root"]) / camera["targets"]["depth"], allow_pickle=False))
                    valid = torch.isfinite(raw) & (raw > 0) & (raw < 999.9)
                    expected = F.interpolate(torch.where(valid, raw, 0)[None, None], size=(256, 448), mode="nearest-exact")[0, 0]
                    torch.testing.assert_close(targets["depth"][slot], expected, atol=0, rtol=0)
            assert rotated["inputs"]["view_mask"][0]
            assert rotated["targets"]["depth_valid"][0].any()
            if split != "train":
                assert rotated["inputs"]["camera_ids"][0] == "stereo_up"
            checks.append({"split": split, "index": index, "sequence": frames[index]["sequence"], "frame_id": frames[index]["frame_id"], "reference": rotated["inputs"]["camera_ids"][0]})

        inference = build_dataset(dataset_config(domain, config), split, load_targets=False)
        with patch("numpy.load", side_effect=AssertionError("推理不应打开任何深度 NPY")):
            assert inference[0]["targets"] == {}

    train = MixedDataset(config, "train")
    index = next(i for i, row in enumerate(train.datasets[0].frames) if row["four_rgb_four_depth"])
    samples = [train[(0, index, epoch)] for epoch in range(4)]
    batch = collate_mixed(samples)
    references = [ids[0] for ids in batch["inputs"]["camera_ids"]]
    assert set(references) == set(WIDE_IDS)
    for sample_index, sample in enumerate(samples):
        for key, value in sample["inputs"].items():
            if torch.is_tensor(value):
                torch.testing.assert_close(batch["inputs"][key][sample_index], value, rtol=0, atol=0)
        torch.testing.assert_close(batch["targets"]["depth"][sample_index], sample["targets"]["depth"], rtol=0, atol=0)
    result = {"status": "passed", "device": "cpu", "checked_samples": checks, "mixed_batch_reference_ids": references, "mixed_batch_rgb_shape": list(batch["inputs"]["rgb"].shape), "inference_target_reads": 0, "model_forward_or_training_performed": False}
    write_json(output / "loader_check.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("../data/CARLA_data"))
    parser.add_argument("--output", type=Path, default=Path("runs/active/v2_training_20261028/carla_prepared"))
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("输出目录非空；不覆盖已经固定的划分")
    sequences = {}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "root": str(root), "splits": SPLITS, "sequences": {}, "manifests": {}, "raw_modified": False, "script_sha256": sha256(Path(__file__))}
    for sequence in ("captured_data0", "captured_data1", "captured_data2", "captured_data3"):
        sequences[sequence], report["sequences"][sequence] = inspect_sequence(root, sequence)
    geometry = geometry_check(root, sequences["captured_data3"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "geometry_check.json", geometry)
    for split, names in SPLITS.items():
        rows = [row for name in names for row in sequences[name]]
        groups = {
            "inventory": rows,
            "all_rgb": [row for row in rows if row["cameras"]],
            "complete": [row for row in rows if len(row["cameras"]) == 4],
            "partial": [row for row in rows if 0 < len(row["cameras"]) < 4],
            "unusable": [row for row in rows if not row["supervised_training_candidate"]],
            "supervised": [row for row in rows if row["supervised_training_candidate"]],
            "upper": [row for row in rows if row["upper_reference_supervised"]],
        }
        report["manifests"][split] = {}
        for name, members in groups.items():
            path = output / f"{split}_{name}.jsonl"
            write_jsonl(path, members)
            report["manifests"][split][name] = {"path": str(path), "frames": len(members), "sha256": sha256(path)}
    camera = {"mode": "wide4", "wide_ids": list(WIDE_IDS), "tele_ids": [], "allow_missing": True, "disabled_ids": []}
    domain = {
        "name": "CARLA_new_wide4",
        "root": str(root),
        "num_views": 4,
        "reference_index": 2,
        "geometry_units": "meters",
        "manifests": {split: str(output / f"{split}_{'supervised' if split == 'train' else 'upper'}.jsonl") for split in SPLITS},
        "near": 2.0,
        "far": 80.0,
        "depth_scale": 1.0,
        "eval_range": [2.0, 80.0],
        "correspondence": True,
        "camera": camera,
        "train_manifest_sha256": report["manifests"]["train"]["supervised"]["sha256"],
    }
    report["notes"] = [
        "主训练使用TRAIN全部>=2RGB且至少一个同相机GT配对帧；参考在可监督相机中轮换。",
        "主VAL/TEST固定上相机；不能用其它相机GT补上相机。其余可监督帧另外保留。",
        "complete/partial按RGB数量划分，unusable按多视图监督条件划分，两类统计维度允许交叠。",
        "全部原始有效深度标签保留；2–80m是本轮预先固定的训练/评价范围，不由VAL或TEST决定。",
        "四轨迹没有地图、路线、世界位姿或时间戳；仅证明轨迹/帧划分隔离，不能保证地图空间无重合。",
        "深度定义结合126参考脚本和跨相机几何证据；不是实采脚本版本逐字认证。",
        "雷达不作为输入或监督，参考采集脚本雷达列顺序问题仍存在。",
        "NPY全量查头和文件长度、32像素稀疏值检查；PNG全量查头，不等于全量像素解码认证。",
    ]
    write_json(output / "data_config.json", {"image_size": [256, 448], "matching_image_size": [512, 896], "datasets": [domain]})
    write_json(output / "report.json", report)
    loader_check = verify_loader(output)
    print("CPU loader check:", loader_check["status"])
    print(json.dumps({"output": str(output), "manifests": report["manifests"], "geometry": geometry["results"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
