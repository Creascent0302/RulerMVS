"""按CARLA真实frame ID整理采集索引；原文件只读，不跨帧补图。"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .data import carla_transform_to_cv

WIDE_IDS = ("stereo_left", "stereo_right", "stereo_up", "stereo_down")
SENSORS = WIDE_IDS + ("depth", "radar")


def carla_pose_matrix(pose):
    """CARLA/UE角度（度）到local-camera -> CARLA-rig，参照官方Transform::GetMatrix。"""
    pitch, yaw, roll = (math.radians(float(pose.get(k, 0.0))) for k in ("pitch", "yaw", "roll"))
    cp, sp, cy, sy, cr, sr = (
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
        math.cos(roll),
        math.sin(roll),
    )
    # 等价于 Rz(yaw) @ Ry(-pitch) @ Rx(-roll)，不是右手系的同名欧拉角组合。
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    rx = np.array([[1.0, 0, 0], [0, cr, sr], [0, -sr, cr]])
    matrix = np.eye(4)
    matrix[:3, :3] = rz @ ry @ rx
    matrix[:3, 3] = [float(pose[k]) for k in ("x", "y", "z")]
    return matrix


def camera_calibration(sensor):
    intr = sensor["intrinsic"]
    K = [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]]
    T = carla_transform_to_cv(carla_pose_matrix(sensor["extrinsic"])).tolist()
    return {"K": K, "T_cam_from_rig": T, "image_size": [int(intr["height"]), int(intr["width"])]}


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def inspect_sequence(
    root, name, depth_type="unverified", depth_scale=1.0, max_depth=1000.0, depth_evidence=None
):
    directory = root / name
    calibration = json.loads((directory / "calibration.json").read_text())
    index = json.loads((directory / "sync_index.json").read_text())
    cal = {k: camera_calibration(calibration[k]) for k in WIDE_IDS}
    depth_cal = camera_calibration(calibration["depth"])
    aligned = [
        k
        for k in WIDE_IDS
        if all(
            np.allclose(cal[k][f], depth_cal[f], atol=1e-6)
            for f in ("K", "T_cam_from_rig", "image_size")
        )
    ]
    if len(aligned) != 1:
        raise ValueError(
            "本适配器要求深度与一台RGB像素对齐；请提供明确的独立深度相机监督适配，不能复制深度到其他视图"
        )
    depth_camera = aligned[0]
    rows, inventory, indexed_paths, missing_files = [], [], set(), []
    frame_keys = set()
    for raw_id, entries in sorted(index.items(), key=lambda kv: int(kv[0])):
        frame_id = int(raw_id)
        if frame_id in frame_keys:
            raise ValueError("索引含重复数值frame ID")
        frame_keys.add(frame_id)
        actual = {}
        for sensor, relative in entries.items():
            if sensor not in SENSORS:
                continue  # 不读取长焦图像
            path = directory / relative
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("索引路径越出轨迹目录")
            if path.stem != f"{sensor}_{frame_id:08d}":
                raise ValueError("文件名frame ID与同步索引不一致: " + str(path))
            indexed_paths.add(str(path.relative_to(directory)))
            if path.is_file():
                actual[sensor] = str(path.relative_to(root))
            else:
                missing_files.append({"frame_id": frame_id, "sensor": sensor, "path": str(path)})
        present = [k for k in WIDE_IDS if k in actual]
        count = len(present)
        quality = {
            "camera_completeness": "complete" if count == 4 else ("partial" if count else "no_rgb"),
            "sensor_completeness": "complete" if all(k in actual for k in SENSORS) else "partial",
            "valid_camera_count": count,
            "missing_camera_ids": [k for k in WIDE_IDS if k not in actual],
            "depth_available": "depth" in actual,
            "radar_available": "radar" in actual,
            "direct_depth_supervision_available": depth_camera in actual and "depth" in actual,
            "depth_definition_verified": depth_type != "unverified",
            "supervised_training_candidate": count >= 2
            and depth_camera in actual
            and "depth" in actual,
        }
        record = {
            "sequence": name,
            "frame_id": frame_id,
            "timestamp": None,
            "sync_basis": "frame_id",
            "timestamp_source": "not_recorded",
            "source_index_frame_id": raw_id,
            **quality,
            "auxiliary_sensors": {
                k: {"path": actual[k], "calibration": calibration[k], "used_as_model_input": False}
                for k in ("depth", "radar")
                if k in actual
            },
            "cameras": [],
        }
        for cid in present:
            camera = {
                "camera_id": cid,
                "role": "wide",
                "frame_id": frame_id,
                "timestamp": None,
                "rgb": actual[cid],
                **cal[cid],
            }
            if cid == depth_camera and "depth" in actual:
                camera["targets"] = {
                    "frame_id": frame_id,
                    "depth": actual["depth"],
                    "depth_type": depth_type,
                    "depth_encoding": "metric_npy",
                    "depth_scale": depth_scale,
                    "max_valid_depth": max_depth,
                    "source_sensor": "depth",
                }
                if depth_evidence is not None:
                    camera["targets"]["depth_evidence"] = depth_evidence
            record["cameras"].append(camera)
        inventory.append(record)
        if count:
            rows.append(record)
    # 排除旧采集副本和未索引帧；永远不根据文件名自动补入另一次轨迹。
    disk_files = [
        p
        for sensor in SENSORS
        for p in (directory / ("camera/" + sensor if sensor in WIDE_IDS else sensor)).rglob("*")
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".npy")
    ]
    unindexed = sorted(
        str(p.relative_to(directory))
        for p in disk_files
        if str(p.relative_to(directory)) not in indexed_paths
    )
    old = (
        json.loads((directory / "matched_frames.json").read_text())
        if (directory / "matched_frames.json").exists()
        else {}
    )
    available8 = {int(k) for k, v in index.items() if len(v) == 8}
    declared8 = {int(k) for k in old.get("matched_frames", [])}
    report = {
        "sequence": name,
        "indexed_frames": len(index),
        "usable_rgb_frames": len(rows),
        "rgb_count_histogram": dict(
            sorted(Counter(r["valid_camera_count"] for r in inventory).items())
        ),
        "complete_rgb": sum(r["camera_completeness"] == "complete" for r in rows),
        "partial_rgb": sum(r["camera_completeness"] == "partial" for r in rows),
        "complete_sensors_without_tele": sum(r["sensor_completeness"] == "complete" for r in rows),
        "supervised_training_candidates": sum(r["supervised_training_candidate"] for r in rows),
        "supervision_ready": depth_type != "unverified",
        "depth_evidence": depth_evidence,
        "depth_aligned_camera": depth_camera,
        "missing_indexed_files": missing_files,
        "unindexed_file_count_excluding_tele": len(unindexed),
        "original_eight_sensor_matched_count": old.get("matched_count"),
        "original_matched_list_agrees_with_index": declared8 == available8,
        "index_sha256": hashlib.sha256((directory / "sync_index.json").read_bytes()).hexdigest(),
        "calibration_sha256": hashlib.sha256(
            (directory / "calibration.json").read_bytes()
        ).hexdigest(),
    }
    return rows, inventory, unindexed, report


def prepare(
    root,
    output,
    train_sequence,
    test_sequence,
    depth_type="unverified",
    depth_scale=1.0,
    max_depth=1000.0,
    depth_evidence=(),
):
    root, output = Path(root).absolute(), Path(output).absolute()
    if train_sequence == test_sequence:
        raise ValueError("训练与测试必须使用不同轨迹，不能随机拆相邻帧")
    if depth_type not in ("unverified", "optical_z", "ray_range"):
        raise ValueError("depth_type非法")
    if not math.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale须为有限正数")
    evidence_by_sequence = {}
    for evidence_file in depth_evidence:
        path = Path(evidence_file).absolute()
        evidence = json.loads(path.read_text())
        name = evidence["sequence"]
        if name not in (train_sequence, test_sequence) or depth_type == "unverified":
            raise ValueError("雷达证据必须属于当前轨迹，并显式选择已经人工检查的深度类型")
        for filename, key in [
            ("calibration.json", "calibration_sha256"),
            ("sync_index.json", "index_sha256"),
        ]:
            actual_hash = hashlib.sha256((root / name / filename).read_bytes()).hexdigest()
            if evidence[key] != actual_hash:
                raise ValueError("雷达证据与当前标定或同步索引不一致，请重新核查")
        evidence_by_sequence[name] = {
            "basis": "empirical_radar_projection_comparison",
            "report": str(path),
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "capture_decode_formula_verified": False,
            "scale_policy": "显式设置；雷达诊断拟合值不自动应用",
        }
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "data_root": str(root),
        "excluded_camera_ids": ["tele_left", "tele_right"],
        "camera_ids": list(WIDE_IDS),
        "depth_type": depth_type,
        "depth_scale": depth_scale,
        "timestamp_policy": "仅使用同一CARLA frame ID；没有保存timestamp，不捏造秒数",
        "split_policy": "轨迹隔离，无验证集；test不能兼作调参validation。没有地图ID/车辆世界位姿，不能证明地图空间独立。",
        "raw_data_modified": False,
        "sequences": {},
        "manifests": {},
    }
    memberships = {}
    for split, name in [("train", train_sequence), ("test", test_sequence)]:
        rows, inventory, unindexed, info = inspect_sequence(
            root, name, depth_type, depth_scale, max_depth, evidence_by_sequence.get(name)
        )
        report["sequences"][split] = info
        memberships[split] = {r["frame_id"] for r in inventory}
        groups = {
            "all": rows,
            "complete": [r for r in rows if r["camera_completeness"] == "complete"],
            "partial": [r for r in rows if r["camera_completeness"] == "partial"],
            "supervised": [r for r in rows if r["supervised_training_candidate"]],
            "inference_only": [r for r in rows if not r["supervised_training_candidate"]],
            "single_view": [r for r in rows if r["valid_camera_count"] == 1],
            "sensor_complete": [r for r in rows if r["sensor_completeness"] == "complete"],
            "no_rgb": [r for r in inventory if not r["cameras"]],
        }
        report["manifests"][split] = {}
        for group, records in groups.items():
            path = output / f"{split}_{group}.jsonl"
            write_jsonl(path, records)
            report["manifests"][split][group] = {"path": str(path), "frames": len(records)}
        (output / f"{split}_unindexed_files.json").write_text(
            json.dumps(unindexed, ensure_ascii=False, indent=2) + "\n"
        )
    report["indexed_frame_id_overlap"] = sorted(memberships["train"] & memberships["test"])
    # CARLA重启可重复frame ID；记录而不把它当内容哈希。此次两组索引无重叠。
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="data/carla_wide4")
    parser.add_argument("--train-sequence", default="captured_data (1)")
    parser.add_argument("--test-sequence", default="captured_data")
    parser.add_argument(
        "--depth-type", choices=["unverified", "optical_z", "ray_range"], default="unverified"
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="显式单位/比例修正；不自动按最大值或诊断拟合值缩放",
    )
    parser.add_argument(
        "--depth-evidence",
        nargs="*",
        default=[],
        help="人工检查过的各轨迹雷达诊断 JSON，保留经验判断来源",
    )
    parser.add_argument("--max-depth", type=float, default=1000.0)
    args = parser.parse_args()
    r = prepare(
        args.root,
        args.output,
        args.train_sequence,
        args.test_sequence,
        args.depth_type,
        args.depth_scale,
        args.max_depth,
        args.depth_evidence,
    )
    print(
        json.dumps(
            {
                "report": str(Path(args.output) / "report.json"),
                "depth_type": r["depth_type"],
                "sequences": {
                    k: {
                        x: v[x]
                        for x in (
                            "indexed_frames",
                            "complete_rgb",
                            "partial_rgb",
                            "supervised_training_candidates",
                        )
                    }
                    for k, v in r["sequences"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
