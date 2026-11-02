"""将 Hypersim 射线距离转换为官方加载器使用的光轴深度。

只依赖 NumPy 和 h5py，读取逐场景的官方相机矩阵，不依赖 PyTorch3D。
输出目录规则和像素中心约定与 MVSAnywhere 的 HypersimDataset 一致。
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def optical_axis_factor(matrix, height, width):
    """Hypersim 相机朝向负 Z；返回单位射线在光轴上的投影长度。"""
    u = np.linspace(-1 + 1 / width, 1 - 1 / width, width, dtype=np.float32)
    v = np.linspace(1 - 1 / height, -1 + 1 / height, height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    pixels = np.stack((uu, vv, np.ones_like(uu)), axis=-1)
    rays = pixels @ np.asarray(matrix, dtype=np.float32).T
    return -rays[..., 2] / np.linalg.norm(rays, axis=-1)


def prepare(raw_root, metadata, split, limit=None):
    raw_root = Path(raw_root).resolve()
    output_root = raw_root.parent.with_name(raw_root.parent.name + "_preprocessed")
    with open(metadata, newline="", encoding="utf-8") as stream:
        cameras = {row["scene_name"]: row for row in csv.DictReader(stream)}
    with open(split, encoding="utf-8") as stream:
        scans = json.load(stream)

    written = 0
    skipped = 0
    visited = 0
    factor_key = None
    for scan, frames in scans.items():
        scene, camera = scan.split("/")
        row = cameras[scene]
        height = int(float(row["settings_output_img_height"]))
        width = int(float(row["settings_output_img_width"]))
        if factor_key != scene:
            matrix = np.array([
                [float(row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)]
                for i in range(3)
            ], dtype=np.float32)
            factor = optical_axis_factor(matrix, height, width)
            factor_key = scene
        folder = Path(scene) / "images" / f"scene_{camera}_geometry_hdf5"
        for frame in frames:
            if limit is not None and visited >= limit:
                return {"written": written, "skipped": skipped, "output": str(output_root)}
            visited += 1
            name = f"frame.{int(frame):04d}"
            source = raw_root / folder / f"{name}.depth_meters.hdf5"
            target = output_root / folder / f"{name}.depth_meters_planar.hdf5"
            if target.exists():
                skipped += 1
                continue
            with h5py.File(source, "r") as stream:
                distance = stream["dataset"][:].astype(np.float32)
            if distance.shape != (height, width):
                raise ValueError(f"原始深度尺寸与相机参数不符：{source}")
            depth = distance * factor
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            with h5py.File(temporary, "w") as stream:
                stream.create_dataset("dataset", data=depth.astype(np.float16),
                                      compression="gzip", compression_opts=4)
            temporary.replace(target)
            written += 1
    return {"written": written, "skipped": skipped, "output": str(output_root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--split", required=True, help="官方 train_files_bd.json 等帧清单")
    parser.add_argument("--limit", type=int, help="先转换少量帧检查，省略则处理全部清单")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须为正整数")
    print(json.dumps(prepare(args.raw_root, args.metadata, args.split, args.limit),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
