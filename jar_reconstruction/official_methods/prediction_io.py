"""跨独立环境的 NPZ+JSON 协议，原生对象另存为 PyTorch 文件。"""

import json
from pathlib import Path
import numpy as np
import torch


def native_to_cpu(value):
    """保留上游 NamedTuple 类型，不能只保留元组里的张量。"""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: native_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [native_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        children = [native_to_cpu(item) for item in value]
        return type(value)(*children) if hasattr(value, "_fields") else tuple(children)
    return value


def export_prediction(path, prediction, converted, inputs, metadata, config):
    path = Path(path)
    arrays = {}
    for key in (
        "depth",
        "depth_K",
        "depth_valid",
        "valid",
        "confidence",
        "coarse_confidence",
        "semantics",
        "occupancy_probability",
    ):
        if key in prediction and torch.is_tensor(prediction[key]):
            arrays[key] = prediction[key].detach().cpu().numpy()
    if "reference_indices" in prediction:
        arrays["reference_indices"] = np.asarray(prediction["reference_indices"], dtype=np.int64)
    arrays["K_input"] = inputs["K"].detach().cpu().numpy()
    arrays["T_cam_from_rig"] = inputs["T_cam_from_rig"].detach().cpu().numpy()
    for index, points in enumerate(converted.get("points", [])):
        arrays[f"postprocess_points_{index}"] = points.detach().cpu().numpy()
    for index, occ in enumerate(prediction.get("occupancy", [])):
        for key, value in occ.items():
            if torch.is_tensor(value):
                arrays[f"occupancy_{index}_{key}"] = value.detach().cpu().numpy()
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    manifest = {
        "format_version": 1,
        "method": config["method"],
        "sample_metadata": metadata,
        "fields": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
        "coordinate_system": "rig x前 y左 z上；CV相机x右 y下 z前；整数像素中心",
        "geometry_units": config.get("data", {}).get("geometry_units", "meters"),
        "postprocess": converted.get("conversion"),
        "config": config,
        "native_file": path.with_suffix(".native.pt").name,
    }
    path.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    torch.save(native_to_cpu(prediction.get("native", prediction)), path.with_suffix(".native.pt"))
