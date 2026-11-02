"""原始几何输出的有限性审计；不采样、不应用有效掩码、不修改预测。"""

import torch


def geometry_tensors(prediction):
    """只收集可比较的几何输出，排除允许含 -inf 的匹配 logits。"""
    tensors = {}
    keys = ("depth", "dense_depth", "points", "normals", "scale", "scales", "confidence")

    def collect(values, prefix, selected):
        for key in selected:
            value = values.get(key)
            if torch.is_tensor(value) and value.is_floating_point():
                tensors[prefix + key] = value

    collect(prediction, "", keys)
    collect(prediction.get("coarse", {}), "coarse.", ("depth",))
    for index, scene in enumerate(prediction.get("scenes", [])):
        collect(scene, f"scenes.{index}.", keys)
        collect(scene.get("aux", {}), f"scenes.{index}.aux.", ("reference_depth",))
    for index, native in enumerate(prediction.get("native", [])):
        if isinstance(native, dict):
            selected = [
                key
                for key in native
                if key.startswith("depth_pred_") or key in ("coarse_depth", "flow1", "flow2")
            ]
            collect(native, f"native.{index}.", selected)
    return tensors


@torch.no_grad()
def raw_output_audit(prediction):
    """统计未清理的网络输出；无效槽中的 NaN/Inf 也记录，不据此丢弃样本。"""
    fields = {}
    for name, value in geometry_tensors(prediction).items():
        counts = (
            torch.stack(
                (torch.isnan(value).sum(), torch.isposinf(value).sum(), torch.isneginf(value).sum())
            )
            .cpu()
            .tolist()
        )
        fields[name] = {
            "shape": list(value.shape),
            "elements": value.numel(),
            "nan": counts[0],
            "positive_inf": counts[1],
            "negative_inf": counts[2],
            "nonfinite": sum(counts),
        }
    return {
        "tensor_count": len(fields),
        "nonfinite_elements": sum(value["nonfinite"] for value in fields.values()),
        "fields": fields,
    }


def aggregate_raw_audits(audits):
    """不同输出表示可能重复同一数值；总数只作报警，不能解释为像素错误率。"""
    fields = {}
    for audit in audits:
        for name, values in audit["fields"].items():
            entry = fields.setdefault(
                name,
                dict(frames=0, elements=0, nan=0, positive_inf=0, negative_inf=0, nonfinite=0),
            )
            entry["frames"] += 1
            for key in ("elements", "nan", "positive_inf", "negative_inf", "nonfinite"):
                entry[key] += values[key]
    return {
        "frames": len(audits),
        "frames_without_comparable_tensors": sum(a["tensor_count"] == 0 for a in audits),
        "frames_with_nonfinite": sum(a["nonfinite_elements"] > 0 for a in audits),
        "nonfinite_elements": sum(a["nonfinite_elements"] for a in audits),
        "fields": fields,
        "definition": (
            "统计网络返回的原始几何张量，含无效槽，未应用GT/预测掩码。"
            "不同表示的计数可能重复，不是像素错误率；不检验深度范围或数值正确性。"
        ),
    }
