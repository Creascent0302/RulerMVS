"""只切换现有标尺用途；不改变投影、标签、编码器或候选族结构。"""

import copy

from research.rulermvs_v6_20261101.model import build_from_checkpoint


SETTINGS = {
    "control": {},
    "no_source_geometry": {"use_geometry": False},
    "no_scale_routing": {"scale_routing": False},
    "no_jacobian_width": {"fine_step_pixels": 0.0},
}

DESCRIPTIONS = {
    "control": "V6 parent；保留全部标尺用途。",
    "no_source_geometry": (
        "仅去掉源注意力pair中的导数方向/投影足迹，以及evidence的加权绝对敏感度；"
        "保留精确投影、有效掩码、尺度路由和细候选半径。"
    ),
    "no_scale_routing": (
        "源采样固定使用stride4层，原三层读取与张量预算不变；"
        "保留精确投影、源几何条件和Jacobian半径。"
    ),
    "no_jacobian_width": (
        "将fine_step_pixels设为0，细搜索半径仅由粗刻度半宽及原上下限决定；"
        "仍为四候选族各三点，候选仍精确投影。"
    ),
}


def build_geometry_model(saved, ablation):
    if ablation not in SETTINGS:
        raise ValueError(f"未知标尺消融: {ablation}")
    config = copy.deepcopy(saved["config"])
    previous = config.get("paper_geometry_ablation")
    if previous is not None and previous != ablation:
        raise ValueError("不能把另一消融训练后的权重作为本组共同起点")
    config["model"].update(SETTINGS[ablation])
    record = dict(saved, config=config)
    model, config = build_from_checkpoint(record, "parent")
    config["paper_geometry_ablation"] = ablation
    config["paper_geometry_description"] = DESCRIPTIONS[ablation]
    model.config = config
    if model.decoder.use_scene or model.decoder.local_kind != "scale_space":
        raise ValueError("本消融只适用于无scene的V6 scale_space底座")
    return model, config
