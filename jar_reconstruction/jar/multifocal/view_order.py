"""四目稠密路径的数值执行顺序规范；不改变相机角色或任何学习参数。

共享网络在实数运算下角色对称，但GPU归约顺序的小差异可触发不同选峰。
因此按已训练配置的稳定ID顺序执行，再按调用者顺序返回。没有主相机，
每台相机仍轮流建立自己的射线候选。六目、稀疏与参考裁减保持原契约。
"""

import torch


INPUT_CAMERA_FIELDS = (
    "rgb",
    "matching_rgb",
    "K",
    "T_cam_from_rig",
    "view_mask",
    "image_mask",
    "matching_image_mask",
    "camera_roles",
    "camera_valid",
    "image_sizes",
)
COARSE_CAMERA_FIELDS = (
    "logits",
    "depth",
    "peaks",
    "confidence",
    "support",
    "texture",
    "matching_evidence_weight",
    "matching_base_scores",
    "candidate_transport_update",
    "supervised_views",
)
DENSE_CAMERA_FIELDS = (
    "depth",
    "valid",
    "confidence",
    "confidence_logits",
    "observed",
    "support",
    "state",
)
REFINEMENT_CAMERA_FIELDS = (
    "depth",
    "gate",
    "step_m",
    "source_support",
    "depth_before",
    "residual_m",
)
POSTERIOR_CAMERA_FIELDS = (
    "logits",
    "valid",
    "depth",
    "lower_depth",
    "upper_depth",
    "mode_index",
    "candidate_valid",
    "probability",
    "reference_log_depth",
)
TOP_CAMERA_FIELDS = (
    "dense_depth",
    "dense_valid",
    "dense_confidence",
    "dense_observed",
    "dense_support",
    "dense_state",
)


def _gather_camera(value, order, axis=1):
    """order[B,N]沿已声明的相机轴gather，支持每个batch使用不同排列。"""
    if value.shape[0] != len(order) or value.shape[axis] != order.shape[1]:
        raise ValueError("声明的相机轴与batch/相机数量不一致")
    shape = [len(order)] + [1] * (value.ndim - 1)
    shape[axis] = order.shape[1]
    index = order.reshape(shape).expand_as(value)
    return value.gather(axis, index)


def canonicalize_wide_inputs(inputs, cfg, *, force=False):
    """返回(inputs_in_training_order, restore_order)，原顺序时不复制或计算张量。"""
    model, camera = cfg["model"], cfg.get("camera", {})
    enabled = (
        camera.get("mode") == "wide4"
        and (force or (
            model.get("dense_decoder", {}).get("enabled", False)
            and not model.get("sparse_decoder", {}).get("enabled", True)
            and model.get("execution", {}).get("reference_indices") is None
        ))
    )
    if not enabled or "camera_ids" not in inputs:
        return inputs, None
    expected = list(camera["wide_ids"])
    rows = inputs["camera_ids"]
    if all(list(row) == expected for row in rows):
        return inputs, None
    if len(rows) != inputs["rgb"].shape[0] or any(
        len(row) != len(expected) or set(row) != set(expected) for row in rows
    ):
        raise ValueError("四目 camera_ids 必须恰好包含配置的四个不同广角 ID")
    order = torch.tensor(
        [[list(row).index(name) for name in expected] for row in rows],
        device=inputs["rgb"].device,
        dtype=torch.long,
    )
    result = dict(inputs)
    for key in INPUT_CAMERA_FIELDS:
        if key in inputs:
            result[key] = _gather_camera(inputs[key], order)
    result["camera_ids"] = [list(expected) for _ in rows]
    return result, order.argsort(1)


def _restore_fields(record, names, order):
    result = dict(record)
    for key in names:
        if key in record:
            result[key] = _gather_camera(record[key], order)
    return result


def restore_wide_outputs(outputs, order):
    """按显式输出契约还原相机轴，不通过尺寸猜测候选/通道/世界节点轴。

    matching_per_view_*定义为[B,reference,source,...]，两轴都需要还原。
    逆深度坐标、细化轮次统计和世界点坐标没有相机轴，保持原对象。
    """
    if order is None:
        return outputs
    result = _restore_fields(outputs, TOP_CAMERA_FIELDS, order)
    if "coarse" in outputs:
        coarse = _restore_fields(outputs["coarse"], COARSE_CAMERA_FIELDS, order)
        for key, value in outputs["coarse"].items():
            if key.startswith("matching_per_view_"):
                coarse[key] = _gather_camera(_gather_camera(value, order), order, axis=2)
        result["coarse"] = coarse
    if "dense" in outputs:
        original = outputs["dense"]
        dense = _restore_fields(original, DENSE_CAMERA_FIELDS, order)
        if "stages" in original:
            dense["stages"] = [
                _restore_fields(stage, DENSE_CAMERA_FIELDS, order) for stage in original["stages"]
            ]
        if "refinement" in original:
            dense["refinement"] = _restore_fields(
                original["refinement"], REFINEMENT_CAMERA_FIELDS, order
            )
        for key in ("posterior", "posterior_full"):
            if original.get(key) is not None:
                dense[key] = _restore_fields(original[key], POSTERIOR_CAMERA_FIELDS, order)
        result["dense"] = dense
        # 公共别名继续指向还原后的同一张量，避免复制两份完整分辨率深度。
        for key in TOP_CAMERA_FIELDS:
            native = key.removeprefix("dense_")
            if key in result and native in dense:
                result[key] = dense[native]
    return result
