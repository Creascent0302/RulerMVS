"""直接解码器的训练数据选择；深度标签只在这个训练专用模块中读取。"""

import torch


def select_reference_targets(inputs, targets, camera_indices, depth_range=None, prediction_valid=None):
    """选取真实参考相机标签，缺标签或缺RGB的像素保持无监督。

    camera_indices可以是统一整数或[B]的原输入槽位。多参考模型返回的
    direct_camera_indices[:, position]正是该槽位，不能直接拿规范相机编号
    索引被重排的targets。返回depth/valid为[B,1,H,W]；不补造源相机深度。
    """
    rgb = inputs["rgb"]
    batch, views = rgb.shape[:2]
    height, width = rgb.shape[-2:]
    indices = torch.as_tensor(camera_indices, device=rgb.device, dtype=torch.long)
    if indices.ndim == 0:
        indices = indices.expand(batch)
    if indices.shape != (batch,) or bool(((indices < 0) | (indices >= views)).any()):
        raise ValueError("camera_indices必须为合法相机索引或[B]索引张量")
    if "depth" not in targets:
        return {
            "depth": rgb.new_zeros(batch, 1, height, width),
            "valid": torch.zeros(batch, 1, height, width, device=rgb.device, dtype=torch.bool),
        }
    depth = targets["depth"]
    if depth.shape != (batch, views, height, width):
        raise ValueError("多相机监督要求depth为[B,N,H,W]，不可广播单相机标签")
    valid = targets.get("depth_valid", torch.isfinite(depth) & (depth > 0))
    if valid.shape != depth.shape:
        raise ValueError("depth_valid形状必须与depth完全一致")
    rows = torch.arange(batch, device=rgb.device)
    selected = depth[rows, indices][:, None]
    valid = valid[rows, indices][:, None].bool()
    valid = valid & torch.isfinite(selected) & (selected > 0)
    valid &= inputs["view_mask"][rows, indices, None, None, None]
    valid &= inputs["image_mask"][rows, indices][:, None]
    if prediction_valid is not None:
        if prediction_valid.shape != valid.shape:
            raise ValueError("prediction_valid必须为[B,1,H,W]")
        valid &= prediction_valid.bool()
    if depth_range is not None:
        near, far = map(float, depth_range)
        if not 0 < near < far:
            raise ValueError("depth_range必须为正的[near,far]米制范围")
        valid &= (selected >= near) & (selected <= far)
    return {"depth": torch.where(valid, selected, 0), "valid": valid}
