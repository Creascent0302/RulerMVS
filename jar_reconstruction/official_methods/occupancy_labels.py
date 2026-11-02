"""从可见深度生成保守射线标签；物体内部和遮挡后方保持 unknown。"""

import itertools
import torch
from jar.geometry import project, sample_maps, pixel_grid, unproject


@torch.no_grad()
def observed_occupancy(
    inputs, targets, roi, voxel_size, batch_index=0, chunk=4096, hit_margin=0.03
):
    """命中体素优先；仅同一相机看见八角且都位于首个命中前，才标 free。

    这是深度传感器观测分辨率下的保守标签，不是完整实体占据真值。
    grid_origin 为最小边界，shape 的轴顺序 x/y/z。语义仅从首个可见命中赋值。
    """
    depth = targets["depth"][batch_index]
    device = depth.device
    bounds = torch.as_tensor(roi, device=device, dtype=torch.float32)
    size = torch.as_tensor(voxel_size, device=device, dtype=torch.float32).expand(3)
    shape_float = (bounds[:, 1] - bounds[:, 0]) / size
    shape = shape_float.round().long()
    if not torch.allclose(shape_float, shape.float(), atol=1e-5):
        raise ValueError("ROI 各轴长度必须为体素边长的整数倍")
    axes = [torch.arange(int(v), device=device) for v in shape]
    indices = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    centers = bounds[:, 0] + (indices + 0.5) * size
    state = torch.full((len(centers),), 2, dtype=torch.uint8, device=device)
    semantic = torch.full((len(centers),), -1, dtype=torch.long, device=device)
    k, t = inputs["K"][batch_index], inputs["T_cam_from_rig"][batch_index]
    valid_depth = targets.get("depth_valid", targets["depth"] > 0)[batch_index]
    offsets = torch.tensor(list(itertools.product((-0.5, 0.5), repeat=3)), device=device) * size
    h, w = depth.shape[-2:]
    for start in range(0, len(centers), chunk):
        center = centers[start : start + chunk]
        probes = (center[:, None] + offsets).reshape(-1, 3)
        uv, z, visible, _ = project(
            probes,
            k,
            t,
            (h, w),
            inputs["view_mask"][batch_index],
            inputs["image_mask"][batch_index],
        )
        measured = sample_maps(depth[:, None], uv, mode="nearest")[..., 0]
        known = sample_maps(valid_depth[:, None].float(), uv, mode="nearest")[..., 0] > 0.5
        before_hit = visible & known & (z > 0) & (z < measured - hit_margin)
        free = before_hit.reshape(len(k), len(center), 8).all(-1).any(0)
        state[start : start + len(center)][free] = 0
    grid = pixel_grid(h, w, device=device).reshape(-1, 2)
    for camera in range(len(k)):
        points, _ = unproject(grid, depth[camera].flatten(), k[camera], t[camera])
        cell = torch.floor((points - bounds[:, 0]) / size).long()
        valid = valid_depth[camera].flatten() & inputs["image_mask"][batch_index, camera].flatten()
        valid &= inputs["view_mask"][batch_index, camera] & ((cell >= 0) & (cell < shape)).all(-1)
        ids = (cell[:, 0] * shape[1] + cell[:, 1]) * shape[2] + cell[:, 2]
        state[ids[valid]] = 1
        if "semantic" in targets:
            # 同体素多类别冲突不猜测多数实体：语义设未知，几何命中仍保留。
            labs = targets["semantic"][batch_index, camera].flatten()
            for label in labs[valid & (labs >= 0)].unique():
                selected = ids[valid & (labs == label)].unique()
                old = semantic[selected]
                semantic[selected] = torch.where((old == -1) | (old == label), label, -2)
    semantic[semantic < 0] = -1
    return dict(
        state=state.reshape(tuple(shape.tolist())),
        semantics=semantic.reshape(tuple(shape.tolist())),
        grid_origin=bounds[:, 0].clone(),
        voxel_size=size.clone(),
    )
