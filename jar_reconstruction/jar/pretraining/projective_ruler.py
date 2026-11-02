"""将标定编译为参考射线上的精确投影及空间标尺。

候选深度为参考相机的光轴深度。每条参考射线只变换一次方向，
随后以分式线性函数求每个深度的投影；没有用一阶近似代替重投影。
K 使用标准针孔底行 (0, 0, 1)，允许 skew 和非中心主点。
"""

from dataclasses import dataclass

import torch

from jar.geometry import sample_maps


@dataclass
class ProjectiveRuler:
    """一个参考视图与多个源视图的标定常量，不缓存图像或深度。"""

    rig_direction: torch.Tensor
    rig_center: torch.Tensor
    camera_direction: torch.Tensor
    translation: torch.Tensor
    projected_direction: torch.Tensor
    projected_translation: torch.Tensor
    camera_pixel_derivative: torch.Tensor
    projected_pixel_derivative: torch.Tensor
    depth_numerator: torch.Tensor

    def project(self, depths, image_hw, view_mask=None, image_mask=None):
        """depths [D,P] -> uv [N,D,P,2] 与同网格标尺、有效掩码。

        depth_derivative 是原图像素/参考深度单位；pixel_derivative 的
        最后两轴依次为源像素坐标、参考像素坐标，已包含当前深度。
        无效投影返回有限数值，调用方必须继续使用 valid 掩码。
        """
        with torch.autocast(device_type=depths.device.type, enabled=False):
            depths = depths.to(self.camera_direction.dtype)
            optical_depth = (depths[None] * self.camera_direction[:, None, :, 2]
                             + self.translation[:, None, None, 2])
            denominator = optical_depth.clamp_min(1e-4)
            numerator = (depths[None, :, :, None] * self.projected_direction[:, None]
                         + self.projected_translation[:, None, None])
            uv = numerator / denominator[..., None]
            height, width = image_hw
            valid = ((optical_depth > 1e-4) & torch.isfinite(optical_depth)
                     & torch.isfinite(uv).all(-1)
                     & (uv[..., 0] >= 0) & (uv[..., 0] <= width - 1)
                     & (uv[..., 1] >= 0) & (uv[..., 1] <= height - 1))
            if view_mask is not None:
                valid = valid & view_mask[:, None, None].bool()

            # 对有效深度，商导数分子与Z无关，可每条射线只计算一次。
            depth_derivative = self.depth_numerator[:, None] / denominator[..., None].square()
            pixel_derivative = (self.projected_pixel_derivative[:, None, None]
                                - uv[..., None] * self.camera_pixel_derivative[:, None, None, 2:3])
            pixel_derivative = pixel_derivative * (depths[None] / denominator)[..., None, None]
            uv = torch.nan_to_num(uv, nan=0.0, posinf=1e6, neginf=-1e6)
            depth_derivative = torch.nan_to_num(depth_derivative, nan=0.0, posinf=0.0, neginf=0.0)
            pixel_derivative = torch.nan_to_num(pixel_derivative, nan=0.0, posinf=0.0, neginf=0.0)
            if image_mask is not None:
                sampled_mask = sample_maps(image_mask[:, None].float(), uv.flatten(1, 2))
                valid = valid & (sampled_mask[..., 0].reshape_as(valid) > 0.999)
            return {
                "uv": uv,
                "optical_depth": torch.nan_to_num(optical_depth),
                "valid": valid,
                "depth_derivative": depth_derivative,
                "pixel_derivative": pixel_derivative,
            }

    def world_points(self, depths):
        """仅场景交互或诊断需要时生成 [D,P,3] rig 点，不在投影中构造。"""
        return self.rig_center + depths[..., None] * self.rig_direction[None]


def compile_projective_ruler(uv, K_ref, T_ref, K_source, T_source):
    """uv [P,2]；外参均为 T_cam_from_rig；不改变标定或深度范围。"""
    with torch.autocast(device_type=uv.device.type, enabled=False):
        dtype = torch.float64 if uv.dtype == torch.float64 else torch.float32
        uv = uv.to(dtype)
        K_ref, T_ref = K_ref.to(dtype), T_ref.to(dtype)
        K_source, T_source = K_source.to(dtype), T_source.to(dtype)
        inverse = torch.linalg.inv(K_ref)
        homogeneous = torch.cat((uv, torch.ones_like(uv[:, :1])), -1)
        camera_ray = homogeneous @ inverse.T
        rig_direction = camera_ray @ T_ref[:3, :3]
        rig_center = -T_ref[:3, :3].T @ T_ref[:3, 3]
        source_rotation = T_source[:, :3, :3]
        camera_direction = torch.einsum("nij,pj->npi", source_rotation, rig_direction)
        translation = (source_rotation @ rig_center + T_source[:, :3, 3])
        projected_direction = torch.einsum("nij,npj->npi", K_source[:, :2], camera_direction)
        projected_translation = (K_source[:, :2] @ translation[..., None])[..., 0]
        rig_pixel_derivative = T_ref[:3, :3].T @ inverse[:, :2]
        camera_pixel_derivative = source_rotation @ rig_pixel_derivative
        projected_pixel_derivative = K_source[:, :2] @ camera_pixel_derivative
        depth_numerator = (projected_direction * translation[:, None, 2:3]
                           - projected_translation[:, None] * camera_direction[..., 2:3])
        return ProjectiveRuler(
            rig_direction, rig_center, camera_direction, translation,
            projected_direction, projected_translation, camera_pixel_derivative,
            projected_pixel_derivative, depth_numerator,
        )
