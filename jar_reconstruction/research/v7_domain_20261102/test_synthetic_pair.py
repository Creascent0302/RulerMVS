"""有真实相机关系的纹理背景平面/前景遮挡测试。"""

import copy

import torch
from torch.nn import functional as F

from research.v7_domain_20261102.synthetic_pair import (
    correspondence_targets, forward_splat, synthesize_pair,
)


def scene(batch=2):
    height, width, views = 16, 24, 4
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    texture = torch.stack((x / width, y / height, ((x + y) % 2).float()))
    rgb = texture[None, None].expand(batch, views, -1, -1, -1).clone()
    depth = torch.full((batch, views, height, width), 8.)
    depth[:, :, 4:12, 8:16] = 4.
    intrinsic = torch.tensor([[32., 0., 11.5], [0., 32., 7.5], [0., 0., 1.]])
    inputs = {
        "rgb": rgb, "K": intrinsic.repeat(batch, views, 1, 1),
        "T_cam_from_rig": torch.eye(4).repeat(batch, views, 1, 1),
        "view_mask": torch.ones(batch, views, dtype=torch.bool),
        "image_mask": torch.ones(batch, views, height, width, dtype=torch.bool),
        "camera_valid": torch.ones(batch, views, dtype=torch.bool),
        "image_sizes": torch.tensor([height, width]).repeat(batch, views, 1),
        "matching_rgb": F.interpolate(rgb.flatten(0, 1), (32, 48), mode="bilinear",
                                       align_corners=False).unflatten(0, (batch, views)),
        "matching_image_mask": torch.ones(batch, views, 32, 48, dtype=torch.bool),
    }
    targets = {"depth": depth, "depth_valid": torch.ones_like(depth, dtype=torch.bool),
               "semantic": torch.ones_like(depth, dtype=torch.long)}
    return inputs, targets


def test_zbuffer_occlusion_and_unknown_holes():
    inputs, targets = scene()
    result = forward_splat(inputs["rgb"][:, 0], targets["depth"][:, 0],
                           targets["depth_valid"][:, 0], inputs["K"][:, 0],
                           torch.full((2,), -.5))
    # fx*B/Z: 背景移动2px，近前景移动4px；冲突时前景胜。
    assert result["depth"][0, 6, 5] == 4.
    assert result["reference_pixel_index"][0, 6, 5] == 6 * 24 + 9
    assert torch.equal(result["rgb"][0, :, 6, 5], inputs["rgb"][0, 0, :, 6, 9])
    assert result["collision_count"].min() > 0
    # 前景移动后新露出的背景从未被参考相机观察，不填假数据。
    assert not result["valid"][0, 6, 12:14].any()
    assert not result["depth"][0, 6, 12:14].any()
    assert not result["rgb"][0, :, 6, 12:14].any()
    assert (result["reference_pixel_index"][~result["valid"]] == -1).all()


def test_contract_and_explicit_source_supervision():
    inputs, targets = scene()
    before_inputs, before_targets = copy.deepcopy(inputs), copy.deepcopy(targets)
    augmented, supervision, record = synthesize_pair(
        inputs, targets, {"depth_scale": 1.},
        generator=torch.Generator().manual_seed(42), baseline_norm=(.5, .5))
    assert torch.equal(record["retained_views"], torch.tensor([2, 2]))
    assert torch.equal(augmented["rgb"][:, 0], inputs["rgb"][:, 0])
    assert torch.equal(augmented["K"][:, 1], inputs["K"][:, 0])
    assert torch.equal(augmented["T_cam_from_rig"][:, 1, 0, 3], supervision["translation_x"])
    rebuilt = F.interpolate(augmented["rgb"].flatten(0, 1), (32, 48), mode="bilinear",
                             align_corners=False).unflatten(0, (2, 4))
    assert torch.equal(augmented["matching_rgb"], rebuilt)
    assert not augmented["matching_image_mask"][:, 2:].any()
    assert not augmented["rgb"][:, 2:].any()
    updated_targets = correspondence_targets(targets, supervision)
    assert torch.equal(updated_targets["depth"][:, 0], targets["depth"][:, 0])
    assert torch.equal(updated_targets["depth"][:, 1], supervision["source_depth"])
    assert (updated_targets["semantic"][:, 1] == -1).all()
    for key, value in inputs.items():
        assert torch.equal(value, before_inputs[key])
    for key, value in targets.items():
        assert torch.equal(value, before_targets[key])


def test_length_units_and_baseline_independence_of_gt():
    inputs, targets = scene()
    kwargs = {"baseline_norm": (.03, .3)}
    first, source, record = synthesize_pair(
        inputs, targets, {"depth_scale": 1.}, generator=torch.Generator().manual_seed(7), **kwargs)
    scaled_inputs, scaled_targets = copy.deepcopy(inputs), copy.deepcopy(targets)
    scaled_inputs["T_cam_from_rig"][:, :, :3, 3] *= 100.
    scaled_targets["depth"] *= 100.
    scaled, scaled_source, _ = synthesize_pair(
        scaled_inputs, scaled_targets, {"depth_scale": 100.},
        generator=torch.Generator().manual_seed(7), **kwargs)
    assert torch.equal(first["rgb"], scaled["rgb"])
    assert torch.equal(first["image_mask"], scaled["image_mask"])
    torch.testing.assert_close(source["source_depth"] * 100., scaled_source["source_depth"])
    changed_targets = copy.deepcopy(targets)
    changed_targets["depth"] *= 2.
    _, _, changed_record = synthesize_pair(
        inputs, changed_targets, {"depth_scale": 1.},
        generator=torch.Generator().manual_seed(7), **kwargs)
    assert torch.equal(record["baseline_data_units"], changed_record["baseline_data_units"])


def test_projection_roundtrip_and_zero_probability():
    inputs, targets = scene()
    unchanged, _, record = synthesize_pair(
        inputs, targets, {"depth_scale": 1.}, generator=torch.Generator().manual_seed(3), probability=0.)
    assert not record["active"].any()
    for key, value in inputs.items():
        assert torch.equal(value, unchanged[key])
    # 0.13基线产生亚像素投影，连续往返与栅格误差分别检验。
    result = forward_splat(inputs["rgb"][:, 0], targets["depth"][:, 0],
                           targets["depth_valid"][:, 0], inputs["K"][:, 0], torch.full((2,), .13))
    uv = result["reference_projected_uv"]
    height, width = targets["depth"].shape[-2:]
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    expected_u = x[None] + 32. * .13 / targets["depth"][:, 0]
    torch.testing.assert_close(uv[:, 0], expected_u, atol=2e-6, rtol=0.)
    torch.testing.assert_close(uv[:, 1], y[None].expand(2, -1, -1).float(), atol=2e-6, rtol=0.)
    rendered_uv = result["projected_uv"]
    quantization = ((rendered_uv[:, 0] - x) ** 2 + (rendered_uv[:, 1] - y) ** 2).sqrt()
    assert quantization[result["valid"]].max() <= 2 ** -.5 + 1e-6
