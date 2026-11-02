"""新增强只改变视觉输入和可见槽，不改变校准、不激活缺失视图。"""

import pytest
import torch

from research.v7_domain_20261102.augmentation import augment_inputs


def sample(views=4):
    torch.manual_seed(81)
    mask = torch.ones(3, views, dtype=torch.bool)
    if views > 2:
        mask[1, -1] = False
    mask[2, 1:] = False
    return {
        "rgb": torch.rand(3, views, 3, 4, 6) * mask[:, :, None, None, None],
        "matching_rgb": torch.rand(3, views, 3, 8, 12) * mask[:, :, None, None, None],
        "view_mask": mask,
        "image_mask": mask[:, :, None, None].expand(-1, -1, 4, 6).clone(),
        "matching_image_mask": mask[:, :, None, None].expand(-1, -1, 8, 12).clone(),
        "camera_valid": mask.clone(),
        "image_sizes": torch.tensor([4, 6]).expand(3, views, -1).clone() * mask[:, :, None],
        "K": torch.eye(3).expand(3, views, -1, -1).clone(),
        "T_cam_from_rig": torch.eye(4).expand(3, views, -1, -1).clone(),
    }


def test_disabled_and_no_inplace():
    inputs = sample()
    before = {key: value.clone() for key, value in inputs.items()}
    output, record = augment_inputs(inputs, grayscale_probability=0,
                                    pair_probability=0, triple_probability=0)
    for key in before:
        assert torch.equal(inputs[key], before[key])
        assert torch.equal(output[key], before[key])
    assert not record["grayscale"].any()


def test_grayscale_same_operator_and_geometry_unchanged():
    inputs = sample()
    output, _ = augment_inputs(inputs, grayscale_probability=1,
                               pair_probability=0, triple_probability=0)
    for name in ("rgb", "matching_rgb"):
        image = output[name]
        expected = (inputs[name] * image.new_tensor([.299, .587, .114])[None, None, :, None, None]).sum(2)
        assert torch.allclose(image[:, :, 0], expected)
        assert torch.equal(image[:, :, 0], image[:, :, 1])
        assert torch.equal(image[:, :, 1], image[:, :, 2])
    for name in ("K", "T_cam_from_rig", "view_mask", "image_mask"):
        assert torch.equal(output[name], inputs[name])


@pytest.mark.parametrize("views", [2, 3, 4])
def test_pair_keeps_reference_and_one_real_source(views):
    inputs = sample(views)
    output, _ = augment_inputs(inputs, grayscale_probability=0,
                               pair_probability=1, triple_probability=0)
    assert torch.equal(output["view_mask"][:, 0], inputs["view_mask"][:, 0])
    assert output["view_mask"].sum(1).tolist() == [2, 2, 1]
    assert not (output["view_mask"] & ~inputs["view_mask"]).any()
    dropped = ~output["view_mask"]
    for name in ("rgb", "matching_rgb", "image_mask", "matching_image_mask", "image_sizes", "camera_valid"):
        assert not output[name][dropped].any()


def test_reproducible_cpu_rng_and_no_targets_required():
    inputs = sample()
    outputs = [augment_inputs(inputs, generator=torch.Generator().manual_seed(9))[0] for _ in range(2)]
    for key in inputs:
        assert torch.equal(outputs[0][key], outputs[1][key])


def test_probabilities():
    with pytest.raises(ValueError):
        augment_inputs(sample(), pair_probability=.8, triple_probability=.3)
