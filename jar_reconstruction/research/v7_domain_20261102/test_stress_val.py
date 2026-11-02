"""压力输入只删除实际源并保持标定，灰度不读取目标标签。"""

import torch

from research.v7_domain_20261102.stress_val import changed_input


def sample():
    transforms = torch.eye(4).repeat(1, 4, 1, 1)
    transforms[0, :, 0, 3] = torch.tensor([0., .6, .1, .3])
    return {'rgb': torch.rand(1, 4, 3, 8, 8), 'matching_rgb': torch.rand(1, 4, 3, 16, 16),
            'view_mask': torch.ones(1, 4, dtype=torch.bool),
            'image_mask': torch.ones(1, 4, 8, 8, dtype=torch.bool),
            'matching_image_mask': torch.ones(1, 4, 16, 16, dtype=torch.bool),
            'K': torch.eye(3).repeat(1, 4, 1, 1), 'T_cam_from_rig': transforms}


def test_true_pair_selection_and_gray_preserve_calibration():
    inputs = sample()
    original = {key: value.clone() for key, value in inputs.items()}
    output, selected = changed_input(inputs, 0, 'gray_closest_pair')
    assert selected[0]['closest_source'] == 2
    assert output['view_mask'].tolist() == [[True, False, True, False]]
    assert torch.equal(output['rgb'][:, :, 0], output['rgb'][:, :, 1])
    assert torch.equal(output['K'], inputs['K'])
    assert torch.equal(output['T_cam_from_rig'], inputs['T_cam_from_rig'])
    assert not output['image_mask'][:, 1].any()
    assert not output['matching_image_mask'][:, 1].any()
    assert all(torch.equal(inputs[key], value) for key, value in original.items())
    inputs['view_mask'][0, 2] = False
    output, selected = changed_input(inputs, 0, 'closest_real_pair')
    assert selected[0]['closest_source'] == 3
    assert output['view_mask'].tolist() == [[True, False, False, True]]


def test_zero_baseline_not_preferred_and_all_zero_keeps_existing():
    inputs = sample()
    inputs['T_cam_from_rig'][0, 2, 0, 3] = 0
    _, selected = changed_input(inputs, 0, 'closest_real_pair')
    assert selected[0]['closest_source'] == 3
    inputs['T_cam_from_rig'][..., :3, 3] = 0
    output, selected = changed_input(inputs, 0, 'closest_real_pair')
    assert selected[0]['pure_rotation_only_kept_existing']
    assert torch.equal(output['view_mask'], inputs['view_mask'])
