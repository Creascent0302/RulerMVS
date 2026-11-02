"""诊断分层的解析几何验证，不测试/修改生产模型。"""

import torch

from research.v7_domain_20261102.diagnosis import strata


def test_parallel_camera_log_depth_sensitivity():
    intrinsic = torch.tensor([[20., 0., 4.], [0., 20., 4.], [0., 0., 1.]])
    transform = torch.eye(4).repeat(1, 2, 1, 1)
    transform[0, 1, 0, 3] = -.4
    depth = torch.full((1, 2, 9, 9), 4.)
    inputs = {'rgb': torch.zeros(1, 2, 3, 9, 9), 'K': intrinsic.repeat(1, 2, 1, 1),
              'T_cam_from_rig': transform, 'view_mask': torch.ones(1, 2, dtype=torch.bool)}
    batch = {'inputs': inputs, 'targets': {'depth': depth, 'depth_valid': torch.ones_like(depth, dtype=torch.bool)}}
    mask = torch.zeros(1, 1, 9, 9, dtype=torch.bool)
    mask[..., 2:7, 3:7] = True
    domain = {'reference_index': 0, 'geometry_units': 'meters'}
    masks, metadata = strata(batch, depth[:, :1].double(), mask, domain)
    # fB/Z = 2 pixels；浮点外参只允许小数值容差。
    torch.testing.assert_close(torch.tensor(metadata['sensitivity_quantiles_px_per_logZ']), torch.full((3,), 2.))
    assert masks['gt_source_consistent'].sum() == mask.sum()
    assert not masks['gt_source_occluded'].any()
    assert masks['texture_low'].sum() == mask.sum()
    inputs['T_cam_from_rig'][0, 1, 0, 3] = 0
    masks, metadata = strata(batch, depth[:, :1].double(), mask, domain)
    assert metadata['sensitivity_quantiles_px_per_logZ'] == [0., 0., 0.]
    assert masks['parallax_lt05'].sum() == mask.sum()
    inputs['view_mask'][0, 1] = False
    masks, _ = strata(batch, depth[:, :1].double(), mask, domain)
    assert masks['no_source_projection'].sum() == mask.sum()
