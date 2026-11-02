"""预登记物理干预的数学检查，不需要GPU或模型权重。"""

import unittest

import numpy as np
import torch

from research.paper_evidence_20261101.physics import alter, confidence_statistics, rotation_y


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        transform = torch.eye(4, dtype=torch.float64).repeat(1, 4, 1, 1)
        transform[0, :, 0, 3] = torch.tensor([-.7, .7, 0., 0.])
        self.inputs = {
            'T_cam_from_rig': transform,
            'K': torch.eye(3, dtype=torch.float64).repeat(1, 4, 1, 1),
            'rgb': torch.zeros(1, 4, 3, 8, 8),
            'view_mask': torch.ones(1, 4, dtype=torch.bool),
        }
        self.domain = {'reference_index': 2, 'near': 2., 'far': 80., 'depth_scale': 3.}

    def test_unit_conversion_cancels_before_geometry(self):
        work, domain, scale = alter(self.inputs, self.domain, {'kind': 'units', 'scale': 2.})
        self.assertEqual(scale, 2.)
        torch.testing.assert_close(work['T_cam_from_rig'][..., :3, 3] / domain['depth_scale'],
                                   self.inputs['T_cam_from_rig'][..., :3, 3] / self.domain['depth_scale'],
                                   rtol=0, atol=0)
        self.assertEqual(domain['far'] / domain['depth_scale'], self.domain['far'] / self.domain['depth_scale'])

    def test_rig_change_preserves_camera_points(self):
        work, _, _ = alter(self.inputs, self.domain, {'kind': 'rig'})
        change = torch.eye(4, dtype=torch.float64)
        change[:3, :3] = rotation_y(23, change)
        change[:3, 3] = change.new_tensor([1.3, -.7, .4])
        point = change.new_tensor([2., -.3, 12., 1.])
        torch.testing.assert_close(work['T_cam_from_rig'] @ (change @ point),
                                   self.inputs['T_cam_from_rig'] @ point, rtol=0, atol=1e-12)

    def test_source_yaw_keeps_camera_center(self):
        work, _, _ = alter(self.inputs, self.domain, {'kind': 'source_yaw', 'degrees': 1.})
        torch.testing.assert_close(torch.linalg.inv(work['T_cam_from_rig'])[..., :3, 3],
                                   torch.linalg.inv(self.inputs['T_cam_from_rig'])[..., :3, 3],
                                   rtol=0, atol=1e-12)
        torch.testing.assert_close(work['T_cam_from_rig'][:, 1:], self.inputs['T_cam_from_rig'][:, 1:], rtol=0, atol=0)
        self.assertIs(work['rgb'], self.inputs['rgb'])

    def test_confidence_ranking_and_oracle(self):
        good = confidence_statistics([np.array([.9, .5, .1])], [np.array([0., 1., 2.])])
        bad = confidence_statistics([np.array([.1, .5, .9])], [np.array([0., 1., 2.])])
        self.assertAlmostEqual(good['AURC'], .5)
        self.assertAlmostEqual(good['excess_AURC'], 0.)
        self.assertGreater(bad['excess_AURC'], 0.)
        self.assertEqual(good['risk_AbsRel'][-1], good['random_expected_risk'])


if __name__ == '__main__':
    unittest.main()
