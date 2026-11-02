"""评测适配只读inputs，同像素分支生效，统计不偷删缺失预测。"""
import unittest
import torch

from research.near_mid_evaluation_20261028.evaluate import predict_with_stages
from research.near_mid_evaluation_20261028.metrics import depth_sums, depth_metrics
from tools.benchmark_public_speed import prepare_model


class FakeModels:
    def get(self, name, domain):
        def predict(inputs, domain, reference):
            self.used = inputs
            z = inputs['matching_rgb'].mean().reshape(1, 1, 1, 1)
            return {'depth': z, 'valid': torch.ones_like(z, dtype=torch.bool),
                    'unit_scale': 2., 'coarse': {'depth': z}, 'fine': {'depth': z}}
        return predict

    def convert(self, prediction, inputs):
        return prediction['depth'], prediction['valid']


class ProtocolTest(unittest.TestCase):
    def test_preload_creates_versioned_parameters_inside_inference_context(self):
        class Loader:
            def get(self, name, domain):
                self.domain = domain
                return torch.nn.Linear(2, 2)
        loader = Loader()
        inputs = {'rgb': torch.ones(1, 4, 3, 8, 8),
                  'view_mask': torch.tensor([[True, False, True, False]])}
        domain = {'reference_index': 2, 'num_views': 4}
        with torch.inference_mode():
            model = prepare_model(loader, 'baseline', {'drop_missing_views': True}, domain, inputs)
        self.assertFalse(torch.is_inference(model.weight))
        self.assertEqual(loader.domain['num_views'], 2)
        self.assertEqual(loader.domain['reference_index'], 1)
        self.assertIsInstance(model.weight._version, int)

    def test_equal_pixels_really_changes_matching_input(self):
        inputs = {'rgb': torch.ones(1, 4, 3, 8, 8),
                  'matching_rgb': torch.zeros(1, 4, 3, 16, 16)}
        models = FakeModels()
        a, _, stages = predict_with_stages(models, 'own', {'kind': 'own', 'input_mode': 'equal_pixels'},
                                           {'reference_index': 2}, inputs)
        self.assertEqual(a.item(), 2.)
        self.assertEqual(stages['coarse'].item(), 2.)
        self.assertEqual(inputs['matching_rgb'].sum().item(), 0.)
        b, _, _ = predict_with_stages(models, 'own', {'kind': 'own', 'input_mode': 'native'},
                                      {'reference_index': 2}, inputs)
        self.assertEqual(b.item(), 0.)
        self.assertIs(models.used, inputs)

    def test_global_rmse_uses_global_squared_error(self):
        gt = torch.ones(1, 1, 1, 4)
        pred = torch.tensor([1., 1., 1., 5.]).reshape_as(gt)
        mask = torch.ones_like(gt, dtype=torch.bool)
        value = depth_metrics(depth_sums(pred, mask, gt, mask))
        self.assertEqual(value['MAE'], 1.)
        self.assertEqual(value['RMSE'], 2.)
        self.assertEqual(value['coverage'], 1.)

    def test_invalid_depth_cannot_be_full_gt_mae(self):
        gt = torch.ones(1, 1, 1, 2)
        pred = torch.tensor([1., float('nan')]).reshape_as(gt)
        mask = torch.ones_like(gt, dtype=torch.bool)
        value = depth_metrics(depth_sums(pred, mask, gt, mask))
        self.assertIsNone(value['full_gt_MAE'])
        self.assertEqual(value['coverage'], .5)
        self.assertEqual(value['delta_1'], .5)


if __name__ == '__main__':
    unittest.main()
