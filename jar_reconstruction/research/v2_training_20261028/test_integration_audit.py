"""CPU审计训练入口的真实损失组合和teacher调用；不读取数据或权重。"""

import copy
import json
from pathlib import Path
import unittest

import torch

from jar.pretraining.losses import depth_objective, masked_mean
from .objectives import fine_actual_regret
from .training import V2Hooks, V2TrainingModel


def example():
    inputs = {
        'rgb': torch.ones(1, 1, 3, 8, 8),
        'image_mask': torch.ones(1, 1, 8, 8, dtype=torch.bool),
        'view_mask': torch.ones(1, 1, dtype=torch.bool),
    }
    targets = {'depth': torch.full((1, 1, 8, 8), 5.),
               'depth_valid': torch.ones(1, 1, 8, 8, dtype=torch.bool)}
    prediction = {'depth': torch.full((1, 1, 8, 8), 4.8, requires_grad=True),
                  'unit_scale': 1.}
    for name, size in (('coarse', 1), ('fine', 2)):
        sampling = torch.tensor([3., 5., 9.]).reshape(1, 3, 1, 1).expand(1, 3, size, size)
        actual = torch.tensor([4.9, 5.5, 8.]).reshape(1, 3, 1, 1).expand_as(sampling)
        logits = torch.tensor([.3, .1, -.2]).reshape(1, 3, 1, 1).repeat(1, 1, size, size)
        logits.requires_grad_(True)
        prediction[name] = {
            'sampling_depth': sampling, 'candidate_depth': actual,
            'scores': logits, 'probability': logits.softmax(1),
            'valid': torch.ones_like(sampling, dtype=torch.bool),
            'ref_valid': torch.ones(1, 1, size, size, dtype=torch.bool),
            'lower': torch.full_like(sampling, 1.), 'upper': torch.full_like(sampling, 20.),
        }
    return prediction, inputs, targets


class TrainingAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = json.loads((Path(__file__).parent / 'configs/train_public.json').read_text())

    def test_actual_risk_replaces_only_legacy_fine_risk(self):
        hooks = V2Hooks(self.config)
        prediction, inputs, targets = example()
        domain = {'near': 1., 'far': 20., 'depth_scale': 1., 'geometry_units': 'meters'}
        legacy = depth_objective(prediction, inputs, targets, domain, {})['total']
        changed = hooks.objective(prediction, inputs, targets, domain, {'actual_fine_risk': True})['total']
        stage = prediction['fine']
        truth = targets['depth'][..., ::4, ::4]
        distance = (stage['sampling_depth'].log() - truth.log()).abs().clamp_max(4.)
        old_risk = masked_mean((stage['probability'] * distance).sum(1), stage['ref_valid'][:, 0])
        actual_distance = (stage['candidate_depth'].detach().log() - truth.log()).abs().clamp_max(4.)
        new_risk = masked_mean((stage['probability'] * actual_distance).sum(1), stage['ref_valid'][:, 0])
        expected = legacy + .1 * (new_risk - old_risk)
        torch.testing.assert_close(changed, expected, rtol=1e-6, atol=1e-7)
        parameters = [prediction[name]['scores'] for name in ('coarse', 'fine')]
        expected_gradient = torch.autograd.grad(expected, parameters, retain_graph=True)
        actual_gradient = torch.autograd.grad(changed, parameters)
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-7)
        terms = fine_actual_regret(prediction, inputs, targets, domain)
        torch.testing.assert_close(terms['old_risk'], old_risk, rtol=1e-6, atol=1e-7)

    def test_teacher_runs_full_input_with_nominal_range_and_no_gradient(self):
        hooks = V2Hooks(self.config)
        prediction, inputs, targets = example()
        prediction['_teacher_work'] = object()  # 若误复用学生特征，完整teacher调用断言会失败。

        class Teacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(5.))
                self.calls = []

            def forward(self, received, domain):
                self.calls.append((received, domain, torch.is_grad_enabled()))
                return {'depth': received['rgb'][:, :, 0] * self.weight}

        teacher = Teacher()
        hooks.get_teacher = lambda device: teacher
        domain = {'near': 2., 'far': 40., 'nominal_near': 1., 'nominal_far': 20.,
                  'depth_scale': 1., 'geometry_units': 'meters'}
        terms = hooks.objective(prediction, inputs, targets, domain,
                                {'actual_fine_risk': True, 'excess_error_weight': .2})
        terms['total'].backward()
        self.assertEqual(len(teacher.calls), 1)
        received, nominal, gradient_enabled = teacher.calls[0]
        self.assertIs(received, inputs)
        self.assertEqual((nominal['near'], nominal['far']), (1., 20.))
        self.assertFalse(gradient_enabled)
        self.assertIsNone(teacher.weight.grad)
        self.assertIsNotNone(prediction['depth'].grad)

    def test_test_reading_flags_are_rejected(self):
        for option in ('zero_shot_before_training', 'transfer', 'test_after_training'):
            config = copy.deepcopy(self.config)
            if option == 'test_after_training':
                config['evaluation'][option] = True
            else:
                config[option] = True
            with self.assertRaisesRegex(ValueError, 'TEST'):
                V2Hooks(config)

    def test_stage_prefixes_train_complete_fusion_and_keep_shared_norm_frozen(self):
        model = V2TrainingModel.__new__(V2TrainingModel)
        torch.nn.Module.__init__(model)
        model.config = self.config
        model.foundation = torch.nn.Module()
        model.foundation.model = torch.nn.Module()
        model.foundation.model.blocks = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(12))
        model.foundation.model.norm = torch.nn.LayerNorm(2)
        model.matching = torch.nn.ModuleDict({'backbone': torch.nn.Linear(2, 2),
                                             'lateral': torch.nn.Linear(2, 2)})
        model.decoder = torch.nn.ModuleDict({'raw_adapters': torch.nn.Linear(2, 2),
                                            'fine_inherit': torch.nn.Linear(2, 2),
                                            'score': torch.nn.Linear(2, 2)})
        for stage in self.config['stages']:
            model.set_stage(stage['number'])
            self.assertTrue(all(parameter.requires_grad for parameter in model.decoder.parameters()))
            self.assertTrue(all(not parameter.requires_grad for parameter in model.foundation.model.norm.parameters()))
            count = stage['foundation_blocks']
            for index, block in enumerate(model.foundation.model.blocks):
                expected = count > 0 and index >= 12 - count
                self.assertTrue(all(parameter.requires_grad == expected for parameter in block.parameters()))
            expected_matching = 'matching.' in stage['trainable_prefixes']
            self.assertTrue(all(parameter.requires_grad == expected_matching for parameter in model.matching.parameters()))


if __name__ == '__main__':
    unittest.main()
