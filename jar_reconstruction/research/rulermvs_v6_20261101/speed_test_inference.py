"""部署准备的冻结约束、幂等性、序列化接口；不把mock编译当GPU验证。"""

import tempfile
import unittest
from unittest.mock import patch

import torch

from research.rulermvs_v5_20261031.test_bridge import tiny_pair
from research.rulermvs_v6_20261101.inference import prepare_fast_inference


class InferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        _, self.model = tiny_pair()

    def test_requires_frozen_eval(self):
        with self.assertRaisesRegex(ValueError, 'requires_grad'):
            prepare_fast_inference(self.model)
        self.model.requires_grad_(False).train()
        with self.assertRaisesRegex(ValueError, 'model.eval'):
            prepare_fast_inference(self.model)

    def test_idempotent_and_state_unchanged(self):
        model = self.model.eval().requires_grad_(False)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        self.assertIs(prepare_fast_inference(model), model)
        self.assertIs(prepare_fast_inference(model), model)
        self.assertEqual(set(before), set(model.state_dict()))
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in before.items()))
        self.assertFalse(model.decoder.settings['projective_ruler'])
        with self.assertRaisesRegex(ValueError, '重新加载'):
            prepare_fast_inference(model, {'compile_source': True})

    def test_compile_install_once_and_keys_preserved(self):
        model = self.model.eval().requires_grad_(False)
        keys = tuple(model.state_dict())
        with tempfile.TemporaryDirectory() as cache:
            settings = {'compile_source': True, 'compile_fusion': True, 'cache_dir': cache}
            with patch('torch.compile', side_effect=lambda method, **options: method) as compile_call:
                prepare_fast_inference(model, settings)
                prepare_fast_inference(model, settings)
                self.assertEqual(compile_call.call_count, 2)
                self.assertTrue(all(call.kwargs == {'fullgraph': True, 'dynamic': False}
                                    for call in compile_call.call_args_list))
            self.assertEqual(keys, tuple(model.state_dict()))


if __name__ == '__main__':
    unittest.main()
