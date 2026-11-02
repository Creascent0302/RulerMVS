"""只检验薄适配边界，不把mock结果当作基线网络前向测试。"""

import unittest
import multiprocessing as mp
from unittest.mock import patch

import torch

from research.rulermvs_v6_20261101.evaluation import AccuracyModels, Models, NativeModels


def check_spawn_registration(queue):
    from research.near_mid_evaluation_20261028 import evaluate
    from research.rulermvs_v6_20261101.evaluation import Models

    queue.put(evaluate.Models is Models)


class EvaluationAdapterTests(unittest.TestCase):
    def test_worker_registration_survives_spawn(self):
        context = mp.get_context('spawn')
        queue = context.Queue()
        process = context.Process(target=check_spawn_registration, args=(queue,))
        process.start()
        try:
            self.assertTrue(queue.get(timeout=30))
            process.join(timeout=30)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            queue.close()

    def inputs(self, mask):
        return {
            'rgb': torch.arange(4.).reshape(1, 4, 1, 1, 1).expand(1, 4, 3, 8, 8),
            'K': torch.eye(3).repeat(1, 4, 1, 1),
            'T_cam_from_rig': torch.eye(4).repeat(1, 4, 1, 1),
            'view_mask': torch.tensor([mask]),
        }

    def model(self, **kwargs):
        return Models({'methods': {'patch': {'kind': 'patchmatchnet'}}}, 'cpu', **kwargs)

    def test_compacts_real_views_and_restores_reference(self):
        inputs = self.inputs([True, False, True, True])
        with patch.object(AccuracyModels, 'forward', autospec=True) as forward:
            forward.return_value = {'depth': torch.ones(1, 1, 8, 8), 'reference_indices': [1]}
            output = self.model().forward('patch', {'num_views': 4, 'reference_index': 2}, inputs)
            _, _, domain, received = forward.call_args.args
        self.assertEqual(domain['num_views'], 3)
        self.assertEqual(domain['reference_index'], 1)
        self.assertTrue(torch.equal(received['rgb'], inputs['rgb'][:, [0, 2, 3]]))
        self.assertEqual(output['reference_indices'], [2])

    def test_unsupported_keeps_grid_and_zero_coverage(self):
        inputs = self.inputs([False, False, True, False])
        with patch.object(AccuracyModels, 'forward', autospec=True) as forward:
            output = self.model().forward('patch', {'num_views': 4, 'reference_index': 2}, inputs)
        forward.assert_not_called()
        self.assertEqual(output['depth'].shape, (1, 1, 8, 8))
        self.assertFalse(bool(output['valid'].any()))
        self.assertIn('unsupported_reason', output)

    def test_speed_uses_native_without_hash_wrapper(self):
        inputs = self.inputs([True] * 4)
        with patch.object(NativeModels, 'forward', autospec=True) as native:
            native.return_value = {'depth': torch.ones(1, 1, 8, 8)}
            with patch.object(AccuracyModels, 'forward', autospec=True) as audited:
                self.model(fixed_random_stream=False).forward(
                    'patch', {'num_views': 4, 'reference_index': 2}, inputs)
                audited.assert_not_called()
        self.assertIs(native.call_args.args[-1], inputs)


if __name__ == '__main__':
    unittest.main()
