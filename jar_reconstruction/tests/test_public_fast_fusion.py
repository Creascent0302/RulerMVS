"""融合代数重排保留函数与训练梯度，不移除任何特征层。"""

import math

import pytest
import torch
from torch import nn

from jar.pretraining.fast_fusion import factorized_visual


@pytest.mark.parametrize("depth", [3, 12])
def test_factorized_visual_outputs_and_gradients(depth):
    torch.manual_seed(41)
    torch.set_num_threads(2)
    query = nn.Conv3d(8, 16, 1)
    output = nn.Conv3d(16, 8, 1)
    features = torch.randn(2, 8, depth, 2, 3, requires_grad=True)
    bank = torch.randn(2, 16, 4, 2, 3, requires_grad=True)
    logits = torch.einsum("bcdhw,bclhw->bldhw", query(features), bank) / math.sqrt(16)
    expected = output(torch.einsum("bldhw,bclhw->bcdhw", logits.softmax(1), bank))
    actual = factorized_visual(features, bank, query, output)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    variables = (features, bank, *query.parameters(), *output.parameters())
    first = torch.autograd.grad(expected.square().mean(), variables, retain_graph=True)
    second = torch.autograd.grad(actual.square().mean(), variables)
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-5)
