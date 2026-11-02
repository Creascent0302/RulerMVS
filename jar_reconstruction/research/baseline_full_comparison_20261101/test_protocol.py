"""随机流隔离和多卡验证BN一致性的回归检查。"""

import copy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from jar.runtime import Runtime
from jar.synthetic import make_synthetic_batch
from official_methods.registry import build_adapter
from official_methods.tests.test_training_validation import configuration
from official_methods.training_validation import validate_depth
from research.baseline_full_comparison_20261101.evaluate import Models


@pytest.mark.parametrize("method", ["patchmatchnet", "diffmvs"])
def test_randomness_is_preserved_but_independent_of_loading_order(method, tmp_path):
    torch.set_num_threads(2)
    cfg = configuration(method)
    native = build_adapter(cfg)
    path = tmp_path / "weights.pt"
    torch.save({"method": method, "model": native.state_dict()}, path)
    entry = {"kind": method, "weights": str(path), "project_checkpoint": True, "sample_seed": 1701}
    settings = {"methods": {"a": entry, "b": dict(entry, sample_seed=2701), "same": copy.deepcopy(entry)}}
    domain = {"near": 2., "far": 80., "reference_index": 0}
    inputs = make_synthetic_batch(N=4, H=64, W=96)["inputs"]
    models = Models(settings, torch.device("cpu"))
    for name in settings["methods"]:
        models.get(name, domain)
    state = torch.get_rng_state().clone()
    with torch.inference_mode():
        first = models.forward("a", domain, inputs)["depth"]
        assert torch.equal(state, torch.get_rng_state())
        other = models.forward("b", domain, inputs)["depth"]
        torch.randn(100)
        repeat = models.forward("same", domain, inputs)["depth"]
    assert torch.equal(first, repeat)
    assert not torch.equal(first, other)  # 未关闭官方随机采样/噪声。


class BufferProbe(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.register_buffer("running_mean", torch.tensor(float(rank + 1)))

    def forward(self, inputs):
        assert self.running_mean.item() == 1  # 所有rank必须用保存checkpoint的rank0状态。
        depth = torch.ones_like(inputs["rgb"][:, :1, 0]) * (self.running_mean + 5)
        return {"depth": depth, "valid": depth > 0, "reference_indices": [0]}


def validation_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        runtime = Runtime(torch.device("cpu"), rank, 2, rank, True, rank == 0)
        model = BufferProbe(rank).train()
        result = validate_depth(model, configuration("patchmatchnet"), runtime, synthetic=True)
        assert result["frames"] == 2
        assert model.training and model.running_mean.item() == rank + 1
    finally:
        dist.destroy_process_group()


def test_distributed_validation_uses_one_checkpoint_and_restores_local_state(tmp_path):
    mp.spawn(validation_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
