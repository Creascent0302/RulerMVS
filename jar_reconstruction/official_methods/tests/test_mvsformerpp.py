"""完整官方网络的四目适配、空监督、冻结骨干与几何格点验证。"""

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import torch

from official_methods.data import FourCameraDataset, collate
from official_methods.methods.mvsformerpp import Adapter


ROOT = Path(__file__).resolve().parents[1] / "methods" / "mvsformerpp"


@pytest.fixture(scope="module")
def model_and_batch():
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads((ROOT / "configs" / "synthetic.json").read_text())
    device = torch.device(os.environ.get("MVSFORMERPP_TEST_DEVICE", "cpu"))
    model = Adapter(config).to(device)
    batch = collate([FourCameraDataset(config, "train", synthetic=True)[0]])
    for key, value in batch["inputs"].items():
        if isinstance(value, torch.Tensor):
            batch["inputs"][key] = value.to(device)
    for key, value in batch["targets"].items():
        if isinstance(value, torch.Tensor):
            batch["targets"][key] = value.to(device)
    return model, batch


def test_full_architecture_and_native_optimizer(model_and_batch):
    model, _ = model_and_batch
    assert len(model.native.vit.blocks) == 12
    assert model.native.vit.embed_dim == 768
    assert len(model.native.fusions[0].cost_reg.attention_layers) == 6
    assert model.native.ndepths == [32, 16, 8, 4]
    assert not any(parameter.requires_grad for parameter in model.native.vit.parameters())
    optimizer = model.configure_optimizer()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["weight_decay"] == 0
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert expected == actual


def test_native_outputs_and_no_label_inference(model_and_batch):
    model, batch = model_and_batch
    model.eval()
    with torch.no_grad():
        prepared = model.prepare_native(batch["inputs"], 2)
        direct = model.call_native(prepared)
        output = model(batch["inputs"])
        changed_targets = deepcopy(batch["targets"])
        changed_targets["depth"].fill_(1234)
        repeated = model(batch["inputs"])
    torch.testing.assert_close(output["depth"][:, 0], direct["refined_depth"], rtol=0, atol=0)
    torch.testing.assert_close(output["confidence"][:, 0], direct["photometric_confidence"], rtol=0, atol=0)
    torch.testing.assert_close(output["depth"], repeated["depth"], rtol=0, atol=0)
    assert output["depth"].shape == (1, 1, 64, 64)
    assert torch.isfinite(output["depth"]).all()
    torch.testing.assert_close(output["depth_K"], batch["inputs"]["K"][:, [2]])


def test_all_stages_backward_and_empty_supervision(model_and_batch):
    model, batch = model_and_batch
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(batch["inputs"])
    losses = model.loss(output, batch["inputs"], batch["targets"])
    assert set(losses) == {"stage1", "stage2", "stage3", "stage4", "total"}
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    for name in ("encoder", "decoder_vit", "FMT_module", "fusions"):
        params = getattr(model.native, name).parameters()
        grads = [p.grad for p in params if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
        assert any(torch.count_nonzero(grad) for grad in grads)
    assert all(p.grad is None for p in model.native.vit.parameters())
    model.zero_grad(set_to_none=True)
    output = model(batch["inputs"])
    empty = deepcopy(batch["targets"])
    empty["depth_valid"].zero_()
    loss = model.loss(output, batch["inputs"], empty)["total"]
    assert loss.item() == 0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_native_camera_lattices(model_and_batch):
    model, batch = model_and_batch
    inputs = batch["inputs"]
    _, projections, depths = model.prepare_native(inputs, 2)
    for stage, stride in enumerate((8, 4, 2, 1), 1):
        camera = projections[f"stage{stage}"]
        expected = inputs["K"][:, 2].clone()
        expected[:, :2] /= stride
        torch.testing.assert_close(camera[:, 0, 1, :3, :3], expected)
        torch.testing.assert_close(camera[:, 0, 0], inputs["T_cam_from_rig"][:, 2])
    assert depths.shape == (1, 192)
    assert depths[0, 0] == 2 and depths[0, -1] == 80


def test_reject_missing_view(model_and_batch):
    model, batch = model_and_batch
    broken = deepcopy(batch["inputs"])
    broken["view_mask"][:, 0] = False
    with pytest.raises(ValueError, match="缺失相机"):
        model.prepare_native(broken, 2)


def test_eval_batch_two_matches_individual(model_and_batch):
    model, batch = model_and_batch
    model.eval()
    inputs = {
        name: value.repeat(2, *([1] * (value.ndim - 1))) if isinstance(value, torch.Tensor) else value
        for name, value in batch["inputs"].items()
    }
    with torch.no_grad():
        individual = model(batch["inputs"])["depth"]
        combined = model(inputs)["depth"]
    torch.testing.assert_close(combined, individual.repeat(2, 1, 1, 1), rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("views", [2, 3])
def test_native_variable_camera_count(model_and_batch, views):
    """公共双目域使用真实N=2；缺图压缩后的N=3也不复制相机。"""
    model, batch = model_and_batch
    inputs = {
        name: value[:, :views] if torch.is_tensor(value) and value.ndim > 1
        and value.shape[1] == 4 else value
        for name, value in batch["inputs"].items()
    }
    model.eval()
    with torch.no_grad():
        native = model.call_native(model.prepare_native(inputs, 0))
    assert native["refined_depth"].shape == (1, 64, 64)
    assert torch.isfinite(native["refined_depth"]).all()


def test_native_checkpoint_cache_filter_keeps_strict_loading(model_and_batch, monkeypatch):
    """作者测试入口丢弃旧pe_dict缓存，但真实缺失参数必须报错。"""
    model, _ = model_and_batch
    state = {"module." + name: value for name, value in model.native.state_dict().items()}
    state["module.legacy.pe_dict.64-64"] = torch.zeros(1)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"state_dict": state})
    model.load_native_weights("mock_author_checkpoint.pth")
    state.pop(next(name for name in state if "pe_dict" not in name))
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_native_weights("mock_incomplete_checkpoint.pth")
