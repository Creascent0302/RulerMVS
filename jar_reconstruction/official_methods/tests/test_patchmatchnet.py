"""真实官方PatchmatchNet图的CPU验证；数值测试不是精度评价。"""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from jar.synthetic import make_synthetic_batch
from official_methods.methods.patchmatchnet import Adapter, UPSTREAM
from official_methods.methods.patchmatchnet.adapter import native_module


@pytest.fixture
def sample():
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    return make_synthetic_batch(B=1, N=4, H=32, W=64, seed=1701)


def config():
    path = UPSTREAM.parent / "configs/synthetic.json"
    value = json.loads(path.read_text())
    value["evaluation"]["reference_indices"] = [0]
    return value


@pytest.mark.parametrize("views", [2, 3, 4])
def test_real_graph_backward_and_native_loss(sample, views):
    inputs = {key: value[:, :views] for key, value in sample["inputs"].items()}
    model = Adapter(config()).train()
    output = model(inputs)
    losses = model.loss(output, inputs, sample["targets"])
    labels, masks = model.prepare_targets(sample["targets"], 0)
    native = output["native"][0]
    expected = native_module().patchmatchnet_loss(
        native["depth_patchmatch"], native["refined_depth"], labels, masks
    )
    assert torch.equal(losses["total"], expected)
    losses["total"].backward()
    for group in ("feature", "patchmatch_3", "patchmatch_2", "patchmatch_1", "upsample_net"):
        gradients = [p.grad for name, p in model.native.named_parameters()
                     if name.startswith(group) and p.grad is not None]
        assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
        assert any(grad.abs().sum() > 0 for grad in gradients)
    optimizer, _ = model.configure_optimizer()
    optimizer.step()
    assert output["depth"].shape == (1, 1, 32, 64)


def test_projection_and_original_preprocessing(sample):
    model = Adapter(config())
    images, projections, _, _ = model.prepare_native(sample["inputs"], 2)
    order = [2, 0, 1, 3]
    for stage in range(4):
        stride = 2 ** stage
        for camera, original in enumerate(order):
            rgb = sample["inputs"]["rgb"][0, original].permute(1, 2, 0).numpy()
            expected = cv2.resize(rgb, (64 // stride, 32 // stride), interpolation=cv2.INTER_LINEAR)
            np.testing.assert_allclose(images[f"stage_{stage}"][0, camera].permute(1, 2, 0),
                                       expected, atol=1e-6)
            intrinsic = sample["inputs"]["K"][0, original].clone()
            intrinsic[:2] /= stride
            expected_projection = intrinsic @ sample["inputs"]["T_cam_from_rig"][0, original, :3]
            torch.testing.assert_close(projections[f"stage_{stage}"][0, camera, :3], expected_projection)


def test_parity_checkpoint_and_no_label_leakage(sample, tmp_path):
    cfg = config()
    cfg["model"]["weights"] = str(UPSTREAM / "checkpoints/model_000007.ckpt")
    model = Adapter(cfg).eval()
    with torch.no_grad():
        torch.manual_seed(19)
        output = model(sample["inputs"])
        torch.manual_seed(19)
        expected = model.native(*model.prepare_native(sample["inputs"], 0))
        assert torch.equal(output["depth"], expected["refined_depth"]["stage_0"])
        assert torch.equal(output["confidence"][:, 0], expected["photometric_confidence"])
        changed = {"depth": sample["targets"]["depth"] * 10,
                   "depth_valid": sample["targets"]["depth_valid"]}
        model.loss(output, sample["inputs"], changed)
        torch.manual_seed(19)
        assert torch.equal(output["depth"], model(sample["inputs"])["depth"])
        checkpoint = tmp_path / "model.pt"
        torch.save(model.state_dict(), checkpoint)
        restored = Adapter(config()).eval()
        restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
        torch.manual_seed(19)
        assert torch.equal(output["depth"], restored(sample["inputs"])["depth"])
        cfg["evaluation"]["reference_indices"] = [0, 1, 2, 3]
        four = model(sample["inputs"])
        assert four["depth"].shape == (1, 4, 32, 64)
        assert torch.isfinite(four["depth"]).all()


def test_empty_supervision_and_missing_view_boundary(sample):
    model = Adapter(config())
    output = model(sample["inputs"])
    targets = copy.deepcopy(sample["targets"])
    targets["depth_valid"].zero_()
    loss = model.loss(output, sample["inputs"], targets)["total"]
    assert loss.item() == 0 and torch.isfinite(loss)
    loss.backward()
    sample["inputs"]["view_mask"][:, 1] = False
    with pytest.raises(ValueError, match="共同完整子集"):
        model(sample["inputs"])
