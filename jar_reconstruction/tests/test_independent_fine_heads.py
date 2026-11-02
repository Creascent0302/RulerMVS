"""细头解耦必须保持起点和粗候选供给，不能吞掉checkpoint错误。"""

import pytest
import torch
from torch import nn

from jar.data import collate_frames
from jar.pretraining.model import CalibratedDecoder, load_pretraining_state
from jar.synthetic import SyntheticRigDataset


def fixture():
    torch.set_num_threads(2)
    torch.manual_seed(37)
    settings = {"channels": 16, "raw_channels": 32, "depth_bins": 12, "peak_count": 2,
                "near": 1., "far": 30., "use_scene": False, "local_kind": "scale_space",
                "proposal_transport": "bilinear", "compact_sources": True, "peak_exclusion_bins": 1}
    original = CalibratedDecoder(settings).eval()
    independent = CalibratedDecoder(dict(settings, independent_fine_heads=True)).eval()
    load_pretraining_state(independent, original.state_dict())
    original.set_range(1., 30.)
    independent.set_range(1., 30.)
    inputs = collate_frames([SyntheticRigDataset(length=1, image_size=(32, 64), num_cameras=3)[0]])["inputs"]
    work = {"pyramid": {stride: torch.randn(1, 3, 16, 64 // stride, 128 // stride)
                        for stride in (4, 8, 16)},
            "raw_maps": [torch.randn(1, 768, 2, 4) for _ in range(4)],
            "K": inputs["K"], "T_cam_from_rig": inputs["T_cam_from_rig"],
            "view_mask": inputs["view_mask"], "image_mask": inputs["image_mask"],
            "reference_index": 1, "image_size": (32, 64), "matching_scale": torch.tensor([[2., 2.]])}
    return original, independent, work


def test_old_checkpoint_copies_heads_and_preserves_initial_predictions():
    original, independent, work = fixture()
    assert not any(key.startswith("fine_score") for key in original.state_dict())
    for head in ("score", "offset"):
        for parameter in ("weight", "bias"):
            assert torch.equal(getattr(getattr(independent, "fine_" + head), parameter),
                               getattr(getattr(original, head), parameter))
    with torch.no_grad():
        first, second = original(work), independent(work)
    assert torch.equal(first["depth"], second["depth"])
    for stage in ("coarse", "fine"):
        for key in ("scores", "candidate_depth", "sampling_depth", "depth"):
            assert torch.equal(first[stage][key], second[stage][key])


@pytest.mark.parametrize("problem", ["partial_heads", "missing_other", "unexpected", "bad_shape"])
def test_checkpoint_loader_remains_strict(problem):
    original, independent, _ = fixture()
    state = dict(original.state_dict())
    if problem == "partial_heads":
        state["fine_score.weight"] = original.score.weight.detach().clone()
    elif problem == "missing_other":
        state.pop("raw_query.weight")
    elif problem == "unexpected":
        state["invented.weight"] = torch.ones(1)
    else:
        state["score.weight"] = torch.ones(1)
    with pytest.raises(RuntimeError):
        load_pretraining_state(independent, state)


def test_full_model_prefix_and_new_checkpoint_roundtrip():
    original, independent, _ = fixture()
    old_model, new_model = nn.Module(), nn.Module()
    old_model.decoder = original
    new_model.decoder = independent
    load_pretraining_state(new_model, old_model.state_dict())
    with torch.no_grad():
        independent.fine_score.bias.add_(.5)
    saved = {key: value.clone() for key, value in new_model.state_dict().items()}
    with torch.no_grad():
        independent.fine_score.bias.zero_()
    load_pretraining_state(new_model, saved)
    assert torch.equal(new_model.decoder.fine_score.bias, saved["decoder.fine_score.bias"])


def test_only_fine_training_keeps_coarse_and_sampling_bitwise_unchanged():
    _, model, work = fixture()
    model.requires_grad_(False)
    for name in ("fine_inherit", "fine_score", "fine_offset", "upsample_mask"):
        getattr(model, name).requires_grad_(True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.train()
    before = model(work)
    loss = before["depth"].mean() + before["fine"]["scores"].square().mean()
    loss.backward()
    for head in ("fine_score", "fine_offset", "fine_inherit", "upsample_mask"):
        gradients = [p.grad for p in getattr(model, head).parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
    assert model.score.weight.grad is None and model.offset.weight.grad is None
    optimizer.step()
    with torch.no_grad():
        after = model(work)
    for key in ("scores", "candidate_depth", "sampling_depth", "selected_index"):
        assert torch.equal(before["coarse"][key], after["coarse"][key])
    assert torch.equal(before["fine"]["sampling_depth"], after["fine"]["sampling_depth"])
    assert not torch.equal(before["fine"]["scores"], after["fine"]["scores"])
    assert torch.isfinite(after["depth"]).all()
