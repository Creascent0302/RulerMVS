"""射线标尺编译接入后，保留旧模型接口、尺度路由和训练梯度。"""

import copy

import pytest
import torch

from jar.geometry import pixel_grid
from jar.pretraining.losses import depth_objective
from jar.pretraining.model import CalibratedDecoder
from jar.pretraining.projective_ruler import compile_projective_ruler
from tests.test_public_pretraining import DOMAIN, model, sample


@pytest.fixture
def configured_model(model):
    settings = copy.deepcopy(model.decoder.settings)
    model.decoder.settings.update(projective_ruler=True, compact_sources=True)
    yield model
    model.decoder.settings.clear()
    model.decoder.settings.update(settings)
    model.zero_grad(set_to_none=True)
    model.eval()


@pytest.mark.parametrize("views,stage", [(2, 1), (3, 2), (4, 3)])
def test_real_geometry_eval_and_training(configured_model, views, stage):
    network = configured_model
    batch = sample(views)
    network.eval()
    with torch.no_grad():
        prediction = network(batch["inputs"], DOMAIN)
    assert prediction["depth"].shape == (1, 1, 32, 64)
    assert torch.isfinite(prediction["depth"]).all()
    network.set_stage(stage)
    network.train()
    network.zero_grad(set_to_none=True)
    prediction = network(batch["inputs"], DOMAIN)
    loss = depth_objective(prediction, batch["inputs"], batch["targets"], DOMAIN, {}, .1)
    loss["total"].backward()
    assert torch.isfinite(loss["total"])
    prefixes = ("matching", "decoder") + (("foundation",) if stage > 1 else ())
    for prefix in prefixes:
        gradients = [p.grad for name, p in network.named_parameters()
                     if name.startswith(prefix) and p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


@pytest.mark.parametrize("stride,routing", [(8, True), (4, True), (8, False)])
def test_observe_preserves_sampling_and_scale_routing(configured_model, stride, routing):
    network = configured_model.eval()
    network.decoder.settings["scale_routing"] = routing
    batch = sample(3)
    batch["inputs"]["view_mask"][:, 1] = False
    batch["inputs"]["K"][..., 0, 1] = 0.2
    with torch.no_grad():
        work = network.encode(batch["inputs"], DOMAIN)
        prepared = network.decoder._prepare(work)
        height, width = 32 // stride, 64 // stride
        candidates = torch.linspace(2., 28., 6)[None, :, None, None].expand(1, -1, height, width)
        widths = torch.ones(6)
        network.decoder.settings["projective_ruler"] = False
        original = network.decoder._observe(prepared, candidates, stride, widths)
        network.decoder.settings["projective_ruler"] = True
        compiled = network.decoder._observe(prepared, candidates, stride, widths)
    assert "points" not in compiled
    for key in ("reference", "ray_derivative"):
        torch.testing.assert_close(original[key], compiled[key], atol=2e-5, rtol=2e-5)
    common = original["source_valid"] & compiled["source_valid"]
    torch.testing.assert_close(original["source"] * common[:, :, None],
                               compiled["source"] * common[:, :, None], atol=2e-5, rtol=2e-5)
    assert torch.equal(original["ref_valid"], compiled["ref_valid"])
    # 严格图像边界会放大FP32舍入，允许差异但限定到真实边界附近；
    # 不以放松生产掩码或删除这些像素的方式换取所谓逐位一致。
    changed = original["source_valid"] != compiled["source_valid"]
    grid = pixel_grid(height, width, stride=stride).reshape(-1, 2)
    ruler = compile_projective_ruler(grid, prepared["K"][0, 0], prepared["T"][0, 0],
                                     prepared["K"][0, 1:], prepared["T"][0, 1:])
    uv = ruler.project(candidates[0].flatten(1), (32, 64))["uv"].reshape(1, 2, 6, height, width, 2)
    border_distance = torch.stack((uv[..., 0].abs(), uv[..., 1].abs(),
                                   (uv[..., 0] - 63).abs(), (uv[..., 1] - 31).abs()), -1).amin(-1)
    assert (border_distance[changed] < 1e-4).all()


@pytest.mark.parametrize("missing_reference,pure_rotation", [(False, False), (True, False), (False, True)])
def test_missing_views_and_rotation_preserve_unknown_contract(configured_model, missing_reference, pure_rotation):
    network = configured_model.eval()
    batch = sample(3, pure_rotation)
    if not pure_rotation:
        indices = [0] if missing_reference else [1, 2]
        for key in ("view_mask", "image_mask", "matching_image_mask"):
            batch["inputs"][key][:, indices] = False
        for key in ("rgb", "matching_rgb"):
            batch["inputs"][key][:, indices] = float("nan")
    with torch.no_grad():
        prediction = network(batch["inputs"], DOMAIN)
        network.decoder.settings["projective_ruler"] = False
        original = network(batch["inputs"], DOMAIN)
    assert torch.isfinite(prediction["depth"]).all()
    assert torch.equal(prediction["valid"], original["valid"])
    assert torch.equal(prediction["unknown"], original["unknown"])
    assert not prediction["observed"].any()
    assert prediction["support"].max() == 0
    if missing_reference:
        assert not prediction["valid"].any()


def test_new_flags_do_not_change_legacy_defaults_or_checkpoint_keys(configured_model):
    network = configured_model
    assert not network.config["model"].get("projective_ruler", False)
    parameters = {key: id(value) for key, value in network.named_parameters()}
    state = network.state_dict()
    network.decoder.settings["projective_ruler"] = True
    network.eval()
    with torch.no_grad():
        prediction = network(sample(2)["inputs"], DOMAIN)
    assert torch.isfinite(prediction["depth"]).all()
    network.load_state_dict(state, strict=True)
    assert parameters == {key: id(value) for key, value in network.named_parameters()}
