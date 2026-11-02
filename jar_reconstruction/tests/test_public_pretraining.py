import copy

import pytest
import torch
from torch.nn import functional as F

from jar.data import collate_frames
from jar.pretraining.data import FullPassSampler
from jar.pretraining.losses import depth_objective
from jar.pretraining.model import PretrainingModel
from jar.synthetic import SyntheticRigDataset


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    config = {"initialization": {"foundation": "", "matching": ""}, "model": {
        "channels": 16, "raw_channels": 32, "depth_bins": 12, "peak_count": 2,
        "near": 2., "far": 80., "use_scene": False, "use_geometry": True,
        "use_fine_offset": True, "local_kind": "scale_space", "proposal_transport": "bilinear"}}
    return PretrainingModel(config, initialize=False)


def sample(views=3, pure_rotation=False):
    batch = collate_frames([SyntheticRigDataset(length=1, image_size=(32, 64), num_cameras=views,
                                                pure_rotation=pure_rotation)[0]])
    inputs = batch["inputs"]
    inputs["matching_rgb"] = F.interpolate(inputs["rgb"].flatten(0, 1), scale_factor=2,
                                           mode="bilinear", align_corners=False).unflatten(0, (1, views))
    inputs["matching_image_mask"] = inputs["image_mask"].repeat_interleave(2, -2).repeat_interleave(2, -1)
    return batch


DOMAIN = {"depth_scale": 1., "near": 1., "far": 30., "correspondence": True}


@pytest.mark.parametrize("views,stage", [(2, 1), (3, 2), (4, 3)])
def test_real_projection_forward_backward_all_stages(model, views, stage):
    batch = sample(views)
    model.zero_grad(set_to_none=True)
    model.set_stage(stage)
    model.train()
    prediction = model(batch["inputs"], DOMAIN)
    loss = depth_objective(prediction, batch["inputs"], batch["targets"], DOMAIN, {}, .1)
    loss["total"].backward()
    assert torch.isfinite(loss["total"])
    for prefix in ("matching", "decoder") + (("foundation",) if stage >= 2 else ()):
        gradients = [p.grad for name, p in model.named_parameters() if name.startswith(prefix) and p.requires_grad]
        present = [g for g in gradients if g is not None]
        assert present and all(torch.isfinite(g).all() for g in present)
        assert sum(float(g.abs().sum()) for g in present) > 0
    if stage == 1:
        assert all(p.grad is None for p in model.foundation.parameters())


def test_source_permutation_unit_invariance_and_no_label_input(model):
    batch = sample(3)
    model.eval()
    with torch.no_grad():
        reference = model(batch["inputs"], DOMAIN)["depth"]
        batch["targets"]["depth"] *= 27
        assert torch.equal(reference, model(batch["inputs"], DOMAIN)["depth"])
        permuted = {k: v[:, [0, 2, 1]] for k, v in batch["inputs"].items()}
        assert torch.allclose(reference, model(permuted, DOMAIN)["depth"], atol=1e-4)
        scaled = {k: v.clone() for k, v in batch["inputs"].items()}
        scaled["T_cam_from_rig"][..., :3, 3] *= 100
        domain = {**DOMAIN, "depth_scale": 100., "near": 100., "far": 3000.}
        assert torch.allclose(reference, model(scaled, domain)["depth"], atol=1e-4)


def test_missing_views_and_rotation_are_finite(model):
    model.eval()
    for rotation in (False, True):
        batch = sample(3, rotation)
        if not rotation:
            for key in ("view_mask", "image_mask", "matching_image_mask"):
                batch["inputs"][key][:, 1:] = False
            for key in ("rgb", "matching_rgb"):
                batch["inputs"][key][:, 1:] = float("nan")
        with torch.no_grad():
            result = model(batch["inputs"], DOMAIN)
        assert torch.isfinite(result["depth"]).all()
        assert not result["observed"].any()
        assert result["support"].max() == 0


def test_sampler_full_coverage_domain_alignment_and_resume():
    options = dict(lengths=[13, 29, 7], epoch=3, world=2, batch_size=2, accumulation=2, seed=7)
    ranks = [FullPassSampler(rank=rank, **options) for rank in (0, 1)]
    batches = [list(sampler) for sampler in ranks]
    covered = {i: set() for i in range(3)}
    for a, b in zip(*batches):
        assert len({key[0] for key in a + b}) == 1
        for domain, index, epoch in a + b:
            covered[domain].add(index)
            assert epoch == 3
    for index, length in enumerate(options["lengths"]):
        assert covered[index] == set(range(length))
    assert list(FullPassSampler(rank=0, start_update=2, **options)) == batches[0][4:]


def test_inference_optimizations_preserve_model_contract(model):
    from jar.pretraining.inference import prepare_inference

    original = model.eval()
    updated = copy.deepcopy(original)
    identities = {name: id(parameter) for name, parameter in updated.named_parameters()}
    prepare_inference(updated, spatial_layout=True, factorized_fusion=True,
                      hoist_reference=True, compact_pair=True)
    assert identities == {name: id(parameter) for name, parameter in updated.named_parameters()}
    for name, value in original.state_dict().items():
        assert torch.equal(value, updated.state_dict()[name])
    for views, rotation in ((2, False), (3, True), (4, False)):
        batch = sample(views, rotation)
        if views == 4:
            batch["inputs"]["view_mask"][:, 1:] = False
        with torch.no_grad():
            first = original(batch["inputs"], DOMAIN)
            second = updated(batch["inputs"], DOMAIN)
        torch.testing.assert_close(first["depth"], second["depth"], atol=1e-4, rtol=1e-4)
        assert torch.equal(first["valid"], second["valid"])
        assert torch.equal(first["support"], second["support"])
        assert torch.isfinite(second["depth"]).all()
    # 部署替换不改变 checkpoint 键；可直接严格加载回原训练模型。
    original.load_state_dict(updated.state_dict(), strict=True)


def test_spatial_layout_training_configuration_is_checkpoint_compatible(model):
    from jar.pretraining.fast_spatial import ContiguousDepthwise3d

    config = copy.deepcopy(model.config)
    config["model"].update(spatial_layout=True, compact_sources=True)
    updated = PretrainingModel(config, initialize=False)
    updated.load_state_dict(model.state_dict(), strict=True)
    assert isinstance(updated.decoder.local.depthwise, ContiguousDepthwise3d)
    assert all(isinstance(block.depthwise, ContiguousDepthwise3d) for block in updated.decoder.local.scales)
    updated.set_stage(1)
    updated.train()
    batch = sample(3)
    prediction = updated(batch["inputs"], DOMAIN)
    loss = depth_objective(prediction, batch["inputs"], batch["targets"], DOMAIN, {}, .1)
    loss["total"].backward()
    assert torch.isfinite(loss["total"])
    assert torch.isfinite(updated.decoder.local.depthwise.weight.grad).all()
