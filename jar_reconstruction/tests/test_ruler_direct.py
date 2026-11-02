"""独立公制解码器的几何/训练接口检查；不评价未训练网络的精度。"""

import copy

import pytest
import torch
from torch.nn import functional as F

from jar.geometry import project, unproject
from jar.models.ruler_direct import MetricSceneFusion, RulerDirectDecoder
from jar.synthetic import make_synthetic_batch


def tiny_settings(**changes):
    settings = dict(depth_bins=12, channels=8, raw_channels=8, scene_channels=16,
                    scene_heads=2, scene_grid=(3, 2, 3), peak_count=4)
    return dict(settings, **changes)


def fixture_inputs(batch=1, views=4, pure_rotation=False, texture=True):
    frame = make_synthetic_batch(B=batch, N=views, H=32, W=48,
                                 pure_rotation=pure_rotation, texture=texture)
    inputs = frame["inputs"]
    rgb = inputs["rgb"]
    # 所有图像来自同一个渲染场景。仅用于接口测试的固定图像描述子，
    # 不是随机独立图像，也不是声称复现了预训练编码器。
    matching = F.avg_pool2d(rgb.flatten(0, 1), 4)
    inputs["matching_features"] = matching.repeat(1, 6, 1, 1)[:, :16].unflatten(0, (batch, views))
    raw = F.avg_pool2d(rgb[:, 0], 16).repeat(1, 256, 1, 1)
    inputs["raw_maps"] = [raw * (index + 1) / 4 for index in range(4)]
    inputs.update(image_size=(32, 48), reference_index=0, matching_scale=(1., 1.))
    return inputs, frame["targets"]


def assert_finite(value):
    if torch.is_tensor(value):
        assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for child in value.values():
            assert_finite(child)


@pytest.mark.parametrize("views", [2, 3, 4])
def test_forward_backward_all_training_branches(views):
    torch.manual_seed(2)
    model = RulerDirectDecoder(tiny_settings())
    inputs, targets = fixture_inputs(batch=2, views=views)
    inputs["view_mask"][1, -1] = False
    inputs["image_mask"][0, 1, :8, :8] = False
    output = model(inputs)
    assert_finite(output)
    assert output["depth"].shape == (2, 1, 32, 48)
    assert output["coarse"]["scores"].shape == (2, 12, 4, 6)
    assert output["fine"]["scores"].shape == (2, 12, 8, 12)
    assert (output["depth"][output["valid"]] >= 2 - 1e-5).all()
    assert (output["depth"][output["valid"]] <= 80 + 1e-5).all()
    assert output["support"].max() <= views - 1
    assert output["unknown"].all() and not output["observed"].any()
    loss = output["depth"].mean()
    for name in ("coarse", "fine"):
        stage = output[name]
        loss = loss + stage["scores"][stage["valid"]].square().mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for prefix in ("source.", "raw_adapters.", "scene.", "score.", "offset.", "upsample_mask."):
        assert sum(parameter.grad.abs().sum() for name, parameter in model.named_parameters()
                   if name.startswith(prefix)) > 0, prefix
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.step()
    assert_finite(model(inputs))


def test_missing_sources_and_reference_are_distinct():
    model = RulerDirectDecoder(tiny_settings()).eval()
    inputs, _ = fixture_inputs()
    inputs["view_mask"][:, 1:] = False
    inputs["matching_features"][:, 1:] = float("nan")
    inputs["K"][:, 1:] = float("nan")
    inputs["T_cam_from_rig"][:, 1:] = float("nan")
    with torch.no_grad():
        output = model(inputs)
    assert_finite(output)
    assert output["valid"].all()
    assert (output["depth"] > 0).all()
    assert output["support"].sum() == 0
    assert output["unknown"].all()
    assert output["coarse"]["valid"].all()
    inputs["view_mask"][:] = False
    with torch.no_grad():
        missing = model(inputs)
    assert_finite(missing)
    assert not missing["valid"].any()
    assert missing["depth"].sum() == 0


def test_missing_slots_do_not_change_predictions_when_corrupted():
    model = RulerDirectDecoder(tiny_settings()).eval()
    inputs, _ = fixture_inputs()
    inputs["view_mask"][:, -1] = False
    altered = copy.deepcopy(inputs)
    altered["matching_features"][:, -1] = float("nan")
    altered["K"][:, -1] = float("nan")
    altered["T_cam_from_rig"][:, -1] = float("nan")
    with torch.no_grad():
        expected, actual = model(inputs), model(altered)
    torch.testing.assert_close(actual["depth"], expected["depth"], atol=0, rtol=0)


def test_pure_rotation_has_no_metric_source_support():
    model = RulerDirectDecoder(tiny_settings()).eval()
    inputs, _ = fixture_inputs(pure_rotation=True)
    with torch.no_grad():
        output = model(inputs)
    assert_finite(output)
    assert output["support"].sum() == 0
    assert output["valid"].all() and output["unknown"].all()


def test_source_order_is_permutation_equivariant():
    model = RulerDirectDecoder(tiny_settings()).eval()
    inputs, _ = fixture_inputs()
    order = [2, 3, 0, 1]
    permuted = dict(inputs)
    for name in ("matching_features", "K", "T_cam_from_rig", "view_mask", "image_mask"):
        permuted[name] = inputs[name][:, order]
    permuted["reference_index"] = order.index(0)
    with torch.no_grad():
        expected, actual = model(inputs), model(permuted)
    torch.testing.assert_close(actual["depth"], expected["depth"], atol=5e-5, rtol=1e-5)
    torch.testing.assert_close(actual["support"], expected["support"])


def test_checkpoint_and_label_independence(tmp_path):
    model = RulerDirectDecoder(tiny_settings()).eval()
    inputs, targets = fixture_inputs()
    with torch.no_grad():
        expected = model(inputs)
    path = tmp_path / "decoder.pt"
    torch.save({"settings": model.settings, "state_dict": model.state_dict()}, path)
    checkpoint = torch.load(path, weights_only=True)
    restored = RulerDirectDecoder(checkpoint["settings"]).eval()
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    inputs["targets"] = {"depth": torch.full((1, 4, 32, 48), 12345.)}
    with torch.no_grad():
        actual = restored(inputs)
    torch.testing.assert_close(actual["depth"], expected["depth"], atol=0, rtol=0)


@pytest.mark.parametrize("use_scene,use_geometry", [(False, True), (True, False), (False, False)])
def test_ablation_switches_run(use_scene, use_geometry):
    model = RulerDirectDecoder(tiny_settings(use_scene=use_scene, use_geometry=use_geometry)).eval()
    inputs, _ = fixture_inputs()
    with torch.no_grad():
        output = model(inputs)
    assert_finite(output)
    assert (output["scene"] is not None) == use_scene


def test_scene_geometry_changes_real_attention():
    torch.manual_seed(3)
    model = RulerDirectDecoder(tiny_settings()).eval()
    inputs, _ = fixture_inputs()
    prepared = model._prepare(inputs)
    candidate = model.depth_lattice[None, :, None, None].expand(1, -1, 4, 6)
    observation = model._observe(prepared, candidate, 8, torch.ones(12))
    feature, raw, valid, _, _, _ = model._tokens(prepared, observation, candidate, 8)
    with torch.no_grad():
        on, diagnostics = model.scene(feature, valid, observation["points"], raw, True)
        off, _ = model.scene(feature, valid, observation["points"], raw, False)
    assert diagnostics["node_count"] == 18
    assert diagnostics["geometry_distance"].max() > 0
    assert (on - off).abs().max() > 1e-5


def test_matching_ruler_derivative_matches_exact_projection():
    model = RulerDirectDecoder(tiny_settings())
    inputs, _ = fixture_inputs()
    prepared = model._prepare(inputs)
    candidates = torch.full((1, 1, 4, 6), 13.)
    observed = model._observe(prepared, candidates, 8, torch.ones(1))
    grid = torch.tensor([[24., 16.]])
    changes = []
    for depth in (12.99, 13.01):
        points, _ = unproject(grid, torch.tensor([depth]), inputs["K"][0, 0], inputs["T_cam_from_rig"][0, 0])
        uv, _, _, _ = project(points, inputs["K"][0], inputs["T_cam_from_rig"][0], (32, 48))
        changes.append((uv[:, 0] + 0.5) / 4 - 0.5)
    numerical = (changes[1] - changes[0]) / 0.02
    analytic = observed["ray_derivative"][0, :, :, 0, 2, 3]
    torch.testing.assert_close(analytic, numerical, atol=1e-4, rtol=2e-3)


def test_absolute_information_switch_matches_explicit_evidence_mask():
    from tools.diagnose_ruler_direct import ablation

    model = RulerDirectDecoder(tiny_settings()).eval()
    inputs, _ = fixture_inputs()
    with torch.no_grad(), ablation(model, "metric_channel_off"):
        expected = model(inputs)
    model.absolute_information = False
    with torch.no_grad():
        actual = model(inputs)
    torch.testing.assert_close(actual["depth"], expected["depth"], atol=0, rtol=0)


def test_disabled_scene_has_no_inactive_trainable_parameters():
    model = RulerDirectDecoder(tiny_settings(use_scene=False, use_geometry=False))
    assert all(not parameter.requires_grad for parameter in model.scene.parameters())
    inputs, _ = fixture_inputs()
    output = model(inputs)
    (output["depth"].mean() + output["coarse"]["scores"].square().mean()
     + output["fine"]["scores"].square().mean()).backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_fine_offset_switch_matches_exact_zero_hook_and_retains_coarse_gradient():
    model = RulerDirectDecoder(tiny_settings()).eval()
    with torch.no_grad():
        model.offset.bias.fill_(0.3)
    inputs, _ = fixture_inputs()
    calls = [0]

    def zero_fine(module, args, output):
        calls[0] += 1
        return torch.zeros_like(output) if calls[0] % 2 == 0 else output

    handle = model.offset.register_forward_hook(zero_fine)
    with torch.no_grad():
        expected = model(inputs)
    handle.remove()
    model.use_fine_offset = False
    actual = model(inputs)
    torch.testing.assert_close(actual["depth"], expected["depth"], atol=0, rtol=0)
    torch.testing.assert_close(actual["fine"]["candidate_depth"], actual["fine"]["sampling_depth"], atol=0, rtol=0)
    (actual["depth"].mean() + actual["coarse"]["depth"].mean()).backward()
    assert torch.isfinite(model.offset.weight.grad).all()
    assert model.offset.weight.grad.abs().sum() > 0


def test_convex_upsample_preserves_constant_depth_at_border():
    model = RulerDirectDecoder(tiny_settings())
    features = torch.ones(1, 8, 12, 8, 12)
    fine = {"selected_index": torch.zeros(1, 1, 8, 12, dtype=torch.long),
            "ref_valid": torch.ones(1, 1, 8, 12, dtype=torch.bool),
            "depth": torch.full((1, 1, 8, 12), 17.),
            "confidence": torch.ones(1, 1, 8, 12),
            "support": torch.ones(1, 12, 8, 12, dtype=torch.long)}
    output = model._upsample(fine, features, torch.ones(1, 1, 32, 48, dtype=torch.bool))
    torch.testing.assert_close(output["depth"], torch.full_like(output["depth"], 17.), atol=1e-5, rtol=1e-6)


def test_fine_pixel_mask_does_not_silently_drop_valid_rgb():
    model = RulerDirectDecoder(tiny_settings()).eval()
    inputs, _ = fixture_inputs()
    inputs["image_mask"][:] = False
    inputs["image_mask"][:, 0, 1, 1] = True
    with torch.no_grad():
        output = model(inputs)
    assert_finite(output)
    assert output["valid"].sum() == 1
    assert output["depth"][0, 0, 1, 1] > 0
    assert output["support"].sum() == 0


def test_depth_bank_reuses_parameters_and_owner_default_is_exact():
    torch.manual_seed(7)
    default = RulerDirectDecoder(tiny_settings()).eval()
    owner = RulerDirectDecoder(tiny_settings(scene_return="owner")).eval()
    bank = RulerDirectDecoder(tiny_settings(scene_return="depth_bank")).eval()
    owner.load_state_dict(default.state_dict(), strict=True)
    bank.load_state_dict(default.state_dict(), strict=True)
    assert sum(p.numel() for p in default.parameters()) == sum(p.numel() for p in bank.parameters())
    inputs, _ = fixture_inputs(batch=2)
    with torch.no_grad():
        expected, actual = default(inputs), owner(inputs)
    for stage in ("coarse", "fine"):
        torch.testing.assert_close(actual[stage]["scores"], expected[stage]["scores"], atol=0, rtol=0)
    torch.testing.assert_close(actual["depth"], expected["depth"], atol=0, rtol=0)


def test_depth_bank_nondivisible_partition_matches_explicit_pixel_attention():
    torch.manual_seed(8)
    scene = MetricSceneFusion(4, 2, tiny_settings(scene_return="depth_bank"))
    features = torch.randn(2, 4, 5, 3, 7)
    grid = (3, 2, 3)
    nodes = torch.randn(2, 18, 16)
    known = torch.ones(2, 18, dtype=torch.bool)
    known[0, 7] = False
    known[1] = False
    actual = scene._depth_bank_return(features, nodes, known, grid)
    query = scene.return_query(features)
    keys = scene.return_key(nodes).reshape(2, 3, 6, 4)
    values = scene.return_value(nodes).reshape(2, 3, 6, 4)
    known = known.reshape(2, 3, 6)
    expected = torch.zeros_like(actual)
    # 小尺寸逐像素参考只用于验证分区/还原；生产路径没有逐像素循环。
    for y in range(3):
        for x in range(7):
            region = (y // 2) * 3 + x // 3
            scores = torch.einsum("bcd,bkc->bdk", query[..., y, x], keys[:, :, region]) / 2
            scores = scores.masked_fill(~known[:, None, :, region], float("-inf"))
            weights = F.pad(scores, (0, 1), value=0).softmax(-1)[..., :-1]
            expected[..., y, x] = torch.einsum("bdk,bkc->bcd", weights, values[:, :, region])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert actual[1].count_nonzero() == 0


def test_depth_bank_reads_nonowner_node_and_rejects_missing_node():
    torch.manual_seed(9)
    scene = MetricSceneFusion(4, 2, tiny_settings(scene_return="depth_bank"))
    features = torch.randn(1, 4, 4, 2, 2)
    nodes = torch.randn(1, 4, 16)
    changed = nodes.clone()
    changed[:, 3] += torch.arange(16) / 4
    known = torch.ones(1, 4, dtype=torch.bool)
    before = scene._depth_bank_return(features, nodes, known, (4, 1, 1))
    after = scene._depth_bank_return(features, changed, known, (4, 1, 1))
    assert (before[:, :, 0] - after[:, :, 0]).abs().max() > 1e-5
    known[:, 3] = False
    before = scene._depth_bank_return(features, nodes, known, (4, 1, 1))
    after = scene._depth_bank_return(features, changed, known, (4, 1, 1))
    torch.testing.assert_close(before, after, atol=0, rtol=0)


@pytest.mark.parametrize("views", [2, 3, 4])
@pytest.mark.parametrize("null_prior", ["single", "balanced"])
def test_depth_bank_full_forward_backward_and_all_invalid(views, null_prior):
    torch.manual_seed(10)
    model = RulerDirectDecoder(tiny_settings(scene_return="depth_bank", scene_null_prior=null_prior))
    inputs, _ = fixture_inputs(batch=2, views=views)
    inputs["view_mask"][1] = False
    output = model(inputs)
    assert_finite(output)
    assert output["depth"][1].count_nonzero() == 0
    assert not output["scene"]["valid_nodes"][1].any()
    loss = output["depth"].mean()
    for stage in ("coarse", "fine"):
        loss = loss + output[stage]["scores"][0].square().mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model.scene.return_query.weight.grad.abs().sum() > 0
    assert model.scene.return_key.weight.grad.abs().sum() > 0
    assert model.scene.return_value.weight.grad.abs().sum() > 0


def test_balanced_null_keeps_half_mass_for_any_positive_node_count():
    scene = MetricSceneFusion(4, 2, tiny_settings(scene_return="depth_bank", scene_null_prior="balanced"))
    with torch.no_grad():
        scene.return_query.weight.zero_()
        scene.return_query.bias.zero_()
        scene.return_value.weight.zero_()
        scene.return_value.bias.fill_(1)
    features = torch.randn(4, 4, 3, 2, 2)
    nodes = torch.randn(4, 4, 16)
    known = torch.arange(4)[None] < torch.tensor([0, 1, 2, 4])[:, None]
    message = scene._depth_bank_return(features, nodes, known, (4, 1, 1))
    assert message[0].count_nonzero() == 0
    torch.testing.assert_close(message[1:], torch.full_like(message[1:], 0.5), atol=1e-7, rtol=0)


def test_balanced_null_is_invariant_to_duplicate_nodes_and_matches_single_owner():
    torch.manual_seed(11)
    scene = MetricSceneFusion(4, 2, tiny_settings(scene_return="depth_bank", scene_null_prior="balanced"))
    features = torch.randn(2, 4, 5, 3, 7)
    node = torch.randn(2, 1, 16)
    one = scene._depth_bank_return(features, node, torch.ones(2, 1, dtype=torch.bool), (1, 1, 1))
    four = scene._depth_bank_return(features, node.expand(-1, 4, -1), torch.ones(2, 4, dtype=torch.bool), (4, 1, 1))
    key = scene.return_key(node).transpose(1, 2)[:, :, :, None, None]
    value = scene.return_value(node).transpose(1, 2)[:, :, :, None, None]
    owner = ((scene.return_query(features) * key).sum(1, keepdim=True) / 2).sigmoid() * value
    torch.testing.assert_close(four, one, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(one, owner, atol=1e-6, rtol=1e-6)


def test_balanced_null_does_not_affect_owner_return():
    torch.manual_seed(12)
    original = RulerDirectDecoder(tiny_settings()).eval()
    balanced = RulerDirectDecoder(tiny_settings(scene_null_prior="balanced")).eval()
    balanced.load_state_dict(original.state_dict(), strict=True)
    inputs, _ = fixture_inputs()
    with torch.no_grad():
        expected, actual = original(inputs), balanced(inputs)
    torch.testing.assert_close(actual["depth"], expected["depth"], atol=0, rtol=0)


def test_return_geometry_is_rigid_invariant_and_has_normalized_prior():
    torch.manual_seed(13)
    scene = MetricSceneFusion(4, 2, tiny_settings(scene_return="depth_bank", scene_null_prior="balanced",
                                                scene_return_geometry=True))
    points = torch.randn(2, 3, 5, 3, 7)
    center = torch.randn(2, 18, 3)
    matrix = torch.randn(2, 18, 3, 3)
    covariance = matrix @ matrix.transpose(-1, -2) + torch.eye(3)
    available = torch.ones(2, 6, 3, dtype=torch.bool)
    available[0, 0, 1] = False
    grid = (3, 2, 3)
    expected = scene._region_log_prior(points, center, covariance, available, grid)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotation[:, 0] *= torch.linalg.det(rotation)
    translation = torch.tensor([4., -2., 1.])
    transformed_points = torch.einsum("ij,bjdhw->bidhw", rotation, points) + translation[None, :, None, None, None]
    transformed_center = center @ rotation.T + translation
    transformed_covariance = rotation @ covariance @ rotation.T
    actual = scene._region_log_prior(transformed_points, transformed_center, transformed_covariance, available, grid)
    # padding候选不参与输出；旋转/平移后只检查真实候选覆盖的区域。
    from jar.models.ruler_direct import _partition
    present = _partition(torch.ones(2, 1, 5, 3, 7), (1, 2, 3))[0][..., 0].bool()
    torch.testing.assert_close(actual[present], expected[present], atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(expected.exp().sum(-1), torch.ones_like(expected[..., 0]), atol=1e-6, rtol=1e-6)


def test_uniform_region_prior_matches_balanced_control_and_all_invalid_is_zero():
    torch.manual_seed(14)
    geometry = MetricSceneFusion(4, 2, tiny_settings(scene_return="depth_bank", scene_null_prior="balanced",
                                                   scene_return_geometry=True))
    control = MetricSceneFusion(4, 2, tiny_settings(scene_return="depth_bank", scene_null_prior="balanced"))
    control.load_state_dict(geometry.state_dict(), strict=True)
    features = torch.randn(2, 4, 5, 3, 7)
    points = torch.randn(2, 3, 5, 3, 7)
    nodes = torch.randn(2, 18, 16)
    known = torch.ones(2, 18, dtype=torch.bool)
    known[1] = False
    center = torch.zeros(2, 18, 3)
    covariance = torch.eye(3).expand(2, 18, 3, 3)
    grid = (3, 2, 3)
    actual = geometry._depth_bank_return(features, nodes, known, grid, points, center, covariance)
    expected = control._depth_bank_return(features, nodes, known, grid)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert actual[1].count_nonzero() == 0
    with torch.no_grad():
        geometry.return_query.weight.zero_()
        geometry.return_query.bias.zero_()
        geometry.return_value.weight.zero_()
        geometry.return_value.bias.fill_(1)
    center[:, 6:] += 3
    actual = geometry._depth_bank_return(features, nodes, known, grid, points, center, covariance)
    torch.testing.assert_close(actual[0], torch.full_like(actual[0], 0.5), atol=1e-6, rtol=1e-6)


def test_return_geometry_full_forward_backward():
    model = RulerDirectDecoder(tiny_settings(scene_return="depth_bank", scene_null_prior="balanced",
                                             scene_return_geometry=True))
    inputs, _ = fixture_inputs(batch=2)
    inputs["view_mask"][1] = False
    output = model(inputs)
    assert_finite(output)
    loss = output["depth"].mean() + output["coarse"]["scores"][0].square().mean() + output["fine"]["scores"][0].square().mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
