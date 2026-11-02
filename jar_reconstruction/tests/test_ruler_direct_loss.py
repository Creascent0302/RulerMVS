"""子射线集合监督：只修改训练目标，不修改推理或fine的逐射线损失。"""

import math

import pytest
import torch

from jar.models.ruler_direct import RulerDirectDecoder
from tests.test_ruler_direct import fixture_inputs, tiny_settings
from tools.train_original_initializer import initializer_loss
from tools.train_ruler_direct import direct_loss, stage_loss, subray_distribution, subray_set_loss


def test_fixed_cell_supervision_does_not_follow_moving_offsets():
    # 两个不重叠单元以3米为界。GT位于右单元，当前左预测却更近。
    scores = torch.tensor([0., 0.]).reshape(1, 2, 1, 1).requires_grad_()
    depths = torch.tensor([2.9, 3.5]).reshape_as(scores).requires_grad_()
    stage = {"scores": scores, "candidate_depth": depths,
             "sampling_depth": torch.tensor([2., 4.]).reshape_as(scores),
             "valid": torch.ones_like(scores, dtype=torch.bool), "probability": scores.softmax(1)}
    truth = torch.tensor([[[[3.05]]]])
    mask = torch.ones_like(truth, dtype=torch.bool)
    old = stage_loss(stage, truth, mask)
    new = stage_loss(stage, truth, mask, label_depth="sampling")
    old_gradient = torch.autograd.grad(old["classification"], scores, retain_graph=True)[0]
    new_gradient = torch.autograd.grad(new["classification"], scores, retain_graph=True)[0]
    assert old_gradient[0, 0, 0, 0] < 0
    assert new_gradient[0, 1, 0, 0] < 0
    gradient = torch.autograd.grad(new["location"], depths)[0]
    assert gradient[0, 0, 0, 0] == 0
    assert gradient[0, 1, 0, 0] > 0
    torch.testing.assert_close(old["risk"], new["risk"], atol=0, rtol=0)


def test_readout_adaptation_preserves_candidates_and_frozen_parameters(tmp_path):
    from tools.train_ruler_direct import configure_training

    settings = tiny_settings(use_fine_offset=False)
    model = RulerDirectDecoder(settings)
    path = tmp_path / "initial.pt"
    torch.save({"model": model.state_dict(), "plan": {"model": settings}}, path)
    configure_training(model, {"model": settings, "init_checkpoint": str(path),
                               "training": {"scope": "readout"}})
    inputs, targets = fixture_inputs()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    prediction = model(inputs)
    coarse = prediction["coarse"]["depth"].detach().clone()
    fine = prediction["fine"]["depth"].detach().clone()
    target = {"depth": targets["depth"][:, :1], "valid": targets["depth_valid"][:, :1]}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = initializer_loss(prediction["depth"], target,
                            {"metric_normalizer_m": 20., "log_weight": .2, "gradient_weight": .05})
    loss["total"].backward()
    optimizer.step()
    assert any(not torch.equal(before[key], value) for key, value in model.state_dict().items()
               if key.startswith("upsample_mask."))
    for key, value in model.state_dict().items():
        if not key.startswith("upsample_mask."):
            torch.testing.assert_close(before[key], value, rtol=0, atol=0)
    after = model(inputs)
    torch.testing.assert_close(after["coarse"]["depth"], coarse, rtol=0, atol=0)
    torch.testing.assert_close(after["fine"]["depth"], fine, rtol=0, atol=0)


def make_stage(sampling=(3., 6., 9.), continuous=None):
    sampling = torch.tensor(sampling).reshape(1, -1, 1, 1)
    depths = sampling.clone() if continuous is None else torch.tensor(continuous).reshape_as(sampling)
    depths.requires_grad_()
    scores = torch.zeros_like(sampling, requires_grad=True)
    return {"scores": scores, "probability": scores.softmax(1), "sampling_depth": sampling,
            "candidate_depth": depths, "valid": torch.ones_like(sampling, dtype=torch.bool)}


def full_targets(children=(3., 3., 9., 9.), masks=(True, True, True, True)):
    truth = torch.full((1, 1, 8, 8), float("nan"))
    valid = torch.zeros_like(truth, dtype=torch.bool)
    for depth, known, (dy, dx) in zip(children, masks, ((0, 0), (0, 4), (4, 0), (4, 4))):
        truth[0, 0, dy, dx] = depth
        valid[0, 0, dy, dx] = known
    return truth, valid


def test_two_surfaces_retain_both_modes_and_set_risk():
    stage = make_stage()
    truth, valid = full_targets()
    distribution, children, known = subray_distribution(stage, truth, valid)
    torch.testing.assert_close(distribution.flatten(), torch.tensor([0.5, 0., 0.5]))
    loss = subray_set_loss(stage, truth, valid)
    torch.testing.assert_close(loss["classification"], torch.tensor(math.log(3.)))
    # 中间6米点距两个真实面均3米，两个真实面本身不互相惩罚。
    torch.testing.assert_close(loss["risk"], torch.tensor(1. / 20))
    assert loss["location"] == 0
    sum(loss.values()).backward()
    assert stage["scores"].grad[0, 1] > 0
    assert stage["scores"].grad[0, 0] < 0 and stage["scores"].grad[0, 2] < 0


def test_single_surface_matches_center_when_nearest_class_agrees():
    stage = make_stage(continuous=(3.1, 6.1, 9.1))
    truth, valid = full_targets((6.4,) * 4)
    expected = stage_loss(stage, truth[..., ::8, ::8], valid[..., ::8, ::8])
    actual = subray_set_loss(stage, truth, valid)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=1e-7, rtol=1e-6)


def test_classification_uses_fixed_sampling_even_if_offset_moves_nearest_class():
    stage = make_stage(sampling=(3., 6., 9.), continuous=(4., 4.5, 9.))
    truth, valid = full_targets((4.2,) * 4)
    q, _, _ = subray_distribution(stage, truth, valid)
    # 固定格最近的是3m，但连续预测最近是4m，仍为同一类。
    assert q[0, 0] == 1
    with torch.no_grad():
        stage["candidate_depth"][0, 0] = 3.1
    q_after, _, _ = subray_distribution(stage, truth, valid)
    torch.testing.assert_close(q_after, q, atol=0, rtol=0)
    assert (stage["candidate_depth"] - 4.2).abs().argmin(1).item() == 1


def test_missing_subrays_are_not_turned_into_surfaces():
    stage = make_stage()
    truth, valid = full_targets((3., float("nan"), 9., 77.), (True, False, True, False))
    q, _, known = subray_distribution(stage, truth, valid)
    torch.testing.assert_close(q.flatten(), torch.tensor([0.5, 0., 0.5]))
    assert known.sum() == 2
    losses = subray_set_loss(stage, truth, valid)
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert torch.isfinite(stage["candidate_depth"].grad).all()


@pytest.mark.parametrize("missing_labels", [True, False])
def test_no_supported_labels_returns_differentiable_zero(missing_labels):
    stage = make_stage()
    truth, valid = full_targets()
    if missing_labels:
        valid[:] = False
        truth[:] = float("nan")
    else:
        stage["valid"][:] = False
    losses = subray_set_loss(stage, truth, valid)
    assert all(value.item() == 0 and value.requires_grad for value in losses.values())
    sum(losses.values()).backward()
    assert stage["scores"].grad.abs().sum() == 0
    assert stage["candidate_depth"].grad.abs().sum() == 0


def test_subray_layout_matches_nearest_proposal_propagation_multiple_batches():
    sample = torch.arange(2 * 16 * 24).reshape(2, 1, 16, 24).float() + 1
    stage = {"scores": torch.zeros(2, 3, 2, 3), "sampling_depth": torch.ones(2, 3, 2, 3),
             "valid": torch.ones(2, 3, 2, 3, dtype=torch.bool)}
    _, children, valid = subray_distribution(stage, sample, torch.ones_like(sample, dtype=torch.bool))
    for child, (dy, dx) in enumerate(((0, 0), (0, 4), (4, 0), (4, 4))):
        torch.testing.assert_close(children[:, child], sample[:, 0, dy::8, dx::8])
    assert valid.all()


def test_default_center_matches_legacy_formula_exactly():
    model = RulerDirectDecoder(tiny_settings()).eval()
    inputs, targets = fixture_inputs()
    targets = {"depth": targets["depth"][:, :1], "valid": targets["depth_valid"][:, :1]}
    settings = {"metric_normalizer_m": 20., "log_weight": 0.2, "gradient_weight": 0.05}
    result = model(inputs)
    expected = initializer_loss(result["depth"], targets, settings)
    for name, stride in (("coarse", 8), ("fine", 4)):
        part = stage_loss(result[name], targets["depth"][..., ::stride, ::stride], targets["valid"][..., ::stride, ::stride])
        expected.update({name + "_" + key: value for key, value in part.items()})
        expected["total"] = expected["total"] + 1.0 * (part["classification"] + 1.0 * part["risk"] + part["location"])
    actual = direct_loss(result, targets, settings)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    explicit = direct_loss(result, targets, dict(settings, coarse_supervision="center"))
    torch.testing.assert_close(actual["total"], explicit["total"], atol=0, rtol=0)


def test_set_loss_backward_and_inference_does_not_read_targets():
    torch.manual_seed(11)
    model = RulerDirectDecoder(tiny_settings(use_fine_offset=False)).eval()
    inputs, target = fixture_inputs(batch=2)
    targets = {"depth": target["depth"][:, :1], "valid": target["depth_valid"][:, :1]}
    targets["valid"][1, :, ::8, ::8] = False
    settings = {"metric_normalizer_m": 20., "log_weight": 0.2, "gradient_weight": 0.05,
                "coarse_supervision": "subray_set"}
    output = model(inputs)
    loss = direct_loss(output, targets, settings)
    loss["total"].backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    with torch.no_grad():
        before = model(inputs)
        inputs["targets"] = {"depth": torch.full_like(targets["depth"], 12345.)}
        after = model(inputs)
    torch.testing.assert_close(before["depth"], after["depth"], atol=0, rtol=0)
    center = direct_loss(output, targets, dict(settings, coarse_supervision="center"))
    for key in ("classification", "risk", "location"):
        torch.testing.assert_close(center["fine_" + key], loss["fine_" + key], atol=0, rtol=0)


def test_entire_unlabeled_batch_returns_zero_in_set_mode():
    model = RulerDirectDecoder(tiny_settings()).eval()
    inputs, target = fixture_inputs()
    targets = {"depth": torch.full_like(target["depth"][:, :1], float("nan")),
               "valid": torch.zeros_like(target["depth_valid"][:, :1])}
    settings = {"metric_normalizer_m": 20., "log_weight": 0.2, "gradient_weight": 0.05,
                "coarse_supervision": "subray_set"}
    loss = direct_loss(model(inputs), targets, settings)
    assert all(value.item() == 0 and value.requires_grad for value in loss.values())
    loss["total"].backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
