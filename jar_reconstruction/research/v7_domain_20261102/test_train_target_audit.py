"""只读审计量的解析例子；不需要数据或GPU。"""

import torch

from .train_target_audit import candidate_maps, quality_maps, retention_maps, summarize, add_maps


def stage(offset=True):
    def tensor(values):
        return torch.tensor(values).reshape(1, -1, 1, 1)
    lattice = tensor([1., 2., 3.])
    return {"sampling_depth": lattice,
            "candidate_depth": tensor([1.49, 2.49, 2.99]) if offset else lattice,
            "valid": torch.ones_like(lattice, dtype=torch.bool),
            "lower": tensor([1., 1.5, 2.5]), "upper": tensor([1.5, 2.5, 3.]),
            "selected_index": torch.ones(1, 1, 1, 1, dtype=torch.long)}


def test_grid_label_is_not_always_closest_actual_candidate():
    truth = torch.tensor([[[[1.6]]]])
    mask = torch.ones_like(truth, dtype=torch.bool)
    maps, valid, grid, actual = candidate_maps(stage(), truth, mask, 1.)
    assert grid.item() == 1 and actual.item() == 0
    assert maps["label_disagreement"].item() == 1
    assert maps["grid_label_unreachable"].item() == 0
    assert maps["actual_best_label_unreachable"].item() == 1
    torch.testing.assert_close(maps["grid_label_actual_native_regret"], truth.new_tensor(.78).reshape_as(truth))
    totals = {}
    add_maps(totals, maps, valid)
    assert summarize(totals)["label_distance_bins"]["mean"] == 1


def test_zero_offset_keeps_log_label():
    truth = torch.tensor([[[[1.6]]]])
    maps, _, _, _ = candidate_maps(stage(False), truth, torch.ones_like(truth, dtype=torch.bool), 1.)
    assert maps["label_disagreement"].item() == 0
    assert maps["grid_label_actual_log_regret"].item() == 0


def test_information_recomputed_from_actual_source_attention():
    candidates = torch.full((1, 3, 2, 2), 2.)
    source = {"information": torch.ones(1, 1, 3, 2, 2),
              "weights": torch.full((1, 2, 3, 2, 2), .25),
              "null_mass": torch.full((1, 1, 3, 2, 2), .2)}
    derivative = torch.ones(1, 2, 2, 3, 2, 2)
    maps = quality_maps(source, candidates, derivative)
    torch.testing.assert_close(maps["information_gate"], candidates.new_full(candidates.shape, .2 + .8 * .25 / 4.25))
    assert bool((maps["information_gate"] == maps["recomputed_gate"]).all())
    source["null_mass"].fill_(1.)
    assert bool((quality_maps(source, candidates, derivative)["information_gate"] == 1).all())


def test_retention_can_be_positive_when_mean_error_improves():
    truth = torch.ones(1, 1, 1, 2)
    current = {"depth": torch.tensor([[[[1., 2.]]]])}
    teacher = {"depth": torch.tensor([[[[2., 1.5]]]])}
    maps = retention_maps(current, teacher, truth, torch.ones_like(truth, dtype=torch.bool), 1.)
    assert maps["log_error_change"].mean() < 0
    assert maps["hinge_retention"].mean() > 0
    assert maps["worse_fraction"].mean() == .5
