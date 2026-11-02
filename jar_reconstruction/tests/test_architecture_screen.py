import copy

import pytest
import torch

from tools.cache_architecture_screen import encoder_digest
from tools.screen_public_architecture import (
    engineering_gate, rgb_paths, select_partition, verify_common_protocol,
)


def records():
    return [
        {"sequence": f"scene_{scene}", "frame_id": index,
         "cameras": [{"rgb": f"scene_{scene}/{index + offset:04d}.png"} for offset in range(4)]}
        for scene in range(3) for index in range(100)
    ]


def test_partition_is_reproducible_and_images_are_disjoint():
    rows = records()
    first = select_partition(rows, 8, 32, 21)
    assert first == select_partition(rows, 8, 32, 21)
    assert len(first["development"]) == 8
    assert len(first["confirmation"]) == 32
    assert not set(first["development"]) & set(first["confirmation"])
    train_images = set().union(*(rgb_paths(rows[i]) for i in first["development"]))
    confirm_images = set().union(*(rgb_paths(rows[i]) for i in first["confirmation"]))
    assert not train_images & confirm_images
    assert {rows[i]["sequence"] for i in first["development"]} == {"scene_0", "scene_1", "scene_2"}


def test_partition_does_not_silently_repeat_samples():
    with pytest.raises(ValueError, match="候选样本不足"):
        select_partition(records()[:6], 8, 32, 21)


def test_gate_catches_single_domain_regression_hidden_by_average():
    base = {"mae": 1., "rmse": 2., "abs_rel": .1}
    domains = {name: {"old": {"full": copy.deepcopy(base)},
                      "new": {"full": copy.deepcopy(base)}} for name in ("a", "b")}
    domains["a"]["new"]["full"]["mae"] = 1.03
    domains["b"]["new"]["full"]["mae"] = .96
    tolerance = {"macro_relative_degradation": .005, "per_domain_relative_degradation": .02}
    gate = engineering_gate(domains, "old", ["old", "new"], tolerance)["new"]
    assert gate["macro_relative_change"]["mae"] < 0
    assert not gate["passes_engineering_screen"]
    assert gate["failures"] == ["a/mae"]


def test_gate_never_averages_native_units():
    domains = {
        "meters": {"old": {"full": {"mae": 1., "rmse": 2., "abs_rel": .1}},
                   "new": {"full": {"mae": 1.01, "rmse": 2., "abs_rel": .1}}},
        "scene_units": {"old": {"full": {"mae": 1000., "rmse": 2000., "abs_rel": .1}},
                        "new": {"full": {"mae": 990., "rmse": 2000., "abs_rel": .1}}},
    }
    tolerance = {"macro_relative_degradation": .005, "per_domain_relative_degradation": .02}
    gate = engineering_gate(domains, "old", ["old", "new"], tolerance)["new"]
    assert abs(gate["macro_relative_change"]["mae"]) < 1e-12
    assert gate["passes_engineering_screen"]


def test_gate_checks_far_and_edge_errors_when_full_metrics_hide_regression():
    metrics = {"mae": 1., "rmse": 2., "abs_rel": .1, "count": 100}
    previous = {"full": metrics, "groups": {"quartile_4": metrics, "edge_5px": metrics}}
    candidate = copy.deepcopy(previous)
    candidate["groups"]["quartile_4"] = dict(metrics, abs_rel=.103)
    candidate["groups"]["edge_5px"] = dict(metrics, rmse=2.06)
    tolerance = {"macro_relative_degradation": .005, "per_domain_relative_degradation": .02}
    result = engineering_gate({"a": {"old": previous, "new": candidate}},
                              "old", ["old", "new"], tolerance)["new"]
    assert not result["passes_engineering_screen"]
    assert set(result["failures"]) == {"a/quartile_4/abs_rel", "a/edge_5px/rmse"}


def test_encoder_digest_tracks_encoder_but_not_decoder_changes():
    state = {"foundation.block.weight": torch.tensor([1., 2.]),
             "matching.block.weight": torch.tensor([3.]),
             "decoder.score.weight": torch.tensor([4.])}
    original = encoder_digest(state)
    state["decoder.score.weight"] += 1
    assert encoder_digest(state) == original
    state["matching.block.weight"] += 1
    assert encoder_digest(state) != original


def test_swapped_checkpoint_must_share_pixel_and_geometry_protocol():
    data = {"image_size": [256, 448], "matching_image_size": [512, 896],
            "datasets": [{"name": "scene", "near": 2., "far": 80., "depth_scale": 1.,
                          "geometry_units": "meters", "num_views": 4, "reference_index": 2}]}
    verify_common_protocol(copy.deepcopy(data), data)
    changed = copy.deepcopy(data)
    changed["datasets"][0]["far"] = 100.
    with pytest.raises(ValueError, match="scene/far"):
        verify_common_protocol(changed, data)
    changed = copy.deepcopy(data)
    changed["matching_image_size"] = [256, 448]
    with pytest.raises(ValueError, match="matching_image_size"):
        verify_common_protocol(changed, data)
