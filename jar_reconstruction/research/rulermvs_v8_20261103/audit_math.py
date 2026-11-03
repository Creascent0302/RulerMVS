"""CPU-only structural probes; no trained weights, dataset, or accuracy claims."""

import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from jar.pretraining.model import CandidateViewAttention


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261103)
    source_attention = CandidateViewAttention(64).eval()
    # Exact equal evidence fixture: all source/null logits zero, pair value one.
    with torch.no_grad():
        source_attention.query.weight.zero_()
        source_attention.pair[-1].weight.zero_()
        source_attention.pair[-1].bias.fill_(1.)
    reference = torch.zeros(1, 64, 1, 1)
    reference[:, 0] = 1.
    cardinality = []
    for count in (1, 3, 7):
        source = reference[:, None, :, None].expand(1, count, 64, 1, 1, 1)
        valid = torch.ones(1, count, 1, 1, 1, dtype=torch.bool)
        derivative = torch.zeros(1, count, 2, 1, 1, 1)
        derivative[:, :, 0] = 2.
        with torch.no_grad():
            actual = source_attention(reference, source, valid, derivative, torch.tensor([.25]))
        balanced = torch.cat((torch.zeros(count), torch.tensor([math.log(count / 3)]))).softmax(0)
        cardinality.append({
            "valid_sources": count,
            "actual_null": float(actual["null_mass"].item()),
            "actual_pooled_first_channel": float(actual["pooled"][0, 0].item()),
            "actual_4cos_bypass": float(4 * actual["weights"].sum().item()),
            "actual_information": float(actual["information"].item()),
            "balanced_null_preserve_three_sources": float(balanced[-1]),
        })

    # Fixed physical candidate; depth_scale changes only the internal coordinate.
    source = reference[:, None, :, None]
    valid = torch.ones(1, 1, 1, 1, 1, dtype=torch.bool)
    units = []
    for scale in (.5, 1., 2.):
        derivative = torch.zeros(1, 1, 2, 1, 1, 1)
        derivative[:, :, 0] = 2. * scale
        with torch.no_grad():
            actual = source_attention(reference, source, valid, derivative, torch.tensor([.25 / scale]))
        information = float(actual["information"].item())
        units.append({
            "internal_depth_scale": scale,
            "actual_information": information,
            "metric_input": math.log1p(information),
            "cell_span": float((derivative[:, :, 0] * (.25 / scale)).item()),
            "log_depth_information": information * (10. / scale) ** 2,
        })

    positions = []
    for near, far in ((.1, 100.), (1., 100.), (.1, 1000.)):
        eta = (math.log(10.) - math.log(near)) / math.log(far / near)
        positions.append({"candidate": 10., "near": near, "far": far,
                          "position": [eta, math.sin(math.pi * eta), math.cos(math.pi * eta)]})

    lattice = torch.logspace(-1, 2, 96)
    separation = []
    for center in (.1, 1., 10., 50.):
        chosen = int((lattice - center).abs().argmin())
        center = float(lattice[chosen])
        suppressed = ((lattice - center).abs() < .1).nonzero().flatten()
        separation.append({"center": center, "excluded_bins_by_fixed_0_1": int(len(suppressed)),
                           "fractional_radius": .1 / center})
    result = {
        "scope": "CPU synthetic current CandidateViewAttention and exact algebra; no training, GPU, dataset or accuracy evaluation",
        "source_cardinality": cardinality,
        "internal_unit_response": units,
        "range_position_response": positions,
        "fixed_internal_peak_separation": separation,
    }
    assert all(abs(row["balanced_null_preserve_three_sources"] - .25) < 1e-7 for row in cardinality)
    assert all(abs(row["cell_span"] - .5) < 1e-7 for row in units)
    assert all(abs(row["log_depth_information"] - 200.) < 1e-7 for row in units)
    destination = Path(__file__).with_name("audit_math.json")
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
