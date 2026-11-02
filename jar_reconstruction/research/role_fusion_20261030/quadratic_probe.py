"""窄区间严格门控的三点抛物线诊断；固定父组，不训练，不读TEST。"""

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.runtime import move, seed_all
from research.early_geometry_20261030.pilot import load_model, path
from research.near_evidence_20261030.diagnose import add_group, finish
from research.near_mid_evaluation_20261028.metrics import ground_truth, regions
from tools.smoke_outdoor_datasets import offline_only


def vertex(z, scores, valid, old, near, far):
    """z/scores/valid=[B,3,H,W]；只对指定父组三点作线性Z坐标拟合。"""
    left, center, right = z.unbind(1)
    h_left, h_right = center - left, right - center
    step = .5 * (h_left + h_right)
    curvature = scores[:, 0] - 2 * scores[:, 1] + scores[:, 2]
    eligible = valid.all(1) & torch.isfinite(z).all(1) & torch.isfinite(scores).all(1)
    eligible &= (h_left > 0) & (h_right > 0)
    eligible &= (h_left - h_right).abs() <= (1e-4 * step.abs() + 1e-7)
    eligible &= (left > near) & (right < far)
    eligible &= (right - left) < .5 * center
    eligible &= curvature < -1e-6
    denominator = torch.where(curvature < -1e-6, curvature, -torch.ones_like(curvature))
    fraction = .5 * (scores[:, 0] - scores[:, 2]) / denominator
    estimate = center + step * fraction
    eligible &= (fraction >= -1) & (fraction <= 1) & torch.isfinite(estimate)
    return torch.where(eligible, estimate, old), eligible


def unit_tests():
    z = torch.tensor([9., 10., 11.]).reshape(1, 3, 1, 1)
    scores = -(z - 10.25).square()
    valid = torch.ones_like(z, dtype=torch.bool)
    old = torch.tensor([[[10.]]])
    result, gate = vertex(z, scores, valid, old, 1., 100.)
    torch.testing.assert_close(result, old + .25)
    assert bool(gate)
    _, wide = vertex(z - 7, scores, valid, old, 1., 100.)
    assert not bool(wide)
    _, convex = vertex(z, -scores, valid, old, 1., 100.)
    assert not bool(convex)
    _, duplicate = vertex(z.clamp_min(10), scores, valid, old, 1., 100.)
    assert not bool(duplicate)
    scaled, same_gate = vertex(z * 17, scores, valid, old * 17, 17., 1700.)
    torch.testing.assert_close(scaled, result * 17)
    assert torch.equal(gate, same_gate)
    return {"checks": 5, "passed": True}


def selected_group(value, index):
    grouped = value.unflatten(1, (-1, 3))
    return grouped.gather(1, (index // 3)[:, :, None].expand(-1, -1, 3, -1, -1))[:, 0]


@torch.no_grad()
def run():
    offline_only()
    seed_all(20261030)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    output = ROOT / "runs/active/role_fusion_20261030/diagnosis/quadratic.json"
    if output.exists():
        raise FileExistsError(output)
    report = {"completed": False, "unit_tests": unit_tests(), "domains": {},
              "test_accessed": False, "training_performed": False,
              "scope": "原固定VAL每域前4development，共16前向。共同stride8、相同GT；仅细层读出，不声称完整最终深度改善。",
              "rule": "原fine所选父组，三点等距无重复且不触near/far，完整span/center<.5，二次差分<−1e−6，顶点在组三点范围内；否则回原fine预测。",
              "source_sha256": digest(__file__)}
    jobs = [("public", ["Hypersim_subset", "DynamicReplica_valid", "MatrixCity_street"]),
            ("carla", ["CARLA_new_stereo_up"])]
    for group, names in jobs:
        previous = ROOT / f"runs/active/early_geometry_20261030/{group}_control"
        settings = json.loads((previous / "settings.json").read_text())
        plan = json.loads((previous / "plan.json").read_text())["validation"]
        model, saved = load_model(settings["checkpoint"], device)
        source = json.loads(path(settings["data_config"]).read_text())
        config = copy.deepcopy(saved["config"])
        config["datasets"] = [d for d in source.get("validation_datasets", source["datasets"]) if d["name"] in names]
        config.pop("validation_datasets", None)
        for domain in config["datasets"]:
            domain["manifests"] = {"val": domain["manifests"]["val"]}
        data = MixedDataset(config, "val")
        for domain_index, domain in enumerate(data.domains):
            totals = {}
            indices = plan[domain["name"]]["development"][:4]
            for index in indices:
                batch = move(collate_mixed([data[(domain_index, index, -1)]]), device)
                native = model(batch["inputs"], domain, reference=0)
                fine = native["fine"]
                chosen = fine["selected_index"]
                z = selected_group(fine["sampling_depth"], chosen)
                s = selected_group(fine["scores"], chosen)
                valid = selected_group(fine["valid"], chosen)
                old = fine["depth"][:, 0]
                estimate, gate = vertex(z, s, valid, old, model.decoder.near, model.decoder.far)
                scale = float(native["unit_scale"])
                truth_full, mask_full = ground_truth(batch, dict(domain, reference_index=0))
                truth = truth_full[:, 0, ::8, ::8].float()
                masks = {key: value[:, 0, ::8, ::8] for key, value in regions(truth_full, mask_full, domain).items()}
                old, estimate = old[..., ::2, ::2] * scale, estimate[..., ::2, ::2] * scale
                gate = gate[..., ::2, ::2]
                values = {"original_MAE": (old - truth).abs(), "quadratic_MAE": (estimate - truth).abs(),
                          "original_MSE": (old - truth).square(), "quadratic_MSE": (estimate - truth).square(),
                          "eligible_fraction": gate.float(), "absolute_update": (estimate - old).abs()}
                masks["near_eligible"] = masks["near"] & gate
                masks["full_eligible"] = masks["full"] & gate
                for key, mask in masks.items():
                    add_group(totals, key, mask, values)
            report["domains"][domain["name"]] = {"units": domain["geometry_units"], "indices": indices,
                                                     "metrics": finish(totals)}
            print(json.dumps({"domain": domain["name"], "near": finish(totals)["near"]["means"]}), flush=True)
        del model, saved, data
        torch.cuda.empty_cache()
    report["completed"] = True
    write_json(output, report)


if __name__ == "__main__":
    run()
