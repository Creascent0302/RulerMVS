"""粗细交接的固定VAL诊断；不训练，GT只在原前向结束后统计。"""

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

import jar.models.ruler_proposals as proposals
from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.runtime import move, seed_all
from research.early_geometry_20261030.pilot import load_model, path
from research.near_evidence_20261030.diagnose import add_group, finish
from research.near_mid_evaluation_20261028.metrics import ground_truth, regions
from tools.smoke_outdoor_datasets import offline_only

OUTPUT = ROOT / "runs/active/role_fusion_20261030/diagnosis"


@contextmanager
def capture(decoder):
    original_prediction = decoder._prediction
    original_transport = proposals.transport_proposals
    records = {}

    def predict(self, features, candidates, valid, similarity, support,
                lower, upper, observation, use_offset=True):
        result = original_prediction(features, candidates, valid, similarity,
                                     support, lower, upper, observation, use_offset)
        records[observation["stride"]] = {"similarity": similarity}
        return result

    def transport(model, coarse, features, observation, prepared):
        result = original_transport(model, coarse, features, observation, prepared)
        # 独立重建同一无学习参数选择，只读取粗后验，检验与实际候选中心一致。
        flow = proposals.SubrayTransport(coarse, prepared, model.proposal_transport)
        available = flow.valid.clone()
        chosen = []
        bins = torch.arange(flow.probability.shape[1], device=features.device)[None, :, None, None]
        for _ in range(model.peak_count):
            active = available.any(1, keepdim=True)
            index = flow.probability.masked_fill(~available, -1).argmax(1, keepdim=True)
            index = torch.where(active, index, flow.probability.argmax(1, keepdim=True))
            chosen.append(index)
            center = flow.depth.gather(1, index)
            available &= (flow.depth - center).abs() >= float(model.settings.get("peak_separation_m", 1.))
            exclusion = int(model.settings.get("peak_exclusion_bins", 0))
            if exclusion:
                available &= (bins - index).abs() > exclusion
            available.scatter_(1, index, False)
        indices = torch.cat(chosen, 1)
        centers = flow.depth.gather(1, indices).clamp(model.near, model.far)
        actual = result[0].unflatten(1, (model.peak_count, 3))[:, :, 1]
        torch.testing.assert_close(centers, actual, atol=0, rtol=0)
        records["parent_probability"] = flow.probability.gather(1, indices)
        records["parent_centers"] = centers
        return result

    decoder._prediction = types.MethodType(predict, decoder)
    proposals.transport_proposals = transport
    try:
        yield records
    finally:
        decoder._prediction = original_prediction
        proposals.transport_proposals = original_transport


def gather(value, index):
    return value.gather(1, index)[:, 0]


def analyze(batch, domain, native, captured):
    truth_full, valid_full = ground_truth(batch, dict(domain, reference_index=0))
    truth = truth_full[:, 0, ::8, ::8].float()
    masks = {name: value[:, 0, ::8, ::8] for name, value in regions(truth_full, valid_full, domain).items()}
    masks["near_edge"] = masks["near"] & masks["gradient_band"]
    masks["near_smooth"] = masks["near"] & masks["smooth"]
    scale = float(native["unit_scale"])
    coarse = native["coarse"]
    fine = {name: value[..., ::2, ::2] for name, value in native["fine"].items()
            if torch.is_tensor(value) and value.ndim >= 3}
    coarse_depth = coarse["depth"][:, 0] * scale
    fine_depth = fine["depth"][:, 0] * scale
    continuous = fine["candidate_depth"] * scale
    sampled = fine["sampling_depth"] * scale
    valid = fine["valid"]
    selected = fine["selected_index"]
    error = (continuous - truth[:, None]).abs().masked_fill(~valid, torch.inf)
    sample_error = (sampled - truth[:, None]).abs().masked_fill(~valid, torch.inf)
    group_error = error.unflatten(1, (-1, 3)).amin(2)
    selected_group = selected // 3
    chosen_group_best = gather(group_error, selected_group)
    fine_error = (fine_depth - truth).abs()
    coarse_error = (coarse_depth - truth).abs()
    coarse_good = coarse_error <= .05 * truth
    fine_bad = fine_error > .05 * truth
    sampling_keep = (sampled - coarse_depth[:, None]).abs().amin(1)
    continuous_keep = (continuous - coarse_depth[:, None]).abs().amin(1)
    epsilon = 1e-5 * coarse_depth.abs().clamp_min(1.)
    parent = captured["parent_probability"][..., ::2, ::2]
    log_prior = parent.clamp_min(1e-12).log().repeat_interleave(3, 1)
    scores = fine["scores"]
    cosine = captured[4]["similarity"][..., ::2, ::2]
    probe_scores = {
        "prior05": scores + .5 * log_prior,
        "prior10": scores + log_prior,
        "learned_only": scores - 4 * cosine,
        "cosine_only": 4 * cosine,
    }
    conditional = scores - scores.unflatten(1, (-1, 3)).logsumexp(2, keepdim=True).expand(-1, -1, 3, -1, -1).flatten(1, 2)
    probe_scores["parent_conditional"] = conditional + log_prior
    estimates = {"coarse": coarse_depth, "fine": fine_depth,
                 "final": native["depth"][:, 0, ::8, ::8] * scale,
                 "sample_oracle": gather(sampled, sample_error.argmin(1, keepdim=True)),
                 "continuous_oracle": gather(continuous, error.argmin(1, keepdim=True))}
    for name, logits in probe_scores.items():
        index = logits.masked_fill(~valid, -torch.inf).argmax(1, keepdim=True)
        estimates[name] = gather(continuous, index)
    first_index = scores[:, :3].argmax(1, keepdim=True)
    estimates["first_parent_only"] = gather(continuous, first_index)
    values = {}
    for name, estimate in estimates.items():
        if not torch.isfinite(estimate[masks["full"]]).all():
            raise RuntimeError(f"{name}预测非有限")
        delta = estimate - truth
        values[name + "_MAE"] = delta.abs()
        values[name + "_MSE"] = delta.square()
        values[name + "_AbsRel"] = delta.abs() / truth.clamp_min(1e-12)
    values.update({
        "coarse_to_fine_degradation": fine_error - coarse_error,
        "degradation_positive_mass": (fine_error - coarse_error).clamp_min(0),
        "improvement_positive_mass": (coarse_error - fine_error).clamp_min(0),
        "coarse_good5pct": coarse_good.float(),
        "coarse_good_fine_bad5pct": (coarse_good & fine_bad).float(),
        "fine_good_coarse_bad5pct": (~coarse_good & ~fine_bad).float(),
        "coarse_center_retained_sampling": (sampling_keep <= epsilon).float(),
        "coarse_center_retained_continuous": (continuous_keep <= epsilon).float(),
        "nearest_sample_to_coarse_delta": sampling_keep,
        "nearest_continuous_to_coarse_delta": continuous_keep,
        "chosen_group_best_error": chosen_group_best,
        "group_selection_regret": chosen_group_best - error.amin(1),
        "within_group_selection_regret": fine_error - chosen_group_best,
        "selected_sampling_MAE": gather(sample_error, selected),
        "chosen_offset_error_change": fine_error - gather(sample_error, selected),
        "selected_first_parent": (selected_group[:, 0] == 0).float(),
        "selected_middle_sample": ((selected[:, 0] % 3) == 1).float(),
        "final_fine_group_oracle": error.amin(1),
        "depth": truth,
    })
    masks["near_coarse_good"] = masks["near"] & coarse_good
    masks["near_coarse_good_fine_bad"] = masks["near"] & coarse_good & fine_bad
    masks["near_sample_missing5pct"] = masks["near"] & (sample_error.amin(1) > .05 * truth)
    masks["near_continuous_available5pct"] = masks["near"] & (error.amin(1) <= .05 * truth)
    return masks, values


@torch.no_grad()
def run():
    offline_only()
    seed_all(20261030)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"拒绝覆盖：{OUTPUT}")
    OUTPUT.mkdir(parents=True)
    report = {"completed": False, "test_accessed": False, "training_performed": False,
              "protocol": "原best，历史16VAL，两分区，同stride8与固定GT分母。probe仅细候选离线重评分，不是最终完整模型输出。",
              "correct_definition": "absolute error <= .05 * GT depth; oracle only offline with GT",
              "domains": {}, "weights": {}, "sources": {str(Path(__file__).relative_to(ROOT)): digest(__file__)}}
    jobs = [
        ("public", ["Hypersim_subset", "DynamicReplica_valid", "MatrixCity_street"]),
        ("carla", ["CARLA_new_stereo_up"]),
    ]
    for group, names in jobs:
        previous = ROOT / f"runs/active/early_geometry_20261030/{group}_control"
        settings = json.loads((previous / "settings.json").read_text())
        plan = json.loads((previous / "plan.json").read_text())["validation"]
        with torch.inference_mode(False):
            model, saved = load_model(settings["checkpoint"], device)
        source = json.loads(path(settings["data_config"]).read_text())
        config = copy.deepcopy(saved["config"])
        config["datasets"] = [d for d in source.get("validation_datasets", source["datasets"]) if d["name"] in names]
        for domain in config["datasets"]:
            domain["manifests"] = {"val": domain["manifests"]["val"]}
        config.pop("validation_datasets", None)
        data = MixedDataset(config, "val")
        report["weights"][group] = {"path": settings["checkpoint"], "sha256": digest(path(settings["checkpoint"]))}
        for domain_index, domain in enumerate(data.domains):
            domain_plan = plan[domain["name"]]
            if digest(path(domain["manifests"]["val"])) != domain_plan["manifest_sha256"]:
                raise RuntimeError("VAL清单改变")
            aggregates = {"all": {}, "development": {}, "confirmation": {}}
            frames = []
            for index in domain_plan["indices"]:
                batch = move(collate_mixed([data[(domain_index, index, -1)]]), device)
                with capture(model.decoder) as records:
                    native = model(batch["inputs"], domain, reference=0)
                masks, values = analyze(batch, domain, native, records)
                part = "development" if index in domain_plan["development"] else "confirmation"
                for label, mask in masks.items():
                    add_group(aggregates["all"], label, mask, values)
                    add_group(aggregates[part], label, mask, values)
                one = {}
                add_group(one, "near", masks["near"], values)
                frames.append({"index": index, "partition": part, **finish(one)})
            report["domains"][domain["name"]] = {
                "units": domain["geometry_units"], "plan": domain_plan,
                "metrics": {name: finish(values) for name, values in aggregates.items()}, "frames": frames}
            write_json(OUTPUT / "summary.json", report)
            print(json.dumps({"domain": domain["name"], "near": report["domains"][domain["name"]]["metrics"]["all"]["near"]["means"]}), flush=True)
        del model, saved, data
        torch.cuda.empty_cache()
    report["completed"] = True
    write_json(OUTPUT / "summary.json", report)


if __name__ == "__main__":
    run()
