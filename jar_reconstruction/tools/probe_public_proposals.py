"""固定权重诊断四峰是否浪费在相邻刻度；不修改生产候选函数。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.models import ruler_proposals
from jar.models.metric_ruler_network import depth_cell_edges
from jar.pretraining.comparison import write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.pretraining.model import PretrainingModel
from jar.runtime import move
from tools.probe_public_stages import diagnose, summarize, sum_records


def proposal_function(radius, captures, original):
    checked = False

    def propose(model, coarse, features, observation, prepared):
        nonlocal checked
        transport = ruler_proposals.SubrayTransport(coarse, prepared, model.proposal_transport)
        candidates, scores = transport.depth, transport.probability
        available = transport.valid.clone()
        chosen = []
        labels = torch.arange(scores.shape[1], device=scores.device)[None, :, None, None]
        for _ in range(model.peak_count):
            active = available.any(1, keepdim=True)
            index = scores.masked_fill(~available, -1).argmax(1, keepdim=True)
            index = torch.where(active, index, scores.argmax(1, keepdim=True))
            chosen.append(index)
            center = candidates.gather(1, index)
            available &= (candidates - center).abs() >= float(model.settings.get("peak_separation_m", 1.))
            available.scatter_(1, index, False)
            if radius:
                available &= (labels - index).abs() > radius
        indices = torch.cat(chosen, 1)
        peaks = candidates.gather(1, indices)
        edges = depth_cell_edges(model.depth_lattice)
        widths = (edges[1:] - edges[:-1])[indices] * .5
        derivative = torch.linalg.vector_norm(observation["ray_derivative"], dim=2)
        derivative = torch.where(observation["source_valid"], derivative, 0).amax(1)
        g = transport.selected(derivative[:, None], indices)[:, 0]
        steps = torch.maximum(widths, float(model.settings.get("fine_step_pixels", .5)) / g.clamp_min(1e-6))
        steps = steps.clamp(float(model.settings.get("fine_min_step_m", .05)),
                           float(model.settings.get("fine_max_step_m", 8.)))
        offsets = peaks.new_tensor((-1., 0., 1.))[None, None, :, None, None]
        fine = (peaks[:, :, None] + steps[:, :, None] * offsets).flatten(1, 2).clamp(model.near, model.far)
        half_width = steps[:, :, None].expand(-1, -1, 3, -1, -1).flatten(1, 2) * .5
        inherited = transport.selected(features, indices).repeat_interleave(3, 2)
        captures.update(indices=indices, peaks=peaks, steps=steps, valid=transport.valid.any(1, keepdim=True))
        if radius == 0 and not checked:
            expected = original(model, coarse, features, observation, prepared)
            for value, reference in zip((fine, half_width, inherited), expected):
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
            checked = True
        return fine, half_width, inherited

    return propose


def diversity(captured):
    indices, peaks, steps, valid = (captured[key] for key in ("indices", "peaks", "steps", "valid"))
    pairs = [(i, j) for i in range(indices.shape[1]) for j in range(i + 1, indices.shape[1])]
    differences = torch.stack([(indices[:, i] - indices[:, j]).abs() for i, j in pairs], 1)
    overlap = torch.stack([(peaks[:, i] - peaks[:, j]).abs() <= 1.5 * (steps[:, i] + steps[:, j]) for i, j in pairs], 1)
    mask = valid.expand_as(differences)
    ordered = indices.sort(1).values
    gaps = ordered[:, 1:] - ordered[:, :-1]
    valid_gaps = valid.expand_as(gaps)
    span = indices.amax(1, keepdim=True) - indices.amin(1, keepdim=True)
    return {"rays": int(valid.sum()), "pair_count": int(mask.sum()),
            "pair_gap_le1": int(((differences <= 1) & mask).sum()),
            "pair_gap_le2": int(((differences <= 2) & mask).sum()),
            "pair_overlap_interval": int((overlap & mask).sum()),
            "sorted_gap_count": int(valid_gaps.sum()), "sorted_gap_sum": int(gaps[valid_gaps].sum()),
            "index_span_sum": int(span[valid].sum()),
            "four_within_3bins": int(((span <= 3) & valid).sum())}


@torch.no_grad()
def run(plan):
    if plan["split"] != "val":
        raise ValueError("本次固定VAL诊断禁止TEST")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(plan["device"])
    torch.cuda.set_device(device)
    output = Path(plan["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("已有诊断输出，禁止覆盖")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", plan)
    saved = torch.load(plan["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    model = PretrainingModel(saved["config"], initialize=False)
    model.load_state_dict(saved["model"], strict=True)
    model.requires_grad_(False).to(device).eval()
    config = dict(saved["config"], datasets=[d for d in saved["config"]["datasets"] if d["name"] in plan["datasets"]])
    del saved
    dataset = MixedDataset(config, "val")
    capture = {}
    handle = model.decoder.upsample_mask.register_forward_hook(lambda module, inputs, result: capture.update(logits=result))
    original = ruler_proposals.transport_proposals
    final = {}
    try:
        for radius in plan["suppression_radii"]:
            name = "original" if radius == 0 else "suppress_" + str(radius)
            final[name] = {}
            ruler_proposals.transport_proposals = proposal_function(radius, capture, original)
            for domain_index, domain in enumerate(dataset.domains):
                length = len(dataset.datasets[domain_index])
                indices = torch.linspace(0, length - 1, min(length, plan["samples_per_domain"])).long().tolist()
                rows = []
                for index in indices:
                    batch = move(collate_mixed([dataset[(domain_index, index, -1)]]), device)
                    prediction = model(batch["inputs"], domain, reference=0)
                    stats = diagnose(prediction, capture.pop("logits"), batch, domain)
                    stats["diversity"] = diversity(capture)
                    rows.append({"index": index, "sample_id": batch.get("sample_id"), "statistics": stats})
                    capture.clear()
                summary = summarize(sum_records([row["statistics"] for row in rows]))
                final[name][domain["name"]] = summary
                write_json(output / (name + "_" + domain["name"] + ".json"), {"summary": summary, "rows": rows})
                print(json.dumps({"mode": name, "dataset": domain["name"], "full": summary["full"],
                                  "fine_coverage": summary["fine"]["counts"], "diversity": summary["diversity"]}), flush=True)
    finally:
        ruler_proposals.transport_proposals = original
        handle.remove()
    write_json(output / "summary.json", {"config": plan, "results": final,
                                         "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                         "notes": ["固定权重改变选峰，不改变候选预算、半径、模型或输入",
                                                   "原始替换函数首次输出与生产函数逐张量完全一致断言通过",
                                                   "±1与±2在运行前共同登记，不逐域挑最好数值，仍需训练对照才能判断最终潜力",
                                                   "GT仅用于评测/oracle，选峰与新增刻度排斥均不使用GT"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))
