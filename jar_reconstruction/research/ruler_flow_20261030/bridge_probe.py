"""带GT的训练桥诊断：比较真实起点与桥内速度拟合，不作为推理精度。"""

import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.runtime import move, seed_all
from research.near_evidence_20261030.pilot import parameter_hash, path
from research.ruler_flow_20261030.diagnostics import aggregate, prepare, summarize_maps, truth_masks
from research.ruler_flow_20261030.flow import bridge_targets
from research.ruler_flow_20261030.pilot import load_model
from tools.smoke_outdoor_datasets import offline_only


@torch.inference_mode()
def run(settings):
    offline_only()
    seed_all(20261030)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    configs, plan, output = prepare(settings, complete=True)
    destination = output / "bridge_probe.json"
    if destination.exists():
        raise FileExistsError("桥诊断结果已存在，拒绝覆盖")
    device = torch.device(settings["device"])
    torch.cuda.set_device(device)
    times = (0., .25, .5, .75, 1.)
    result = {
        "plan_sha256": digest(output / "plan.json"), "source_sha256": digest(Path(__file__)),
        "times": times, "sigma": .05, "seed": 20261030, "domains": {},
        "scope": "原DEV前2；GT仅构造训练桥，不能称推理精度或独立验证。direct只训练t=0，其余t为结构诊断。所有时间和两种方法采用相同initial/target/assigned交集。q_t/t是使用桥内状态的解析探针，t0无定义，非可部署算法。",
    }
    for group, config in configs.items():
        group_plan = plan["groups"][group]
        models = {name: load_model(group_plan["methods"][label]["checkpoint"], device, 1)[0]
                  for name, label in (("direct", "direct1"), ("flow", "flow1"))}
        before = {name: parameter_hash(dict(model.named_parameters())) for name, model in models.items()}
        dataset = MixedDataset(config, "val")
        for domain_index, domain in enumerate(dataset.domains):
            name = domain["name"]
            frames = []
            for index in group_plan["selection"][name]["stochastic_indices"]:
                batch = move(collate_mixed([dataset[(domain_index, index, -1)]]), device)
                contexts = {}
                scale = float(domain["depth_scale"])
                for method, model in models.items():
                    model.decoder.set_range(domain["near"] / scale, domain["far"] / scale)
                    encoded = model.encode(batch["inputs"], domain, reference=0)
                    contexts[method] = model.decoder._coarse_context(encoded)
                context = contexts["flow"]
                for key in ("anchor", "anchor_log", "radius", "low_state", "high_state", "prior"):
                    torch.testing.assert_close(contexts["direct"][key], context[key], atol=0, rtol=0)
                truth, masks = truth_masks(batch, domain)
                bridge = bridge_targets(
                    context["anchor_log"], context["radius"], context["low_state"], context["high_state"],
                    truth[:, None].float() / scale, masks["full"][:, None], "flow", .05,
                    torch.Generator(device=device).manual_seed(20261030 + index),
                )
                reads, states = {}, {}
                assigned = bridge["assigned"].clone()
                for time in times:
                    state = (1 - time) * bridge["initial"] + time * bridge["endpoint"]
                    states[time] = state
                    for method, model in models.items():
                        record = model.decoder._read(contexts[method], state, time)
                        reads[(method, time)] = record
                        assigned &= record["valid"]
                selected_count = assigned.sum(1).clamp_min(1)
                assigned_masks = {region: mask & assigned.any(1) for region, mask in masks.items()}
                target = bridge["velocity"]

                def reduce(value):
                    return torch.where(assigned, value, 0.).sum(1) / selected_count

                per_time = {}
                for time in times:
                    maps = {"target_velocity_MSE": reduce(target.square()),
                            "target_velocity_MAE": reduce(target.abs()),
                            "initial_noise_MSE": reduce(bridge["initial"].square())}
                    for method in models:
                        velocity = reads[(method, time)]["velocity"]
                        maps[method + "_velocity_MSE"] = reduce((velocity - target).square())
                        maps[method + "_velocity_MAE"] = reduce((velocity - target).abs())
                        maps[method + "_prediction_abs"] = reduce(velocity.abs())
                    maps["gn_suggestion_velocity_MSE"] = reduce((reads[("flow", time)]["direction"][:, 2] - target).square())
                    if time:
                        analytic = states[time] / max(time, .05)
                        maps["analytic_q_over_t_velocity_MSE"] = reduce((analytic - target).square())
                        # 不作为模型结果：严格验证解析探针误差恰来自 initial/t。
                        torch.testing.assert_close((analytic - target)[assigned],
                                                   (bridge["initial"] / time)[assigned], atol=2e-6, rtol=2e-5)
                    per_time[str(time)] = summarize_maps(maps, assigned_masks)
                frames.append({"index": index, "assigned_pixels": int(assigned.any(1).sum()),
                               "original_gt_pixels": int(masks["full"].sum()),
                               "time_metrics": per_time, "analytic_t0": None,
                               "same_initial_and_target_for_both_methods": True})
            result["domains"][name] = {"units": domain["geometry_units"], "frames": frames,
                "aggregate": {str(time): aggregate([frame["time_metrics"][str(time)] for frame in frames]) for time in times}}
            print(json.dumps({"bridge_done": name}), flush=True)
        after = {name: parameter_hash(dict(model.named_parameters())) for name, model in models.items()}
        if before != after:
            raise RuntimeError("桥诊断改变了模型参数")
        result.setdefault("parameter_hashes", {})[group] = {"before": before, "after": after, "identical": True}
        del models, contexts, reads, context, encoded
        gc.collect()
        torch.cuda.empty_cache()
    result["completed"] = True
    write_json(destination, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(path(parser.parse_args().config).read_text()))
