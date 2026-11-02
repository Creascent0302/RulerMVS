"""原固定权重的默认零噪声入口诊断；保留先前带噪bridge结果。"""

import argparse
import copy
import gc
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.runtime import move, seed_all
from research.near_evidence_20261030.pilot import parameter_hash, path
from research.ruler_flow_20261030.diagnostics import aggregate, summarize_maps, truth_masks
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
    output = path(settings["output"])
    original_plan = output / "plan.json"
    plan = json.loads(original_plan.read_text())
    old_bridge = json.loads((output / "bridge_probe.json").read_text())
    destination = output / "bridge_zero_start.json"
    if destination.exists():
        raise FileExistsError(destination)
    if shutil.disk_usage(output).free / 2 ** 30 < 200:
        raise RuntimeError("数据盘剩余空间不足200GiB")
    device = torch.device(settings["device"])
    torch.cuda.set_device(device)
    result = {
        "original_plan_sha256": digest(original_plan), "original_bridge_sha256": digest(output / "bridge_probe.json"),
        "source_sha256": {name: digest(ROOT / name) for name in (
            "research/ruler_flow_20261030/bridge_zero_start.py", "research/ruler_flow_20261030/variants.py",
            "research/ruler_flow_20261030/flow.py", "research/ruler_flow_20261030/direction.py")},
        "sigma": 0., "time": 0., "sign_target_threshold_q": 1e-4,
        "domains": {}, "parameter_hashes": {},
        "scope": "原诊断相同4域DEV前2，原固定direct/flow权重；真实默认q=0、t=0、sigma=0，只调用一次_read。GT仅用于事后可达分支责任与误差。零速度表示不更新；符号准确率分母排除|GT速度|<=1e-4的数值近零方向。不是完整TEST或新权重测试。先前sigma=.05桥诊断保持原样。",
    }
    for group, group_plan in plan["groups"].items():
        original_settings = json.loads((output.parent / f"{group}_flow/settings.json").read_text())
        source = json.loads(path(original_settings["data_config"]).read_text())
        config = copy.deepcopy(source)
        config["datasets"] = [copy.deepcopy(domain) for domain in source.get("validation_datasets", source["datasets"])
                              if domain["name"] in group_plan["selection"]]
        config.pop("validation_datasets", None)
        for domain in config["datasets"]:
            domain["manifests"] = {"val": domain["manifests"]["val"]}
            expected = group_plan["selection"][domain["name"]]["manifest_sha256"]
            if digest(path(domain["manifests"]["val"])) != expected:
                raise RuntimeError("VAL清单改变")
        models = {}
        for method in ("direct", "flow"):
            info = group_plan["methods"][method + "1"]
            if digest(path(info["checkpoint"])) != info["checkpoint_sha256"]:
                raise RuntimeError("原诊断固定权重改变")
            models[method] = load_model(info["checkpoint"], device, 1)[0]
        before = {name: parameter_hash(dict(model.named_parameters())) for name, model in models.items()}
        if before != old_bridge["parameter_hashes"][group]["before"]:
            raise RuntimeError("实际参数与带噪桥诊断不同")
        dataset = MixedDataset(config, "val")
        for domain_index, domain in enumerate(dataset.domains):
            name = domain["name"]
            frames = []
            for index in group_plan["selection"][name]["stochastic_indices"]:
                batch = move(collate_mixed([dataset[(domain_index, index, -1)]]), device)
                contexts, reads = {}, {}
                scale = float(domain["depth_scale"])
                # 前向仅使用RGB/标定；GT责任在两模型取证完成后构造。
                for method, model in models.items():
                    model.decoder.set_range(domain["near"] / scale, domain["far"] / scale)
                    contexts[method] = model.decoder._coarse_context(model.encode(batch["inputs"], domain, reference=0))
                    reads[method] = model.decoder._read(contexts[method], torch.zeros_like(contexts[method]["anchor"]), 0.)
                context = contexts["flow"]
                for key in ("anchor", "anchor_log", "radius", "low_state", "high_state", "prior"):
                    torch.testing.assert_close(contexts["direct"][key], context[key], atol=0, rtol=0)
                truth, masks = truth_masks(batch, domain)
                bridge = bridge_targets(context["anchor_log"], context["radius"], context["low_state"], context["high_state"],
                                        truth[:, None].float() / scale, masks["full"][:, None], "flow", 0.,
                                        torch.Generator(device=device).manual_seed(20261030 + index))
                assigned = bridge["assigned"] & reads["direct"]["valid"] & reads["flow"]["valid"]
                target = bridge["velocity"]
                if bool(bridge["initial"].ne(0).any()):
                    raise RuntimeError("默认起点不为零")
                labels = {region: mask & assigned.any(1) for region, mask in masks.items()}
                directional = assigned & (target.abs() > 1e-4)
                sign_labels = {region: mask & directional.any(1) for region, mask in masks.items()}

                def reduce(value):
                    return torch.where(assigned, value, 0.).sum(1) / assigned.sum(1).clamp_min(1)

                errors = {"zero_velocity_MSE": reduce(target.square()), "target_velocity_MAE": reduce(target.abs())}
                signs = {"zero_velocity_sign_accuracy": torch.zeros_like(truth)}
                for method in models:
                    estimate = reads[method]["velocity"]
                    errors[method + "_velocity_MSE"] = reduce((estimate - target).square())
                    errors[method + "_velocity_MAE"] = reduce((estimate - target).abs())
                    signs[method + "_sign_accuracy"] = ((estimate * target > 0) & directional).sum(1) / directional.sum(1).clamp_min(1)
                frames.append({"index": index, "assigned_pixels": int(assigned.any(1).sum()),
                               "original_gt_pixels": int(masks["full"].sum()),
                               "velocity_errors": summarize_maps(errors, labels),
                               "sign_accuracy": summarize_maps(signs, sign_labels)})
            result["domains"][name] = {"frames": frames,
                "velocity_errors": aggregate([frame["velocity_errors"] for frame in frames]),
                "sign_accuracy": aggregate([frame["sign_accuracy"] for frame in frames])}
            print(json.dumps({"zero_start_done": name}), flush=True)
        after = {name: parameter_hash(dict(model.named_parameters())) for name, model in models.items()}
        if before != after:
            raise RuntimeError("诊断修改了模型参数")
        result["parameter_hashes"][group] = {"before": before, "after": after, "identical": True}
        del models, contexts, reads
        gc.collect()
        torch.cuda.empty_cache()
    result["completed"] = True
    write_json(destination, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(path(parser.parse_args().config).read_text()))
