"""同输入三权重责任质量对照；每域预定DEV前4帧，禁止训练和TEST。"""

import argparse
import copy
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import add_statistics, digest, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.pretraining.model import PretrainingModel
from jar.runtime import move, seed_all
from research.near_evidence_20261030.pilot import path
from research.near_evidence_20261030.visibility import capture_attention, visibility_targets
from research.near_mid_evaluation_20261028.metrics import depth_metrics, depth_sums, ground_truth, regions
from tools.smoke_outdoor_datasets import offline_only


@torch.no_grad()
def run(settings):
    offline_only()
    output = path(settings["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("不覆盖已有机制探针")
    source = json.loads(path(settings["data_config"]).read_text())
    source["datasets"] = source.get("validation_datasets", source["datasets"])
    source.pop("validation_datasets", None)
    previous = json.loads(path(settings["old_plan"]).read_text())["validation"]
    selection = {}
    for domain in source["datasets"]:
        domain["manifests"] = {"val": domain["manifests"]["val"]}
        assert digest(path(domain["manifests"]["val"])) == previous[domain["name"]]["manifest_sha256"]
        selection[domain["name"]] = previous[domain["name"]]["development"][:4]
    plan = {"selection": selection, "methods": settings["methods"], "test_accessed": False,
            "weights_sha256": {name: digest(path(filename)) for name, filename in settings["methods"].items()},
            "source_sha256": {name: digest(ROOT / name) for name in (
                "research/near_evidence_20261030/visibility_probe.py",
                "research/near_evidence_20261030/visibility.py", "jar/pretraining/model.py")},
            "scope": "既定DEV前4帧三固定权重；用于解释机制，不选择训练超参。遮挡质量另取三个模型均具资格的同一负源交集。"}
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    write_json(output / "settings.json", settings)
    seed_all(20261030)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(settings["device"])
    torch.cuda.set_device(device)
    dataset = MixedDataset(source, "val")
    records, depth = {}, {}
    for method, filename in settings["methods"].items():
        saved = torch.load(path(filename), map_location="cpu", weights_only=False, mmap=True)
        model = PretrainingModel(saved["config"], initialize=False)
        model.load_state_dict(saved["model"], strict=True)
        model.requires_grad_(False).eval().to(device)
        install_position_cache(model.foundation.model)
        depth[method] = {}
        for domain_index, domain in enumerate(dataset.domains):
            name = domain["name"]
            sums = {}
            for index in selection[name]:
                batch = move(collate_mixed([dataset[(domain_index, index, -1)]]), device)
                with capture_attention(model) as captured:
                    prediction = model(batch["inputs"], domain, reference=0)
                truth, valid = ground_truth(batch, dict(domain, reference_index=0))
                masks = regions(truth, valid, domain)
                for region in ("full", "near"):
                    add_statistics(sums.setdefault(region, {}), depth_sums(
                        prediction["depth"] * float(prediction["unit_scale"]), prediction["valid"], truth, masks[region]))
                sample = records.setdefault((name, index), {})
                sample[method] = {}
                if not domain.get("correspondence", True):
                    continue
                geometric = dict(prediction, visibility_eval_range=domain.get("eval_range"))
                assert len(captured.records) == 2
                for (stage, stride), observation in zip((("coarse", 8), ("fine", 4)), captured.records):
                    target = visibility_targets(geometric, batch["inputs"], batch["targets"], stage, observation, stride)
                    nearest = target["nearest"][:, None]
                    count = observation["weights"].shape[1]
                    weights = observation["weights"].float().gather(2, nearest.expand(-1, count, -1, -1, -1))[:, :, 0]
                    null = observation["null_mass"].float().gather(2, nearest)[:, 0, 0]
                    sample[method][stage] = {"weights": weights.cpu(), "null": null.cpu(),
                        "labels": target["labels"].cpu(), "valid": target["reference_valid"].cpu(),
                        "covered": target["covered"].cpu()}
            depth[method][name] = {region: depth_metrics(value) for region, value in sums.items()}
        del model, saved
        gc.collect()
        torch.cuda.empty_cache()
        print("固定权重前向完成", method, flush=True)

    totals = {}
    for (domain, index), sample in records.items():
        for stage in ("coarse", "fine"):
            if not all(stage in item for item in sample.values()):
                continue
            observations = [item[stage] for item in sample.values()]
            common_valid = torch.stack([item["valid"] for item in observations]).all(0)
            common_occluded = torch.stack([item["labels"] == 0 for item in observations]).all(0)
            common_negative_ray = common_occluded.any(1) & common_valid
            for method, item in sample.items():
                item = item[stage]
                entry = totals.setdefault(domain, {}).setdefault(stage, {}).setdefault(method, {})
                occluded = item["labels"] == 0
                common_bad_mass = (item["weights"] * common_occluded).sum(1)
                add_statistics(entry, {
                    "reference_rays": int(common_valid.sum()),
                    "common_occluded_pairs": int(common_occluded.sum()),
                    "common_negative_rays": int(common_negative_ray.sum()),
                    "own_occluded_pairs": int(occluded.sum()),
                    "own_covered_rays": int(item["covered"].sum()),
                    "null_mass_sum": float(item["null"][common_valid].sum()),
                    "common_bad_mass_sum": float(common_bad_mass[common_negative_ray].sum()),
                })
    for domains in totals.values():
        for stages in domains.values():
            for values in stages.values():
                values["null_mean"] = values["null_mass_sum"] / max(values["reference_rays"], 1)
                values["bad_mass_negative_rays"] = values["common_bad_mass_sum"] / max(values["common_negative_rays"], 1)
                values["bad_mass_all_rays"] = values["common_bad_mass_sum"] / max(values["reference_rays"], 1)
    assert {name: digest(path(filename)) for name, filename in settings["methods"].items()} == plan["weights_sha256"]
    write_json(output / "metrics.json", {"completed": True, "depth": depth, "responsibility": totals,
        "test_accessed": False, "scope": plan["scope"],
        "caveat": "各模型在自身最接近GT的采样候选读取责任；共有负例要求三者都满足投影接近和区间覆盖。拒绝资格变化本身不计作质量改善。Synth不允许源深度对应监督，跳过责任质量。"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(path(parser.parse_args().config).read_text()))
