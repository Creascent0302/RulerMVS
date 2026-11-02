"""冻结编码器的候选诊断与等预算训练；只允许 TRAIN/VAL，不读取 TEST。"""

import argparse
import copy
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.pretraining.losses import depth_objective
from jar.pretraining.model import CalibratedDecoder, PretrainingModel, load_pretraining_state
from jar.runtime import move
from tools.probe_public_stages import diagnose, sum_records, summarize


def join(values):
    """缓存保留 batch=1，拼接同域样本；尺寸、参考号等静态字段不堆叠。"""
    first = values[0]
    if torch.is_tensor(first):
        return torch.cat(values)
    if isinstance(first, dict):
        return {key: join([value[key] for value in values]) for key in first}
    if isinstance(first, list):
        return [join([value[index] for value in values]) for index in range(len(first))]
    if any(value != first for value in values):
        raise ValueError("一个批次不能混合不同的域或静态编码约定")
    return first


class FeatureDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return torch.load(self.paths[index], map_location="cpu", weights_only=False)


def save_record(record, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < 204 * 2 ** 30:
        raise RuntimeError("数据盘须保留200GiB及临时保存余量")
    temporary = path.with_suffix(".tmp")
    torch.save(record, temporary)
    temporary.replace(path)


def prepare(plan, device):
    root = Path(plan["cache"])
    if (root / "index.json").exists():
        raise FileExistsError("已有完整缓存，不能覆盖")
    saved = torch.load(plan["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    config = copy.deepcopy(saved["config"])
    config["augmentation"] = False
    model = PretrainingModel(config, initialize=False).to(device).eval().requires_grad_(False)
    model.load_state_dict(saved["model"], strict=True)
    index = {"checkpoint_sha256": digest(plan["checkpoint"]), "domains": config["datasets"],
             "plan": plan, "splits": {"train": {}, "development": {}, "confirmation": {}},
             "manifests": {}, "dtype": "float32", "test_accessed": False}
    for split in ("train", "val"):
        dataset = MixedDataset(config, split)
        for domain_index, domain in enumerate(dataset.domains):
            name = domain["name"]
            if split == "train" and name not in plan["train_domains"]:
                continue
            count = len(dataset.datasets[domain_index])
            index["manifests"][name + "/" + split] = digest(domain["manifests"][split])
            if split == "train":
                choices = list(range(count))
                random.Random(plan["seed"] + domain_index).shuffle(choices)
                groups = {"train": choices[:plan["train_samples"]]}
            else:
                development = torch.linspace(0, count - 1, min(count, plan["development_samples"])).long().tolist()
                remaining = [i for i in range(count) if i not in set(development)]
                random.Random(plan["seed"] + 100 + domain_index).shuffle(remaining)
                groups = {"confirmation": sorted(remaining[:plan["confirmation_samples"]])}
                if name in plan["train_domains"]:
                    groups["development"] = development
            for group, choices in groups.items():
                paths = []
                for number in choices:
                    path = root / group / name / f"{number:06d}.pt"
                    batch = move(collate_mixed([dataset[(domain_index, number, -1)]]), device)
                    with torch.no_grad():
                        work = model.encode(batch["inputs"], domain, reference=0)
                    # 标签与编码严格分开；编码器从未看到depth。
                    record = {"work": move(work, torch.device("cpu")),
                              "inputs": {key: batch["inputs"][key].cpu() for key in ("view_mask", "image_mask")},
                              "targets": {key: batch["targets"][key][:, :1].cpu() for key in ("depth", "depth_valid")}}
                    save_record(record, path)
                    paths.append(str(path))
                index["splits"][group][name] = paths
                print(json.dumps({"event": "cached", "split": group, "dataset": name,
                                  "frames": len(paths)}, ensure_ascii=False), flush=True)
    write_json(root / "index.json", index)


@torch.no_grad()
def evaluate(decoder, index, split, device, output):
    decoder.eval()
    result = {}
    captured = []
    hook = decoder.upsample_mask.register_forward_hook(lambda module, args, value: captured.append(value))
    try:
        for domain in index["domains"]:
            paths = index["splits"][split].get(domain["name"], [])
            if not paths:
                continue
            decoder.set_range(domain["near"] / domain["depth_scale"], domain["far"] / domain["depth_scale"])
            records = []
            for path in paths:
                row = move(torch.load(path, map_location="cpu", weights_only=False), device)
                prediction = decoder(row["work"])
                prediction["unit_scale"] = domain["depth_scale"]
                records.append(diagnose(prediction, captured.pop(), row, domain))
            result[domain["name"]] = summarize(sum_records(records))
    finally:
        hook.remove()
    result["macro_abs_rel"] = sum(row["full"]["abs_rel"] for row in result.values()) / len(result)
    write_json(output, result)
    return result


def run(plan, device):
    index = json.loads((Path(plan["cache"]) / "index.json").read_text())
    if index["checkpoint_sha256"] != digest(plan["checkpoint"]):
        raise ValueError("缓存必须属于同一个冻结编码器checkpoint")
    output = Path(plan["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("实验目录非空，禁止覆盖")
    output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(plan["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    settings = {**saved["config"]["model"], **plan.get("model_overrides", {})}
    decoder = CalibratedDecoder(settings).to(device)
    initial = {key.removeprefix("decoder."): value for key, value in saved["model"].items() if key.startswith("decoder.")}
    load_pretraining_state(decoder, initial)
    if plan.get("decoder_checkpoint"):
        decoder.load_state_dict(torch.load(plan["decoder_checkpoint"], map_location="cpu", weights_only=False)["model"], strict=True)
    write_json(output / "plan.json", plan)
    write_json(output / "audit.json", {"checkpoint_sha256": index["checkpoint_sha256"],
               "cache_index_sha256": digest(Path(plan["cache"]) / "index.json"),
               "sources": {name: digest(ROOT / name) for name in (
                   "jar/pretraining/model.py", "jar/pretraining/losses.py",
                   "jar/pretraining/candidate_supervision.py", "jar/models/ruler_proposals.py",
                   "tools/refine_public_candidates.py")}, "test_accessed": False})
    if plan["action"] == "evaluate":
        evaluate(decoder, index, plan.get("split", "confirmation"), device, output / "evaluation.json")
        return
    initial_result = evaluate(decoder, index, "development", device, output / "step_0000.json")
    if plan.get("trainable_modules"):
        decoder.requires_grad_(False)
        for name in plan["trainable_modules"]:
            getattr(decoder, name).requires_grad_(True)
    optimizer = torch.optim.AdamW([p for p in decoder.parameters() if p.requires_grad],
                                  lr=plan["lr"], weight_decay=1e-4)
    domains = {domain["name"]: domain for domain in index["domains"]}
    rng = random.Random(plan["seed"])
    names = list(plan["train_domains"])
    order = (names * math.ceil(plan["steps"] / len(names)))[:plan["steps"]]
    rng.shuffle(order)
    paths_by_domain = {}
    for name in names:
        paths = index["splits"]["train"][name]
        required = order.count(name) * plan["batch_size"]
        selected = []
        while len(selected) < required:
            cycle = list(paths)
            rng.shuffle(cycle)
            selected.extend(cycle)
        paths_by_domain[name] = iter(DataLoader(FeatureDataset(selected[:required]),
            batch_size=plan["batch_size"], collate_fn=join, num_workers=plan.get("workers", 1), pin_memory=True))
    best = initial_result["macro_abs_rel"]
    best_step = 0
    save_record({"model": decoder.state_dict(), "settings": settings, "plan": plan, "step": 0}, output / "best.pt")
    started = time.monotonic()
    with (output / "train.jsonl").open("w") as stream:
        for step, name in enumerate(order, 1):
            domain = domains[name]
            row = move(next(paths_by_domain[name]), device)
            decoder.set_range(domain["near"] / domain["depth_scale"], domain["far"] / domain["depth_scale"])
            decoder.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = decoder(row["work"])
            prediction["unit_scale"] = domain["depth_scale"]
            loss = depth_objective(prediction, row["inputs"], row["targets"], domain,
                                   plan.get("loss", {}), correspondence_weight=0.)
            loss["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            record = {"step": step, "dataset": name, "loss": float(loss["total"].detach()),
                      "grad_norm": float(norm), "elapsed_s": time.monotonic() - started}
            stream.write(json.dumps(record) + "\n")
            if step % 20 == 0:
                stream.flush()
                print(json.dumps(record), flush=True)
            if step % plan["evaluate_every"] == 0 or step == plan["steps"]:
                result = evaluate(decoder, index, "development", device, output / f"step_{step:04d}.json")
                if result["macro_abs_rel"] < best:
                    best, best_step = result["macro_abs_rel"], step
                    save_record({"model": decoder.state_dict(), "settings": settings, "plan": plan, "step": step}, output / "best.pt")
                save_record({"model": decoder.state_dict(), "settings": settings, "plan": plan, "step": step}, output / "last.pt")
                print(json.dumps({"event": "validation", "step": step, "macro_abs_rel": result["macro_abs_rel"],
                                  "best_step": best_step}), flush=True)
    write_json(output / "complete.json", {"steps": plan["steps"], "best_step": best_step,
               "best_macro_abs_rel": best, "test_accessed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    plan = json.loads(Path(parser.parse_args().config).read_text())
    if plan["action"] not in ("cache", "train", "evaluate"):
        raise ValueError("action须为cache/train/evaluate")
    torch.set_num_threads(2)
    torch.manual_seed(plan["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(plan["device"])
    torch.cuda.set_device(device)
    if plan["action"] == "cache":
        prepare(plan, device)
    else:
        run(plan, device)


if __name__ == "__main__":
    main()
