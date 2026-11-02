"""缓存七域少量 TRAIN 与固定开发 VAL；只供冻结编码器的等预算解码器短训。"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.pretraining.model import PretrainingModel
from jar.runtime import move
from tools.refine_public_candidates import save_record
from tools.screen_public_architecture import balanced_sample, selection


def encoder_digest(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if name.startswith(("foundation.", "matching.")):
            value.update(name.encode())
            value.update(str((tensor.shape, tensor.dtype)).encode())
            value.update(tensor.detach().contiguous().numpy().tobytes())
    return value.hexdigest()


@torch.inference_mode()
def run(plan):
    root = Path(plan["cache"])
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("缓存目录非空，不覆盖已有实验")
    torch.set_num_threads(2)
    torch.manual_seed(plan["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(plan["device"])
    torch.cuda.set_device(device)
    saved = torch.load(plan["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    config = copy.deepcopy(saved["config"])
    config["augmentation"] = False
    model = PretrainingModel(config, initialize=False).to(device).eval().requires_grad_(False)
    model.load_state_dict(saved["model"], strict=True)
    install_position_cache(model.foundation.model)
    choices = selection(plan, config)
    index = {
        "checkpoint_sha256": digest(plan["checkpoint"]),
        "encoder_sha256": encoder_digest(saved["model"]),
        "domains": config["datasets"], "plan": plan, "dtype": "float32",
        "test_accessed": False, "augmentation": False,
        "reference_index_in_cache": 0,
        "splits": {"train": {}, "development": {}, "confirmation": {}},
        "manifests": {}, "selection": choices,
    }
    for split in ("train", "val"):
        dataset = MixedDataset(config, split)
        group = "train" if split == "train" else "development"
        for domain_index, domain in enumerate(dataset.domains):
            name = domain["name"]
            manifest = Path(domain["manifests"][split])
            index["manifests"][name + "/" + split] = digest(manifest)
            if split == "train":
                rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
                sequences = {}
                for number, row in enumerate(rows):
                    sequences.setdefault(str(row.get("sequence", "single_sequence")), []).append(number)
                selected = balanced_sample(sequences, plan["train_samples"], plan["seed"] + domain_index)
                index["selection"][name]["train"] = selected
            else:
                selected = choices[name]["development"]
            paths = []
            for number in selected:
                batch = move(collate_mixed([dataset[(domain_index, number, -1)]]), device)
                work = model.encode(batch["inputs"], domain, reference=0)
                record = {
                    "work": move(work, torch.device("cpu")),
                    "inputs": {key: batch["inputs"][key].cpu() for key in ("view_mask", "image_mask")},
                    "targets": {key: batch["targets"][key][:, :1].cpu() for key in ("depth", "depth_valid")},
                }
                path = root / group / name / f"{number:06d}.pt"
                save_record(record, path)
                paths.append(str(path))
            index["splits"][group][name] = paths
            print(json.dumps({"split": group, "dataset": name, "frames": len(paths)}, ensure_ascii=False), flush=True)
    write_json(root / "index.json", index)
    write_json(root / "audit.json", {
        "checkpoint_sha256": index["checkpoint_sha256"],
        "encoder_sha256": index["encoder_sha256"],
        "sources": {name: digest(ROOT / name) for name in (
            "tools/cache_architecture_screen.py", "tools/screen_public_architecture.py",
            "jar/pretraining/encoders.py", "jar/pretraining/model.py", "jar/pretraining/data.py")},
        "confirmation_cached": False, "test_accessed": False,
        "notes": "仅冻结编码器短训适用；修改编码器或解冻训练必须从RGB运行，不得复用旧特征。",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
