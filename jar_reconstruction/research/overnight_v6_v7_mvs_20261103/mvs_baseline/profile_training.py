"""B8稳态训练计算计时，用于选择正式并行卡数；不是精度实验。"""

import json
from pathlib import Path
import statistics
import sys
import time

import torch

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from jar.pretraining.data import MixedDataset, collate_mixed
from jar.runtime import move
from research.overnight_v6_v7_mvs_20261103.mvs_baseline.model import MVSAnywhereTrainingModel
from tools.smoke_outdoor_datasets import offline_only


def main():
    offline_only()
    torch.set_num_threads(2)
    torch.cuda.set_device(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads(Path("/data/liuzonghao_data/dataset/public_expansion_20261102/train_v6_expanded.json").read_text())
    directory = Path("/data/liuzonghao_data/dataset/public_expansion_20261102/mvs_initialization")
    config["augmentation"] = False
    config["initialization"].update(
        dinov2=str(directory / "dinov2_vitb14_pretrain.pth"),
        matching_antialiased=str(directory / "resnet18_lpf4_finetune-8cc58f59.pth"),
    )
    dataset = MixedDataset(config)
    batch = move(collate_mixed([dataset[(0, 3056 + i, 0)] for i in range(8)]), torch.device("cuda", 6))
    model = MVSAnywhereTrainingModel(config).cuda(6).train()
    optimizer, scheduler = model.configure_optimizer()
    durations, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for step in range(10):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch["inputs"], config["datasets"][0], targets=batch["targets"])
        output["loss"].backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        durations.append(time.monotonic() - start)
        losses.append(float(output["loss"]))
        print(step, durations[-1], losses[-1], float(gradient), flush=True)
    report = dict(batch_per_gpu=8, views=4, dtype="bf16 autocast", warmup=3, steps=10,
                  median_seconds=statistics.median(durations[3:]),
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                  durations=durations, losses=losses,
                  excludes="数据加载、DDP通信和VAL，不能直接当完整训练预计时长")
    (directory / "profile_b8.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
