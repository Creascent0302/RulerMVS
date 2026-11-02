"""原生模型训练的可选验证：只用VAL选权重，预测调用不传标签。"""

from functools import partial
import json
from pathlib import Path
import shutil

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from jar.runtime import capture_rng, move, restore_rng
from .data import FourCameraDataset, collate
from .metrics import depth_statistics, finish_depth
from .postprocess import depth_on_image_grid


def check_storage(output, minimum_gib):
    if minimum_gib and shutil.disk_usage(Path(output).resolve()).free < minimum_gib * 1024 ** 3:
        raise OSError(f"checkpoint数据盘可用空间低于{minimum_gib}GiB，停止训练而不删除旧权重")


@torch.no_grad()
def validate_depth(adapter, config, runtime, synthetic=False):
    """无重复分片汇总所有有效GT；保持训练随机流、BN和训练模式。"""
    dataset = FourCameraDataset(config, "val", synthetic, load_targets=True)
    loader = DataLoader(
        dataset, batch_size=1,
        sampler=range(runtime.rank, len(dataset), runtime.world_size),
        num_workers=config["training"].get("num_workers", 0),
        collate_fn=partial(collate, expected_views=dataset.num_views),
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    rng, was_training = capture_rng(runtime), adapter.training
    # 训练保留各rank本地BN；验证必须全部使用实际保存的rank0模型状态。
    local_buffers = {name: value.clone() for name, value in adapter.named_buffers()} if runtime.distributed else {}
    torch.manual_seed(config["seed"] + runtime.rank)
    adapter.eval()
    keys = ("gt_pixels", "predicted_pixels", "abs_sum", "sq_sum", "abs_rel_sum",
            "delta_1_hits", "delta_2_hits", "delta_3_hits")
    totals = torch.zeros(len(keys) + 1, dtype=torch.float64, device=runtime.device)
    try:
        if runtime.distributed:
            for buffer in adapter.buffers():
                dist.broadcast(buffer, src=0)
        for batch in loader:
            batch = move(batch, runtime.device)
            inputs, targets = batch["inputs"], batch["targets"]
            prediction = adapter(inputs)
            converted = depth_on_image_grid(prediction, inputs)
            refs = prediction["reference_indices"]
            mask = (targets["depth_valid"][:, refs] & inputs["image_mask"][:, refs]
                    & inputs["view_mask"][:, refs, None, None])
            stats = depth_statistics(
                converted["depth"].double(), targets["depth"][:, refs].double(),
                mask, converted["depth_valid"], *config["evaluation"]["depth_range"],
            )
            totals[:-1] += totals.new_tensor([stats[key] for key in keys])
            totals[-1] += len(inputs["rgb"])
        if runtime.distributed:
            dist.all_reduce(totals)
        values = totals.cpu().tolist()
        sums = dict(zip(keys, values[:-1]))
        metrics = finish_depth(sums, config["data"].get("geometry_units", "meters"))
        near, far = config["evaluation"]["depth_range"]
        missing = sums["gt_pixels"] - sums["predicted_pixels"]
        score = (sums["abs_sum"] + missing * (far - near)) / sums["gt_pixels"] if sums["gt_pixels"] else None
        return {
            "frames": int(values[-1]), "depth": metrics,
            "selection_score": score,
            "selection_rule": "全VAL逐像素MAE，缺失/非正/非有限预测按预定far-near惩罚；不删GT。此选择分数不冒充实际MAE。",
        }
    finally:
        for name, buffer in adapter.named_buffers():
            if name in local_buffers:
                buffer.copy_(local_buffers[name])
        adapter.train(was_training)
        restore_rng(rng, runtime)


def append_record(output, record):
    with (Path(output) / "validation.jsonl").open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
