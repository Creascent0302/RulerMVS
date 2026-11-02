"""全覆盖、多域有限重放采样；不读取图像、标签或验证集指标。"""

import math
import random

from torch.utils.data import Sampler


def _balanced_prefix_order(updates, domain_count, rng):
    """按最大份额欠额交错各域，保留域内随机批次顺序与完整曝光量。"""
    queues = [[] for _ in range(domain_count)]
    for domain, entries in updates:
        queues[domain].append(entries)
    totals = [len(queue) for queue in queues]
    used = [0] * domain_count
    tie_order = list(range(domain_count))
    rng.shuffle(tie_order)
    ordered = []
    for step in range(len(updates)):
        # 与 (step + 1) * p - used 等价；整数计算避免浮点误差打破平局。
        available = (domain for domain in tie_order if used[domain] < totals[domain])
        domain = max(available, key=lambda d: (step + 1) * totals[d] - len(updates) * used[d])
        ordered.append((domain, queues[domain][used[domain]]))
        used[domain] += 1
    return ordered


class RebalancedFullPassSampler(Sampler):
    """每轮先完整访问各域，再按倍率追加无放回重放。

    一个update只含同一域，跨rank和梯度累积共同组成global batch。
    最后不足一个global batch时显式补齐；补齐会重复现有样本及增强。
    domain_counts区分真实重放量和补齐量，不能把补齐称为新样本。
    默认shuffle保持旧顺序；balanced_prefix仅交错各域，避免短训前缀偏离整轮比例。
    """

    def __init__(self, lengths, epoch, rank, world, batch_size, accumulation,
                 seed, start_update=0, domain_multipliers=None, domain_schedule="shuffle"):
        multipliers = [1.] * len(lengths) if domain_multipliers is None else list(domain_multipliers)
        if not lengths or any(int(n) != n or n < 1 for n in lengths):
            raise ValueError("每个训练域必须包含正整数个样本")
        if len(multipliers) != len(lengths) or any(not math.isfinite(m) or m < 1 for m in multipliers):
            raise ValueError("每个域需要一个有限且不小于1的重放倍率")
        if world < 1 or not 0 <= rank < world or batch_size < 1 or accumulation < 1 or epoch < 0:
            raise ValueError("rank/world/batch_size/accumulation/epoch不合法")
        if domain_schedule not in ("shuffle", "balanced_prefix"):
            raise ValueError("domain_schedule必须是shuffle或balanced_prefix")
        self.rank, self.world = rank, world
        self.batch_size, self.accumulation = batch_size, accumulation
        self.start_update = start_update
        self.domain_schedule = domain_schedule
        self.updates, self.repeats, self.domain_counts = [], {}, {}
        rng = random.Random(seed + epoch)
        global_batch = world * batch_size * accumulation
        for domain, (length, multiplier) in enumerate(zip(lengths, multipliers)):
            requested = math.ceil(length * multiplier)
            cycles = math.ceil(requested / length)
            entries = []
            for cycle in range(cycles):
                indices = list(range(length))
                rng.shuffle(indices)
                # MixedDataset据此轮换参考相机并更换增强种子。
                visit_epoch = epoch * cycles + cycle
                remaining = requested - len(entries)
                entries.extend((index, visit_epoch) for index in indices[:remaining])
            padded = math.ceil(requested / global_batch) * global_batch
            padding = padded - requested
            entries += [entries[i % requested] for i in range(padding)]
            self.repeats[domain] = padding
            self.domain_counts[domain] = {
                "unique_samples": length,
                "multiplier": multiplier,
                "requested_exposures": requested,
                "replay_exposures": requested - length,
                "padding_exposures": padding,
                "total_exposures": padded,
                "updates": padded // global_batch,
                "cycles": cycles,
            }
            self.updates.extend((domain, entries[start:start + global_batch])
                                for start in range(0, padded, global_batch))
        rng.shuffle(self.updates)
        if domain_schedule == "balanced_prefix":
            self.updates = _balanced_prefix_order(self.updates, len(lengths), rng)
        if int(start_update) != start_update or not 0 <= start_update <= len(self.updates):
            raise ValueError("start_update必须指向本轮已有update或轮末")

    def __len__(self):
        return (len(self.updates) - self.start_update) * self.accumulation

    def __iter__(self):
        for domain, entries in self.updates[self.start_update:]:
            for micro in range(self.accumulation):
                start = (micro * self.world + self.rank) * self.batch_size
                yield [(domain, index, visit_epoch)
                       for index, visit_epoch in entries[start:start + self.batch_size]]
