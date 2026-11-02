"""七域联合调度：每轮全覆盖，域内批处理，不混淆尺度和真实相机数。"""

import math
import random

import torch
from torch.utils.data import Dataset, Sampler

from jar.data import build_dataset, collate_frames


def dataset_config(domain, config):
    result = {"data": {"root": domain["root"], "protocol": "calibrated_multiview",
                     "geometry_units": domain["geometry_units"], "num_views": domain["num_views"],
                     "image_size": config["image_size"], "matching_image_size": config["matching_image_size"],
                     "depth_resize_mode": "nearest-exact",
                     **{key + "_manifest": value for key, value in domain["manifests"].items()}}}
    if "camera" in domain:
        # CARLA按真实相机ID补缺失槽，公共多视图保留原来的真实N=2/4协议。
        result["data"].pop("protocol")
        result["camera"] = domain["camera"]
    return result


class MixedDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config, self.split = config, split
        self.domains = config["datasets"]
        self.datasets = [build_dataset(dataset_config(d, config), split) for d in self.domains]

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, key):
        domain_index, index, epoch = key
        sample = self.datasets[domain_index][index]
        inputs, targets = sample["inputs"], sample["targets"]
        n = inputs["rgb"].shape[0]
        reference = self.domains[domain_index]["reference_index"]
        if self.split == "train" and epoch >= 0:
            labelled = targets["depth_valid"].flatten(1).any(1) & inputs["view_mask"]
            allowed = labelled.nonzero().flatten().tolist()
            if allowed:
                reference = allowed[(index + epoch) % len(allowed)]
        order = [reference] + [i for i in range(n) if i != reference]
        for collection in (inputs, targets):
            for name, value in collection.items():
                if torch.is_tensor(value) and value.ndim and value.shape[0] == n:
                    collection[name] = value[order]
        if "camera_ids" in inputs:
            inputs["camera_ids"] = [inputs["camera_ids"][i] for i in order]
        if self.split == "train" and self.config.get("augmentation", True) and epoch >= 0:
            generator = torch.Generator().manual_seed(self.config["seed"] + epoch * 100003 + index * 13 + domain_index)
            gains = .85 + .3 * torch.rand(n, 1, 1, 1, generator=generator)
            for name in ("rgb", "matching_rgb"):
                inputs[name] = (inputs[name] * gains).clamp(0, 1)
            if n > 2 and torch.rand((), generator=generator) < .15:
                missing = int(torch.randint(1, n, (), generator=generator))
                inputs["view_mask"][missing] = False
                inputs["image_mask"][missing] = False
                inputs["matching_image_mask"][missing] = False
                inputs["rgb"][missing] = 0
                inputs["matching_rgb"][missing] = 0
                if "camera_valid" in inputs:
                    inputs["camera_valid"][missing] = False
                    inputs["image_sizes"][missing] = 0
        sample["domain_index"] = domain_index
        return sample


def collate_mixed(samples):
    domain = samples[0]["domain_index"]
    if any(s["domain_index"] != domain for s in samples):
        raise ValueError("同一microbatch必须属于同一域")
    result = collate_frames(samples)
    result["domain_index"] = domain
    return result


class FullPassSampler(Sampler):
    """先按32组组成域内更新，再打乱更新；八rank和累积步共享域。"""

    def __init__(self, lengths, epoch, rank, world, batch_size, accumulation, seed, start_update=0):
        self.batch_size = batch_size
        self.world, self.rank, self.accumulation = world, rank, accumulation
        self.epoch, self.start_update = epoch, start_update
        rng = random.Random(seed + epoch)
        self.updates = []
        global_batch = world * batch_size * accumulation
        self.repeats = {}
        for domain, length in enumerate(lengths):
            indices = list(range(length))
            rng.shuffle(indices)
            padded = math.ceil(length / global_batch) * global_batch
            self.repeats[domain] = padded - length
            indices += [indices[i % length] for i in range(padded - length)]
            self.updates.extend((domain, indices[i:i + global_batch]) for i in range(0, padded, global_batch))
        rng.shuffle(self.updates)

    def __len__(self):
        return (len(self.updates) - self.start_update) * self.accumulation

    def __iter__(self):
        for domain, indices in self.updates[self.start_update:]:
            for micro in range(self.accumulation):
                start = (micro * self.world + self.rank) * self.batch_size
                yield [(domain, index, self.epoch) for index in indices[start:start + self.batch_size]]
