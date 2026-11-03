"""实际B8 BF16训练容量与通用初始化配对检查，不保存试训权重。"""

import copy
import gc
import json
from pathlib import Path
import statistics
import sys
import time

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

import torch

from jar.data import collate_frames
from jar.pretraining.comparison import equal_pixel_inputs, write_json
from jar.pretraining.data import MixedDataset
from jar.runtime import move
from research.rulermvs_v6_20261101.model import V6Model
from research.rulermvs_v6_20261101.objective import objective
from research.overnight_v6_v7_mvs_20261103.v7.model import V7Model
from tools.smoke_outdoor_datasets import offline_only


def main():
    offline_only()
    torch.set_num_threads(2)
    torch.cuda.set_device(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda', 4)
    config = json.loads((PROJECT.parent / 'data/public_expansion_20261102/train_v6_expanded.json').read_text())
    config['initial_model_checkpoint'] = ''
    config['augmentation'] = False
    config['stages'] = [dict(number=1, trainable_prefixes=['decoder.', 'bridge.', 'matching.'],
                             foundation_blocks=8, bridge_detach_backbone=True)]
    config['architecture'] = 'rulermvs_v6'
    torch.manual_seed(20261103)
    v6 = V6Model(config, initialize=True)
    config7 = copy.deepcopy(config)
    config7.update(architecture='rulermvs_v7_refusion', refusion=dict(hidden=32, heads=4, bands=8))
    torch.manual_seed(20261103)
    v7 = V7Model(config7, initialize=True)
    common = v6.state_dict()
    changed = [key for key, value in common.items() if not torch.equal(value, v7.state_dict()[key])]
    if changed:
        raise RuntimeError(f'共同初始化不一致：{changed[:10]}')
    report = dict(completed=False, generic_initialization_common_tensors=len(common),
                  common_tensors_identical=True, task_checkpoint_loaded=False,
                  precision='bf16', batch_size=8, views=4, foundation_trainable_blocks=8,
                  matching_trainable=True, bridge_detach_backbone=True, records={})
    output = Path('/data/liuzonghao_data/dataset/public_expansion_20261102/v7_refusion_probe_20261103/capacity.json')
    train = MixedDataset(config)
    batch = move(collate_frames([train[0, i, -1] for i in range(8)]), device)
    inputs = equal_pixel_inputs(batch['inputs'])
    domain = train.domains[0]
    for name, model in [('v6', v6), ('v7', v7)]:
        model.set_stage(1)
        model.to(device).train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        torch.cuda.reset_peak_memory_stats()
        durations, losses = [], []
        try:
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = model(inputs, domain)
                    loss = objective(prediction, inputs, batch['targets'], domain, config['loss'])['total']
                if not bool(loss.isfinite()):
                    raise RuntimeError('nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - start)
                losses.append(float(loss.detach()))
                del prediction, loss
            report['records'][name] = dict(ok=True, micro_seconds=durations,
                steady_micro_seconds=statistics.mean(durations[1:]), losses=losses,
                peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30)
        except torch.cuda.OutOfMemoryError as error:
            report['records'][name] = dict(ok=False, error=str(error),
                peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30)
        write_json(output, report)
        print(name, report['records'][name], flush=True)
        model.zero_grad(set_to_none=True)
        model.cpu()
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
    report['completed'] = True
    write_json(output, report)


if __name__ == '__main__':
    main()
