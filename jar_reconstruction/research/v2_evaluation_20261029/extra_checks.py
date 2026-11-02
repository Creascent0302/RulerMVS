"""无标签输入一致性、CPU传输开销及四参考输出计时；在精度任务退出后单卡运行。"""
from pathlib import Path
import copy
import json
import sys
import time
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import Models, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from tools.benchmark_public_speed import prepare_model, summarize
from tools.compare_public_pretrained import configure
from tools.smoke_outdoor_datasets import fingerprint, offline_only


def main():
    config = json.loads((Path(__file__).parent / 'carla.json').read_text())
    config['methods'] = {k: v for k, v in config['methods'].items() if not k.endswith('_equal')}
    domain = config['datasets'][0]
    output = ROOT / 'runs/active/v2_evaluation_20261029'
    offline_only()
    device = configure(config, 0)
    data_cfg = dataset_config(domain, config)
    labeled = build_dataset(data_cfg, 'val', load_targets=True)
    unlabeled = build_dataset(data_cfg, 'val', load_targets=False)
    rows = [json.loads(line) for line in Path(domain['manifests']['val']).read_text().splitlines()]
    complete = [i for i, row in enumerate(rows) if row['valid_camera_count'] == 4]
    partial = [i for i, row in enumerate(rows) if row['valid_camera_count'] < 4]
    selected = [complete[0], partial[0]]
    models = Models(config, device)
    old_name = 'v2_original_identity_check'
    identity = copy.deepcopy(config)
    identity['methods'] = {old_name: dict(config['methods']['v2_public'],
        weights=str(ROOT / 'runs/model_releases/RulerMVS-V2/public_recommended/b411275c3203/pretrained.pt'))}
    old = Models(identity, device)
    checks = []
    with torch.inference_mode():
        for index in selected:
            a = collate_frames([labeled[index]])
            with patch('numpy.load', side_effect=AssertionError('无标签路径不得读取深度NPY')):
                b = collate_frames([unlabeled[index]])
            assert not b['targets']
            assert fingerprint(a['inputs']) == fingerprint(b['inputs'])
            x, y = move(a['inputs'], device), move(b['inputs'], device)
            for name in config['methods']:
                active = prepare_model(models, name, config['methods'][name], domain, x)
                p, _ = models.convert(models.forward(name, domain, x), x)
                q, _ = models.convert(models.forward(name, domain, y), y)
                torch.testing.assert_close(p, q, rtol=0, atol=0)
                caches = [m._frozen_position_cache for m in active.modules()
                          if hasattr(m, '_frozen_position_cache')]
                assert caches and all(c.key is not None for c in caches)
                for cache in caches:
                    cache.enabled = False
                uncached, _ = models.convert(models.forward(name, domain, x), x)
                torch.testing.assert_close(q, uncached, rtol=0, atol=0)
                for cache in caches:
                    cache.enabled = True
                    cache.clear()
            prepare_model(old, old_name, identity['methods'][old_name], domain, x)
            p, _ = models.convert(models.forward('v2_public', domain, x), x)
            q, _ = old.convert(old.forward(old_name, domain, x), x)
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            checks.append({'val_index': index, 'valid_views': int(x['view_mask'].sum()),
                           'inputs_identical': True, 'predictions_bitwise_identical_without_labels': True,
                           'position_caches_active_and_bitwise_equivalent': True,
                           'public_export_original_bitwise_identical': True})
    old.loaded.clear()
    del old
    indices = [complete[0], complete[len(complete)//2], complete[-1]]
    names = list(config['methods'])
    records = []
    with torch.inference_mode():
        for index in indices:
            with patch('numpy.load', side_effect=AssertionError('计时不能读GT')):
                cpu = collate_frames([unlabeled[index]])['inputs']
            inputs = move(cpu, device)
            for name in names:
                prepare_model(models, name, config['methods'][name], domain, inputs)
            local_domains = [dict(domain, reference_index=i) for i in range(4)]
            for _ in range(8):
                for name in names:
                    for d in local_domains:
                        p = models.forward(name, d, inputs)
                        models.convert(p, inputs)
            times = {name: {'four_reference_network_ms': [], 'cpu_ready_to_depth_cpu_ms': []} for name in names}
            for iteration in range(20):
                order = names[iteration % len(names):] + names[:iteration % len(names)]
                for name in order:
                    torch.cuda.synchronize(device)
                    begin = time.perf_counter()
                    outputs = [models.convert(models.forward(name, d, inputs), inputs) for d in local_domains]
                    torch.cuda.synchronize(device)
                    times[name]['four_reference_network_ms'].append((time.perf_counter()-begin)*1000)
                    del outputs
                    # baseline不消费额外匹配图，不为它拷贝自研专有的大图。
                    host = cpu if config['methods'][name]['kind'] == 'own' else {
                        k: v for k, v in cpu.items() if not k.startswith('matching_')}
                    torch.cuda.synchronize(device)
                    begin = time.perf_counter()
                    work = move(host, device)
                    depth, valid = models.convert(models.forward(name, domain, work), work)
                    transferred = depth.cpu(), valid.cpu()
                    torch.cuda.synchronize(device)
                    times[name]['cpu_ready_to_depth_cpu_ms'].append((time.perf_counter()-begin)*1000)
                    del transferred, depth, valid, work
            records.append({'val_index': index, 'input_sha256': fingerprint(cpu), 'samples': times})
    summary = {name: {kind: summarize([v for frame in records for v in frame['samples'][name][kind]])
                     for kind in next(iter(records[0]['samples'].values()))} for name in names}
    for row in summary.values():
        row['four_reference_network_ms']['four_depth_update_hz'] = row['four_reference_network_ms'].pop('single_reference_hz')
    write_json(output/'extra_checks.json', {'completed': True, 'label_independence': checks,
        'latency': summary, 'frames': records, 'gpu': torch.cuda.get_device_name(device),
        'scope': '四个参考顺序运行现有入口并保留四深度图；没有跨参考特征缓存/点云融合，不是完整场景重建。CPU往返项仅单上参考，起点为解码好的CPU张量，不包含文件IO和图像预处理。',
        'protocol': 'VAL四目完整帧首中末，8轮预热+20次交替计时，共60次/方法；GPU0独占，FP32/TF32off，无GT。'})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
