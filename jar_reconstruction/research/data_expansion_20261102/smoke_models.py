"""新增真实TRAIN各一组：V6损失/反向与Hero推理；不更新权重、不评价精度。"""
import copy
import json
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from jar.data import collate_frames
from jar.pretraining.comparison import digest, equal_pixel_inputs, write_json
from jar.pretraining.data import MixedDataset
from jar.runtime import move
from research.rulermvs_v6_20261101.evaluation import Models
from research.rulermvs_v6_20261101.objective import objective
from tools.compare_public_pretrained import configure
from tools.smoke_outdoor_datasets import offline_only
from direct_http import ROOT


def main():
    offline_only()
    config = json.loads((PROJECT / 'runs/active/rulermvs_v6_20261101/full_training/'
                         'paper_eth3d/evaluation.json').read_text())
    config['methods'] = {k: config['methods'][k] for k in ('v6_best', 'mvsanywhere_refined')}
    training_config = ROOT / 'train_v6_expanded.json'
    train = json.loads(training_config.read_text())
    selected = [('Hypersim_subset', 'Hypersim_expanded', 597),
                ('TartanAir_forest', 'TartanAir_expanded', 3056),
                ('DynamicReplica_valid', 'DynamicReplica_train', 0)]
    domains = []
    for name, folder, _ in selected:
        domain = copy.deepcopy(next(d for d in train['datasets'] if d['name'] == name))
        domain['root'] = str(ROOT.parent / folder)
        domain['manifests'] = {'train': domain['root'] + '/manifests/train.jsonl'}
        domains.append(domain)
    train['datasets'] = domains
    dataset = MixedDataset(train)
    device = configure(config, 7)
    models = Models(config, device)
    records = []
    for i, (_, folder, index) in enumerate(selected):
        batch = move(collate_frames([dataset[i, index, -1]]), device)
        inputs = equal_pixel_inputs(batch['inputs'])
        domain = dict(domains[i], reference_index=0)
        model = models.get('v6_best', domain)
        model.decoder.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        prediction = model(inputs, domain, reference=0)
        loss = objective(prediction, inputs, batch['targets'], domain, train['loss'])['total']
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [p.grad for p in model.decoder.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        norm = sum(float(g.square().sum()) for g in gradients) ** .5
        assert norm > 0
        with torch.inference_mode():
            baseline = models.forward('mvsanywhere_refined', domain, inputs)
            depth, valid = models.convert(baseline, inputs)
        assert torch.isfinite(depth).all() and valid.any()
        records.append(dict(dataset=folder, index=index, input_shape=list(inputs['rgb'].shape),
            loss=float(loss.detach()), decoder_gradient_norm=norm,
            baseline_shape=list(depth.shape), baseline_finite=True, optimizer_steps=0))
        print(records[-1], flush=True)
        model.zero_grad(set_to_none=True)
    write_json(ROOT / 'gpu_interface_smoke.json', dict(completed=True, records=records,
        training_config_sha256=digest(training_config), precision='FP32', weights_updated=False,
        note='使用已发布扩容TRAIN范围/尺度；仅管线冒烟，不把单样本损失当精度结果'))


if __name__ == '__main__':
    main()
