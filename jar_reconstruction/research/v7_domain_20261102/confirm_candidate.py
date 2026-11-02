"""固定一个候选的外测确认；不搜索权重、不训练，不按TEST换模型。"""

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.data import build_dataset, collate_frames
from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import digest, equal_pixel_inputs, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move, seed_all
from research.near_mid_evaluation_20261028.metrics import depth_sums, ground_truth, regions
from research.paper_ablation_20261102.handoff_fullval import finish
from research.paper_ablation_20261102.handoff_sharedval import add_record, metric_record
from research.rulermvs_v6_20261101.evaluation import Models
from research.v7_domain_20261102.model_round2 import build_model
from tools.smoke_outdoor_datasets import offline_only


@torch.inference_mode()
def main(settings):
    offline_only()
    seed_all(20261102)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(settings['device'])
    torch.cuda.set_device(device)
    output = Path(settings['output'])
    if output.exists():
        raise FileExistsError(output)
    if not output.resolve().is_relative_to('/data'):
        raise ValueError('结果必须位于数据盘')
    output.mkdir(parents=True)
    config = json.loads(Path(settings['data_config']).read_text())
    config['methods'] = {'mvsanywhere_refined': config['methods']['mvsanywhere_refined']}
    config['tf32'] = False
    baseline = Models(config, device)
    models = {}
    audit = {'completed': False, 'split': 'test', 'settings': settings,
             'source_sha256': digest(__file__), 'datasets': {}, 'weights_sha256': {}}
    for item in settings['models']:
        saved = torch.load(item['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        model, _ = build_model(saved, item.get('variant', 'control'))
        model.eval().requires_grad_(False).to(device)
        install_position_cache(model.foundation.model)
        models[item['name']] = model
        audit['weights_sha256'][item['name']] = digest(item['checkpoint'])
        del saved
    audit['weights_sha256']['mvsanywhere_refined'] = digest(config['methods']['mvsanywhere_refined']['weights'])
    write_json(output / 'plan.json', audit)
    for domain in config['datasets']:
        dataset = build_dataset(dataset_config(domain, config), 'test', load_targets=True)
        totals = {name: {} for name in [*models, 'mvsanywhere_refined']}
        manifest = [json.loads(line) for line in Path(domain['manifests']['test']).read_text().splitlines()]
        with (output / (domain['name'] + '.jsonl')).open('w') as stream:
            for index in range(len(dataset)):
                batch = collate_frames([dataset[index]])
                inputs = move(equal_pixel_inputs(batch['inputs']), device)
                truth, mask = ground_truth(batch, domain)
                masks = regions(truth, mask, domain)
                row = {'index': index, 'frame_id': manifest[index]['frame_id'], 'models': {}}
                for name, model in models.items():
                    prediction = model(inputs, domain, reference=domain['reference_index'])
                    raw, _ = metric_record(prediction, truth, masks, float(prediction['unit_scale']))
                    add_record(totals[name], raw)
                    row['models'][name] = raw
                    del prediction
                prediction = baseline.forward('mvsanywhere_refined', domain, inputs)
                depth, valid = baseline.convert(prediction, inputs)
                depth, valid = depth.cpu(), valid.cpu()
                raw = {'depth': {name: depth_sums(depth, valid, truth, selected)
                                 for name, selected in masks.items()}}
                add_record(totals['mvsanywhere_refined'], raw)
                row['models']['mvsanywhere_refined'] = raw
                stream.write(json.dumps(row) + '\n')
                if index % 25 == 0:
                    stream.flush()
                    print(json.dumps({'dataset': domain['name'], 'frames': index + 1,
                                      'total': len(dataset)}), flush=True)
                del prediction
        audit['datasets'][domain['name']] = {
            'frames': len(dataset), 'units': domain['geometry_units'],
            'manifest_sha256': digest(domain['manifests']['test']),
            'models': {name: {'raw': values, 'metrics': finish(values)}
                       for name, values in totals.items()},
        }
        write_json(output / 'results.json', audit)
    audit['completed'] = True
    write_json(output / 'results.json', audit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    main(json.loads(Path(parser.parse_args().config).read_text()))
