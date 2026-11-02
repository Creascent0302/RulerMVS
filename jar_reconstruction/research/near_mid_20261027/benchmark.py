"""新表面解码的完整RGB预算检查；只检查速度和冷启动数值，不宣称精度。"""

import copy
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.pretraining.inference import prepare_inference
from jar.pretraining.model import PretrainingModel, load_pretraining_state
from jar.models.position_cache import install_position_cache
from jar.runtime import move
from research.ruler_joint_20261027.benchmark import measure_frame, distribution


def run(plan):
    torch.set_num_threads(2)
    torch.manual_seed(20261027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    checkpoint = ROOT / 'runs/active/public_refinement_20261025/recommended/pretrained.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    surface_checkpoint = ROOT / plan['surface_checkpoint'] if plan.get('surface_checkpoint') else None
    surface_saved = torch.load(surface_checkpoint, map_location='cpu', weights_only=False,
                               mmap=True) if surface_checkpoint else None
    config = copy.deepcopy(saved['config'])
    config['augmentation'] = False
    models = {}
    names = ('original', 'surface', 'original_compiled', 'surface_compiled') if plan.get('compile', False) else ('original', 'surface')
    for name in names:
        settings = copy.deepcopy(config)
        if name.startswith('surface'):
            if surface_saved:
                settings = copy.deepcopy(surface_saved['config'])
            settings['model']['decoder_kind'] = 'surface_transport'
        model = PretrainingModel(settings, initialize=False)
        weights = surface_saved if name.startswith('surface') and surface_saved else saved
        load_pretraining_state(model, weights['model'])
        if name.endswith('_compiled'):
            model.load_state_dict(models[name.removesuffix('_compiled')].state_dict(), strict=True)
        model.to(device).eval().requires_grad_(False)
        install_position_cache(model.foundation.model)
        prepare_inference(model, spatial_layout=True)
        if name == 'original_compiled':
            model.decoder._upsample = torch.compile(model.decoder._upsample, fullgraph=True, dynamic=False)
        elif name == 'surface_compiled':
            model.decoder.surface_readout.compile(fullgraph=True, dynamic=False)
        models[name] = model
    dataset = MixedDataset(config, 'val')
    index = json.loads((ROOT / 'runs/active/architecture_screen_20261026/features/index.json').read_text())
    output = ROOT / plan['output']
    output.mkdir(parents=True, exist_ok=False)
    report = {'scope': '完整RGB、单参考；FP32/TF32off；计时不代替精度评测',
              'source_checkpoint': str(checkpoint), 'gpu': torch.cuda.get_device_name(device),
              'source_sha256': digest(checkpoint),
              'surface_checkpoint': str(surface_checkpoint) if surface_checkpoint else None,
              'surface_sha256': digest(surface_checkpoint) if surface_checkpoint else None,
              'surface_is_trained': surface_saved is not None,
              'params': {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
              'datasets': {}, 'compiled_parity': [], 'frames': []}
    with torch.inference_mode():
        for domain_index, domain in enumerate(dataset.domains):
            if domain['name'] not in ('TartanAir_forest', 'VKITTI2'):
                continue
            paths = index['splits']['development'][domain['name']]
            frames = []
            for position in (0, len(paths) // 2, len(paths) - 1):
                row_index = int(Path(paths[position]).stem)
                batch = move(collate_mixed([dataset[(domain_index, row_index, -1)]]), device)
                inputs = batch['inputs']

                def predict(name, domain, inputs):
                    return models[name](inputs, domain, reference=0)

                if plan.get('compile', False):
                    for kind in ('original', 'surface'):
                        a = predict(kind, domain, inputs)['depth']
                        b = predict(kind + '_compiled', domain, inputs)['depth']
                        error = (a - b).abs()
                        parity = {'dataset': domain['name'], 'index': row_index, 'method': kind,
                                  'max_difference_normalized_depth': float(error.max()),
                                  'mean_difference_normalized_depth': float(error.mean()),
                                  'unequal_fraction': float((a != b).float().mean()),
                                  'finite': bool(torch.isfinite(a).all() & torch.isfinite(b).all())}
                        report['compiled_parity'].append(parity)
                        print('compiled_parity', parity, flush=True)
                result = measure_frame(predict, list(models), inputs, domain, device, 8, 30)
                frames.append(result)
                report['frames'].append({'dataset': domain['name'], 'index': row_index, 'methods': result})
                print(domain['name'], row_index,
                      {name: row['summary']['wall_network_ms']['median_ms'] for name, row in result.items()},
                      flush=True)
            report['datasets'][domain['name']] = {
                name: distribution([value for frame in frames
                                    for value in frame[name]['samples']['wall_network_ms']])
                for name in models
            }
            write_json(output / 'latency.json', report)
    report['completed'] = True
    write_json(output / 'latency.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
