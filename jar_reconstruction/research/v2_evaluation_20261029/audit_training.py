"""核对训练导出、选择与输入清单；不执行或选择TEST权重。"""
from pathlib import Path
import json
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from jar.pretraining.comparison import digest, write_json


def main():
    torch.set_num_threads(2)
    folder = ROOT / 'runs/active/v2_training_20261028'
    original_path = ROOT / 'runs/model_releases/RulerMVS-V2/public_recommended/b411275c3203/pretrained.pt'
    original = torch.load(original_path, map_location='cpu', weights_only=False, mmap=True)
    report = {'original_sha256': digest(original_path), 'training': {}}
    for name in ('public', 'carla_new'):
        directory = folder/name
        selected = torch.load(directory/'incumbent.pt', map_location='cpu', weights_only=False, mmap=True)
        exported = torch.load(directory/'pretrained.pt', map_location='cpu', weights_only=False, mmap=True)
        last = torch.load(directory/'last.pt', map_location='cpu', weights_only=False, mmap=True)
        events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
        changed = [k for k, v in original['model'].items() if not torch.equal(v, exported['model'][k])]
        assert all(torch.equal(v, exported['model'][k]) for k, v in selected['model'].items())
        validations = [{k: e[k] for k in ('stage', 'epoch', 'macro_relative_change', 'guard_failures')}
                       for e in events if e['event'] == 'validation' and e['full']]
        report['training'][name] = {'exit_code': int((directory/'exit_code.txt').read_text()),
            'export_sha256': digest(directory/'pretrained.pt'), 'promotion_eligible': exported['promotion_eligible'],
            'selected_stage': selected.get('stage'), 'selected_epoch': selected.get('epoch'),
            'selected_step': selected['global_step'], 'selection_score': selected['selection_score'],
            'total_updates': last['state']['global_step'], 'total_exposures': last['state']['exposures'],
            'epochs': last['state']['global_epoch'], 'changed_tensors_from_original': len(changed),
            'export_equals_incumbent': True, 'full_validations': validations,
            'completed_utc': events[-1]['time_utc']}
    report['limitations'] = [
        '公共全部state_dict逐位等于原推荐，导出文件SHA不同来自配置/元数据，不代表新学习权重。',
        '公共last.pt也在阶段交界回退，失败的阶段末模型未独立保存，无法事后测其TEST；仅保留日志VAL。',
        '新CARLA按VAL选择stage2 epoch1，不按TEST换权重；基线无本轮新CARLA微调，训练预算不相等。',
        '历史公共TEST已用于研究诊断，不能称严格盲测或双方同等未见域。']
    cfg = json.loads((ROOT/'research/v2_evaluation_20261029/carla.json').read_text())
    domain = cfg['datasets'][0]
    rows = [json.loads(l) for l in Path(domain['manifests']['test']).read_text().splitlines()]
    report['carla_test'] = {'frames': len(rows), 'by_views': {str(n):sum(r['valid_camera_count']==n for r in rows) for n in (2,3,4)},
                           'sequences': sorted({r['sequence'] for r in rows})}
    assert report['carla_test']['sequences'] == ['captured_data3']
    write_json(ROOT/'runs/active/v2_evaluation_20261029/training_audit.json', report)
    print(json.dumps({k:{a:b for a,b in v.items() if a!='full_validations'} for k,v in report['training'].items()},ensure_ascii=False))


if __name__ == '__main__':
    main()
