"""全部论文实验真实完成后生成证据清单；缺项直接报错，不补造结果。"""

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'runs/active/rulermvs_v6_20261101/full_training'


def main():
    paths = {
        'main_evaluation': 'paper_audit/completion.json',
        'firstpass': 'paper_firstpass/completed.json',
        'firstpass_speed': 'paper_firstpass/speed/results.json',
        'fast_full_test': 'paper_fast_test/completed.json',
        'mechanism': 'paper_mechanism/results.json',
        'robustness': 'paper_robustness/results.json',
        'range': 'paper_range_probe/results.json',
        'split': 'paper_split_audit/results.json',
        'external': 'paper_eth3d/evaluation/full_test/summary.json',
        'physics': 'paper_physics/clean_audit.json',
        'ablation_queue': 'paper_handoff_ablation/queue_completed.json',
        'handoff': 'paper_handoff_ablation/summary.json',
        'geometry': 'paper_geometry_ablation/analysis/summary.json',
    }
    records = {}
    evidence = {}
    for name, relative in paths.items():
        path = OUTPUT / relative
        raw = path.read_bytes()
        record = json.loads(raw)
        complete = (record['complete_training'] and record['complete_full_val']
                    if name == 'geometry' else record['completed'])
        if not complete:
            raise RuntimeError(f'{name}尚未完成')
        records[name] = record
        evidence[name] = {'file': relative, 'sha256': hashlib.sha256(raw).hexdigest()}
    queue = records['ablation_queue']
    assert not queue['errors']
    assert len(queue['stages']['train']) == 38
    assert records['fast_full_test']['test_references'] == 12666
    assert records['physics']['actual_forwards'] == 6336
    assert len(records['handoff']['checks']) == 30
    assert records['handoff']['seeds'] == [20261102, 20261103, 20261104]
    assert not records['handoff']['test_accessed'] and not records['geometry']['test_accessed']
    frames = sum(domain['frames'] for domain in records['handoff']['datasets'].values())
    assert frames == 8368
    analysis = json.loads((OUTPUT / 'paper_analysis/retraining/audit.json').read_text())
    assert set(analysis['completed']) == {'handoff', 'geometry'} and not analysis['missing']
    free_gib = shutil.disk_usage(OUTPUT).free / 1024**3
    if free_gib < 200:
        raise RuntimeError('数据盘低于约定200GiB底线')
    report = {'completed': True, 'evidence': evidence, 'training_runs': 38,
              'handoff_seeds': 3, 'geometry_seeds': 1,
              'full_val_reference_samples': frames, 'full_val_model_predictions': frames * 19,
              'fast_test_references': 12666, 'external_predictions': 27 * 2 * 8,
              'physics_predictions': 6336, 'free_data_gib': free_gib,
              'storage': str(OUTPUT.resolve()),
              'limitations': ['训练消融为共同V5起点、冻结编码器的896/256更新适配，不是从零全预算。',
                              '不同训练种子、参考图及条件不冒充独立物理场景。',
                              '未进行小障碍实例召回、完整占据、车辆端到端或等预算强baseline预训练。',
                              '快速完整TEST没有重新计算表面，不借普通执行表面指标。']}
    (OUTPUT / 'paper_delivery.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'evidence'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
