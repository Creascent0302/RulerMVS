"""只在完整队列成功结束后核验产物并更新状态；保留原启动与失败证据。"""

import json
from datetime import datetime
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from jar.pretraining.comparison import digest, write_json
from research.v2_restart_20261029.post_evaluate import verify_existing_speed

BASE = ROOT / 'runs/active/rulermvs_v6_20261101/full_training'
EVAL = BASE / 'evaluation_recovery'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    assert (EVAL / 'queue.exit_code.txt').read_text().strip() == '0'
    assert read(EVAL / 'completed.json')['completed']
    plan = read(EVAL / 'plan.json')
    weights = read(EVAL / 'weight_audit.json')
    assert all(digest(path) == value for path, value in weights.items())
    assert all(digest(ROOT / path) == value for path, value in plan['extension_source_sha256'].items())
    assert all(digest(plan['settings']['training_configs'][phase]) == value
               for phase, value in plan['training_config_sha256'].items())
    audit = {'completed': True, 'time_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
             'training': {}, 'groups': {}, 'prediction_calls': {'val': 0, 'test': 0},
             'weights_sha256_unchanged': True, 'extension_source_sha256_unchanged': True,
             'training_configs_sha256_unchanged': True, 'old_failed_queue_exit': (BASE / 'queue.exit_code.txt').read_text().strip(),
             'recovered_queue_exit': '0', 'free_gib': shutil.disk_usage(BASE.resolve()).free / 2**30}
    for phase in ('public', 'carla_new'):
        events = [json.loads(line) for line in (BASE / phase / 'events.jsonl').read_text().splitlines() if line.strip()]
        complete = [e for e in events if e.get('event') == 'complete']
        assert complete and (BASE / phase / 'exit_code.txt').read_text().strip() == '0'
        stages = [e for e in events if e.get('event') == 'stage_complete']
        audit['training'][phase] = {'complete_time_utc': complete[-1]['time_utc'],
                                     'best_global_step': stages[-1]['best_global_step'],
                                     'no_rollback': all(e['selection'] == 'last_parameters_no_rollback' for e in stages)}
    for group, settings in plan['groups'].items():
        audit['groups'][group] = {}
        for split in ('val', 'test'):
            folder = Path(settings['output']) / ('full_' + split)
            report = read(folder / 'summary.json')
            config = read(settings['val_config'] if split == 'val' else settings['config'])
            expected_counts = settings['val_frames'] if split == 'val' else settings['frames']
            assert report['completed'] and report['split'] == split
            expected_methods = set(config['methods'])
            assert set(report['datasets']) == set(expected_counts)
            call_count = 0
            for name, domain in report['datasets'].items():
                methods = domain['methods']
                assert set(methods) == expected_methods
                assert all(m['frames'] == expected_counts[name] for m in methods.values())
                for region in ('full', 'near', 'mid', 'far', 'gradient_band', 'smooth'):
                    assert len({m['depth'][region]['gt_pixels'] for m in methods.values()}) == 1
                indices = []
                for part in sorted((folder / name).glob('part_*.json')):
                    data = read(part)
                    assert data['completed'] and data['protocol'] == report['protocol']
                    indices.extend(f['index'] for f in data['frames'])
                assert sorted(indices) == list(range(expected_counts[name]))
                call_count += sum(m['frames'] for m in methods.values())
            audit['groups'][group][split] = {'completed': True, 'reference_samples': sum(expected_counts.values()),
                                             'methods': len(expected_methods), 'prediction_calls': call_count,
                                             'same_gt_denominators': True, 'summary_sha256': digest(folder / 'summary.json')}
            audit['prediction_calls'][split] += call_count
        if 'speed_config' in settings:
            assert verify_existing_speed(settings['speed_config'])
            speed = read(Path(settings['output']) / 'latency/latency.json')
            audit['groups'][group]['speed'] = {'completed': True, 'datasets': list(speed['datasets']),
                                               'summary_sha256': digest(Path(settings['output']) / 'latency/latency.json')}
    combined = read(EVAL / 'summary.json')
    assert combined['completed'] and combined['prediction_calls'] == audit['prediction_calls']['test']
    deployment = read(EVAL / 'deployment_speed/results.json')
    drift = read(EVAL / 'deployment_validation/results.json')
    assert deployment['completed'] and drift['completed'] and drift['test_accessed'] is False
    assert len(drift['datasets']) == 11 and sum(d['frames'] for d in drift['datasets'].values()) == 176
    audit['deployment'] = {'speed_completed': True, 'numerical_validation_completed': True,
                            'validation_frame_references': 176, 'test_accuracy_mode': 'eager',
                            'fast_is_not_claimed_bitwise_equal': True,
                            'speed_sha256': digest(EVAL / 'deployment_speed/results.json'),
                            'drift_sha256': digest(EVAL / 'deployment_validation/results.json')}
    write_json(BASE / 'paper_audit/completion.json', audit)

    # 原启动清单逐字备份后再添加完成观察；原失败queue与全部失败日志不覆盖。
    manifest_path = BASE / 'launch_manifest.json'
    snapshot_path = BASE / 'launch_manifest_before_completion.json'
    if not snapshot_path.exists():
        shutil.copy2(manifest_path, snapshot_path)
    manifest = read(manifest_path)
    manifest['status'] = 'training_and_recovered_evaluation_complete'
    manifest['completion_audit'] = str(BASE / 'paper_audit/completion.json')
    manifest['completed_at_beijing'] = audit['time_beijing']
    manifest['original_launch_snapshot'] = str(snapshot_path)
    manifest['original_evaluation_failed_then_repaired'] = True
    write_json(manifest_path, manifest)
    plan_path = ROOT / 'research/rulermvs_v6_20261101/configs/training_plan.json'
    training_plan = read(plan_path)
    training_plan['status'] = 'training_and_recovered_evaluation_complete'
    training_plan['completion_audit'] = str(BASE / 'paper_audit/completion.json')
    training_plan['completed_at_beijing'] = audit['time_beijing']
    training_plan['evaluation_plan'] = str(EVAL / 'plan.json')
    write_json(plan_path, training_plan)
    lines = ['# V6完整训练与评测完成审计', '',
             f"核验时间：{audit['time_beijing']}。公共和CARLA训练均正常完成；原自动评测因spawn注册失败，修复后独立队列全部exit0。原失败记录与启动快照保留。", '',
             f"完整VAL共{audit['prediction_calls']['val']}次参考预测、完整TEST共{audit['prediction_calls']['test']}次参考预测，含主表8方法及单列V3。不是同样数量的独立场景。", '',
             '逐组核验了帧数、方法集合、相同GT分母、逐帧索引完整性、协议hash、权重hash和训练/扩展源码hash。MAE有效预测口径仍须结合coverage；完成审计不代表所有方法覆盖率相同。', '',
             '公共/CARLA普通推理测速，以及V5/V6快速部署与176VAL数值漂移均已完成。快速路径不宣称逐像素严格等价；完整TEST精度采用eager。', '',
             '完整指标与联合速度CSV位于evaluation_recovery；此审计不按TEST改变VAL-best或旧模型默认。']
    (BASE / 'paper_audit/完成状态.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'completed': True, 'prediction_calls': audit['prediction_calls'],
                      'audit': str(BASE / 'paper_audit/completion.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
