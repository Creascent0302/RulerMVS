"""核对补充压力实验的清洁输入与原完整VAL逐帧统计一致。"""

import json
from pathlib import Path


def main():
    folder = Path('runs/active/rulermvs_v6_20261101/full_training')
    result = json.loads((folder / 'paper_physics/results.json').read_text())
    plan = json.loads((folder / 'evaluation_recovery/plan.json').read_text())
    assert result['completed']
    checks = []
    forwards = 0
    for name, domain in result['datasets'].items():
        group = 'carla_new_main' if name.startswith('CARLA') else 'public_main'
        directory = Path(plan['groups'][group]['output']) / 'full_val' / name
        original = {}
        for part in directory.glob('part_*.json'):
            original.update({f['index']: f for f in json.loads(part.read_text())['frames']})
        max_relative_difference = 0.0
        for frame in domain['frames']:
            for condition in frame['conditions'].values():
                forwards += sum('depth_sums' in value for value in condition.values())
            for method in ('v5_best', 'v6_best', 'mvsanywhere_refined'):
                a = frame['conditions']['clean'][method]['depth_sums']['full']
                b = original[frame['index']]['methods'][method]['depth']['full']
                assert a['gt_pixels'] == b['gt_pixels'] and a['predicted_pixels'] == b['predicted_pixels']
                for key in ('abs_sum', 'sq_sum', 'abs_rel_sum'):
                    relative = abs(a[key] - b[key]) / max(abs(b[key]), 1.0)
                    max_relative_difference = max(max_relative_difference, relative)
        assert max_relative_difference < 1e-6
        checks.append({'domain': name, 'frames': len(domain['frames']),
                       'max_relative_clean_sum_difference': max_relative_difference})
    assert forwards == 6336
    audit = {'completed': True, 'actual_forwards': forwards, 'checks': checks,
             'firstpass_clean_not_matched_to_full_val': '当前没有该方法完整VAL，未虚构核对'}
    (folder / 'paper_physics/clean_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
