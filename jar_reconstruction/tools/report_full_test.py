"""将完成的精度与独立计时结果汇成中文报告和CSV，不选择或训练权重。"""

import argparse
import csv
import json
from pathlib import Path


def number(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config['output'])
    protocol = config['protocol']
    lines = [config['title'], '', *config.get('findings', []), '',
             protocol[0], '', *protocol[1:], '']
    records = []
    merged = {}
    for group in config['groups']:
        accuracy = json.loads(Path(group['accuracy']).read_text())
        timing = json.loads(Path(group['timing']).read_text())
        if not accuracy['completed'] or not timing['completed']:
            raise ValueError('评测或计时尚未完成，不生成完整报告')
        merged[group['name']] = {'accuracy': accuracy, 'timing': timing}
        settings = accuracy['config']['methods']
        lines.extend([f"## {group['name']}", ''])
        for name, domain in accuracy['datasets'].items():
            units = '米' if domain['units'] == 'meters' else '原场景单位（不是米）'
            lines.extend([f'### {name}：{units}', '',
                          '| 方法 | 帧数 | MAE↓ | RMSE↓ | AbsRel↓ | δ1↑ | δ2↑ | δ3↑ | 预测覆盖率↑ |',
                          '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
            speed = timing['datasets'].get(name, {})
            for method, row in domain['methods'].items():
                metric = row['metrics']
                label = settings[method].get('label', method)
                values = [number(metric[key]) for key in ('MAE', 'RMSE', 'AbsRel')]
                values += [number(100 * metric[key], 3) + '%' for key in ('delta_1', 'delta_2', 'delta_3')]
                values += [number(100 * metric['prediction_coverage'], 7) + '%']
                lines.append('| ' + ' | '.join([label, str(row['frames']), *values]) + ' |')
                record = {'group': group['name'], 'dataset': name, 'method': method,
                          'label': label, 'frames': row['frames'], 'units': domain['units'],
                          **{key: metric[key] for key in ('MAE', 'RMSE', 'AbsRel', 'delta_1',
                            'delta_2', 'delta_3', 'prediction_coverage', 'gt_pixels', 'predicted_pixels')}}
                if method in speed.get('summary', {}):
                    value = speed['summary'][method]
                    record.update(network_ms=value['wall_network_ms']['median_ms'],
                                  network_p90_ms=value['wall_network_ms']['p90_ms'],
                                  with_resampling_ms=value['wall_with_resampling_ms']['median_ms'],
                                  hz=value['wall_with_resampling_ms']['single_reference_hz'])
                record.update(speed.get('isolated_resources', {}).get(method, {}))
                records.append(record)
            lines.extend(['', '同卡交替计时与单模型独占显存（固定VAL样本；不是全TEST延迟分布）：', '',
                          '| 方法 | 网络中位ms↓ | 网络P90 ms↓ | 含格点转换ms↓ | 单参考Hz↑ | 峰值allocated GiB↓ | 参数百万 |',
                          '|---|---:|---:|---:|---:|---:|---:|'])
            for method, row in speed.get('summary', {}).items():
                memory = speed.get('isolated_resources', {}).get(method, {})
                label = settings[method].get('label', method)
                values = [number(row['wall_network_ms']['median_ms'], 3),
                          number(row['wall_network_ms']['p90_ms'], 3),
                          number(row['wall_with_resampling_ms']['median_ms'], 3),
                          number(row['wall_with_resampling_ms']['single_reference_hz'], 2),
                          number(memory.get('peak_allocated_gib'), 3),
                          number(memory.get('parameters', 0) / 1e6, 3)]
                lines.append('| ' + ' | '.join([label, *values]) + ' |')
            lines.append('')
            if any(row.get('distance_bands') for row in domain['methods'].values()):
                lines.extend(['按真实深度分段的MAE；范围只用于离线评价，不进入推理：', '',
                              '| 方法 | 2–10m | 10–24m | 24–80m |', '|---|---:|---:|---:|'])
                for method, row in domain['methods'].items():
                    values = [number(row['distance_bands'][band]['MAE']) for band in ('2-10', '10-24', '24-80')]
                    lines.append('| ' + ' | '.join([settings[method].get('label', method), *values]) + ' |')
                lines.append('')
        lines.extend(['适用性限制：', ''])
        for method, reason in accuracy['not_applicable'].items():
            lines.append(f'- {method}：{reason}。')
        for row in accuracy['unsupported']:
            lines.append(f"- {row['dataset']} / {row['method']}：{row['reason']}")
        lines.append('')
    keys = list(dict.fromkeys(key for row in records for key in row))
    with (output / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    (output / 'comparison_summary.json').write_text(json.dumps(merged, ensure_ascii=False, indent=2) + '\n')
    Path(config['report']).write_text('\n'.join(lines) + '\n')
    print(config['report'])


if __name__ == '__main__':
    main()
