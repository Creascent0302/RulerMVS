"""复核已有MatrixCity测试：逐组改善、误差方向及输入相机间距，不重新推理。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read_json(path):
    return json.loads(path.read_text())


def describe(values):
    cuts = statistics.quantiles(values, n=10, method="inclusive")
    return dict(count=len(values), minimum=min(values), median=statistics.median(values),
                mean=statistics.mean(values), p10=cuts[0], p90=cuts[8], maximum=max(values))


def camera_distances(path):
    distances = []
    for line in path.read_text().splitlines():
        sample = json.loads(line)
        centers = []
        for camera in sample['cameras']:
            transform = camera['T_cam_from_rig']
            centers.append([-sum(transform[j][i] * transform[j][3] for j in range(3))
                            for i in range(3)])
        ref = sample['reference_slot']
        distances.extend(math.dist(center, centers[ref]) for i, center in enumerate(centers) if i != ref)
    return describe(distances)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison', type=Path)
    args = parser.parse_args()
    root = args.comparison
    summary_path = root / 'accuracy/full_test/summary.json'
    summary = read_json(summary_path)['datasets']['MatrixCity_street']
    frames = sorted([frame for path in (root / 'accuracy/full_test/MatrixCity_street').glob('part_*.json')
                     for frame in read_json(path)['frames']], key=lambda frame: frame['index'])
    assert [frame['index'] for frame in frames] == list(range(summary['frames']))
    result = dict(source_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                  frames=len(frames), scope='既有TEST最终深度，事后描述统计，无新推理或调参', methods={})
    for name in ('v6', 'v7'):
        metric = summary['methods'][name]
        frame_mae = [f['methods'][name]['abs_sum'] / f['methods'][name]['gt_pixels'] for f in frames]
        result['methods'][name] = dict(
            metrics=metric, frame_mae=describe(frame_mae),
            over_contribution=(metric['MAE'] + metric['signed_bias']) / 2,
            under_contribution=(metric['MAE'] - metric['signed_bias']) / 2)
    a, b = result['methods']['v6'], result['methods']['v7']
    result['v7_mae_winning_frames'] = sum(f['methods']['v7']['abs_sum'] < f['methods']['v6']['abs_sum']
                                       for f in frames)
    gain = a['metrics']['MAE'] - b['metrics']['MAE']
    result['over_error_share_of_net_mae_improvement'] = (a['over_contribution'] - b['over_contribution']) / gain
    result['relative_improvement'] = {key: 1 - b['metrics'][key] / a['metrics'][key]
                                      for key in ('MAE', 'RMSE', 'AbsRel')}
    gains = sorted([f['methods']['v6']['abs_sum'] - f['methods']['v7']['abs_sum'] for f in frames], reverse=True)
    result['top_frames_share_of_net_abs_error_reduction'] = {str(n): sum(gains[:n]) / sum(gains)
                                                            for n in (1, 5, 10, 20)}
    domain = next(d for d in read_json(root / 'config.json')['datasets'] if d['name'] == 'MatrixCity_street')
    result['reference_source_distance_m'] = {split: camera_distances(Path(domain['manifests'][split]))
                                           for split in ('train', 'val', 'test')}
    result['notes'] = [
        'over/under贡献均除以全部有效像素数，不是各自像素子集的条件MAE或像素比例。',
        '逐帧来源于同一城市，非160个独立场景；没有多训练seed或阶段因果消融。',
        '候选重融合和错误减少的机制解释为待检验假设。']
    output = root / 'analysis/matrixcity_diagnosis.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
