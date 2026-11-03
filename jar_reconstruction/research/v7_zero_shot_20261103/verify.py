"""完成后只读核验零样本评测，并把证据写到本次数据盘目录。"""

import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

PROJECT = Path('/home/liuzonghao/3Dscene/jar_reconstruction')
DATA = Path('/data/liuzonghao_data/dataset/V7_zero_shot_20261103')


def load(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    protocol = load(DATA / 'results/execution_protocol.json')
    plan = load(DATA / 'results/plan.json')
    progress = load(DATA / 'results/progress.json')
    delivery = load(DATA / 'results/delivery.json')
    assert progress['completed'] and delivery['completed']
    assert progress['predictions'] == delivery['predictions'] == 2076
    for name in ['preparation.exit_code.txt', 'tanks_prepare.exit_code.txt', 'results/queue.exit_code.txt']:
        assert (DATA / name).read_text().strip() == '0', name
    assert load(DATA / 'preparation_status.json')['completed']
    assert load(DATA / 'results/tanks_plan.json')['completed']
    for path, digest in protocol['source_sha256'].items():
        assert sha(PROJECT / path) == digest, path
    for name, item in protocol['config']['methods'].items():
        assert sha(Path(item['weights'])) == plan['weights_sha256'][name], name
    manifest_records = []
    expected_frames = 0
    prediction_count = 0
    frame_checks = 0
    counts = {}
    coverages = []
    for scene in protocol['selected_scenes']:
        manifests = {}
        for views in [2, 4, 8]:
            name = f'{scene}_{views}views'
            path = DATA / 'manifests' / f'{name}.jsonl'
            manifests[views] = [json.loads(line) for line in path.read_text().splitlines()]
            record = next(r for r in progress['records'] if r['domain'] == name)
            assert sha(path) == record['manifest_sha256']
            part = load(DATA / 'results/accuracy/full_test' / name / 'part_0000.json')
            assert part['completed'] and part['protocol'] == record['protocol']
            assert part['indices'] == list(range(len(manifests[views])))
            assert len(part['frames']) == len(manifests[views])
            manifest_records.append(dict(name=name, groups=len(part['frames']), sha256=sha(path)))
            expected_frames += len(part['frames'])
            for frame in part['frames']:
                frame_checks += 1
                assert set(frame['methods']) == {'v6', 'v7', 'mvs_retrained', 'mvs_hero'}
                for region in ['full', 'near', 'mid', 'far']:
                    gt_counts = {m[region]['gt_pixels'] for m in frame['methods'].values()}
                    assert len(gt_counts) == 1
                for regions in frame['methods'].values():
                    assert regions['full']['gt_pixels'] > 0
                    coverages.append(regions['full']['predicted_pixels'] / regions['full']['gt_pixels'])
                    for region, stats in regions.items():
                        assert 0 <= stats['predicted_pixels'] <= stats['gt_pixels'], (name, region, stats)
                        assert all(math.isfinite(v) for v in stats.values())
                    assert sum(regions[r]['gt_pixels'] for r in ['near', 'mid', 'far']) == regions['full']['gt_pixels']
                    prediction_count += 1
            for row in manifests[views]:
                assert len(row['cameras']) == views
            counts[name] = len(part['frames'])
        assert len(manifests[2]) == len(manifests[4]) == len(manifests[8])
        for i, row in enumerate(manifests[8]):
            assert manifests[2][i]['cameras'] == row['cameras'][:2]
            assert manifests[4][i]['cameras'] == row['cameras'][:4]
            assert manifests[2][i]['evaluation'] == manifests[4][i]['evaluation'] == row['evaluation']
            assert Path(row['evaluation']['native_depth']).is_file()
            for camera in row['cameras']:
                assert not {'depth', 'depth_valid', 'semantics', 'occupancy', 'targets'} & camera.keys()
                assert Path(camera['rgb']).is_file()
                k = np.asarray(camera['K'])
                t = np.asarray(camera['T_cam_from_rig'])
                r = t[:3, :3]
                assert np.isfinite(k).all() and np.isfinite(t).all()
                assert k[0, 0] > 0 and k[1, 1] > 0
                assert np.allclose(t[3], [0, 0, 0, 1], atol=1e-6)
                assert np.allclose(r @ r.T, np.eye(3), atol=2e-5)
                assert abs(np.linalg.det(r) - 1) < 2e-5
    assert expected_frames == frame_checks == 519 and prediction_count == 2076
    network = [json.loads(line) for line in (DATA / 'network.jsonl').read_text().splitlines()]
    assert network and all(r['local_ip'] == '172.16.2.4' and r['interface'] == 'eno5' and r['proxy'] is False for r in network)
    speed = {}
    for views in [2, 4, 8]:
        item = load(DATA / 'results/speed' / f'{views}views.json')
        assert item['completed']
        speed[str(views)] = {m: s['depth_output_ms'] for m, s in item['summary'].items()}
    free = shutil.disk_usage(DATA).free / 2**30
    assert free >= 200
    report = dict(completed=True, unix_time=time.time(), predictions=prediction_count,
                  groups_all_view_settings=expected_frames, unique_reference_groups=173,
                  prediction_coverage_min=min(coverages), prediction_coverage_max=max(coverages),
                  source_files_unchanged=len(protocol['source_sha256']),
                  weights_unchanged=list(protocol['config']['methods']), manifests=manifest_records,
                  network=dict(requests=len(network), local_ips=sorted({r['local_ip'] for r in network}),
                               remote_ips=sorted({r['remote_ip'] for r in network}),
                               downloaded_bytes=sum(r['bytes'] for r in network), proxy=False, interface='eno5'),
                  parallel_preparation_exit_codes=(DATA / 'remaining_preparation.exit_codes.txt').read_text().strip(),
                  free_gib=free, speed=speed, adapter_consistency=load(DATA / 'results/adapter_consistency.json'))
    (DATA / 'results/verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['manifests', 'speed', 'adapter_consistency']}, indent=2))


if __name__ == '__main__':
    main()
