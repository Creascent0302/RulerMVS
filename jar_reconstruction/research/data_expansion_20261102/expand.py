"""独立数据扩充任务；原始划分和权重不变，达标后才写READY。"""

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys
import traceback
import zipfile

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'tools'))
from direct_http import ROOT, RemoteZipReader, extract_members, fetch, reserve


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def read_rows(path, root=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if root is not None:
        for row in rows:
            for camera in row['cameras']:
                camera['rgb'] = str(Path(root) / camera['rgb'])
                for key in ['depth', 'depth_valid', 'semantic', 'instance', 'foreground_mask_png']:
                    if key in camera.get('targets', {}):
                        camera['targets'][key] = str(Path(root) / camera['targets'][key])
    return rows


def manifest(path, rows):
    keys = [(r['sequence'], r['frame_id']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('不接受重复参考帧计数')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def selected_zip(url, size, destination, choose, name, block_size=8 * 2**20):
    marker = ROOT / 'completed_archives' / (name + '.json')
    if marker.exists():
        return json.loads(marker.read_text())
    with zipfile.ZipFile(RemoteZipReader(url, size, block_size)) as archive:
        selected = [m for m in archive.infolist() if not m.is_dir() and choose(m.filename)]
        if not selected:
            raise ValueError(f'ZIP没有目标数据：{name}')
        selected.sort(key=lambda item: item.header_offset)
        print(f'{name}: 选择{len(selected)}文件，{sum(m.file_size for m in selected)/2**30:.2f}GiB', flush=True)
        extract_members(archive, destination, selected)
        result = {'url': url, 'archive_bytes': size, 'selected_bytes': sum(m.file_size for m in selected),
                  'members': [dict(name=m.filename, bytes=m.file_size, crc=m.CRC) for m in selected],
                  'verification': 'HTTP Content-Range + ZIP CRC；未声称整个归档SHA通过'}
    write_json(marker, result)
    return result


def hypersim(plan):
    from prepare_hypersim_subset import prepare
    root = Path(plan['root'])
    root.mkdir(parents=True, exist_ok=True)
    old = Path(plan['old_root'])
    rows = read_rows(old / 'manifests/train.jsonl', old)
    for entry in plan['scenes']:
        scene = entry['scene']
        output = root / f'manifests/by_scene/{scene}/train.jsonl'
        if not output.exists():
            def choose(name):
                return (('/_detail/cam_' in name and name.endswith('.hdf5'))
                        or name.endswith('/_detail/metadata_scene.csv')
                        or name.endswith('.tonemap.jpg') or name.endswith('.depth_meters.hdf5'))
            selected_zip(entry['url'], entry['bytes'], root / 'raw', choose, scene, 2**20)
            config = dict(root=str(root), metadata=str(PROJECT / plan['metadata']),
                          splits={'train': [scene]}, reference_slot=2, geometry_groups_per_scene=2,
                          minimum_free_disk_gib=200, manifest_directory=f'manifests/by_scene/{scene}',
                          summary_file=f'manifests/by_scene/{scene}/summary.json')
            prepare(config)
        rows.extend(read_rows(output))
        manifest(root / 'manifests/train.jsonl', rows)
        write_json(root / 'progress.json', dict(groups=len(rows), target=plan['target'], last_scene=scene))
        print('Hypersim 有效独立组数', len(rows), flush=True)
        if len(rows) >= plan['target']:
            break
    for split in ['val', 'test']:
        manifest(root / f'manifests/{split}.jsonl', read_rows(old / f'manifests/{split}.jsonl', old))
    if len(rows) < plan['minimum']:
        raise RuntimeError(f'Hypersim未达最低组数：{len(rows)}')
    return {'train_groups': len(rows), 'minimum': plan['minimum'], 'root': str(root)}


def tartan(plan):
    from prepare_tartanair import load_trajectory, choose_tuples, make_record, geometry_diagnostics
    root = Path(plan['root'])
    root.mkdir(parents=True, exist_ok=True)
    old = Path(plan['old_root'])
    rows = read_rows(old / 'prepared/manifests/train.jsonl', old)
    settings = dict(frame_window=20, minimum_baseline_m=.1, maximum_baseline_m=4.,
                    preferred_baseline_m=1., maximum_angle_degrees=30.,
                    minimum_probe_overlap=.5, probe_depths_m=[2, 5, 10, 20, 40, 80])
    official = set((PROJECT / plan['official_train_scans']).read_text().split())
    for environment in plan['environments']:
        for entry in environment['files']:
            selected_zip(entry['url'], entry['bytes'], root / 'raw', lambda name: True,
                         'tartan_' + entry['path'].replace('/', '_'))
        for folder in sorted((root / 'raw' / environment['name'] / 'Easy').glob('P*')):
            trajectory = folder.relative_to(root / 'raw').as_posix()
            if trajectory not in official:
                continue
            output = root / 'manifests/by_trajectory' / (trajectory.replace('/', '_') + '.jsonl')
            if not output.exists():
                poses, stats = load_trajectory(folder, 1000.)
                groups, rejected = choose_tuples(poses, settings)
                records = [make_record(root, folder, trajectory, g, poses, 2, 1000.) for g in groups]
                manifest(output, records)
                diagnostics = geometry_diagnostics(folder, poses, groups, 1000., 2)
                write_json(output.with_suffix('.audit.json'), dict(groups=len(groups), rejected=len(rejected),
                           frames=len(stats), geometry=diagnostics, selection_uses_ground_truth=False))
            rows.extend(read_rows(output))
        manifest(root / 'manifests/train.jsonl', rows)
        write_json(root / 'progress.json', dict(groups=len(rows), target=plan['target']))
        print('TartanAir 有效独立组数', len(rows), flush=True)
        if len(rows) >= plan['target']:
            break
    manifest(root / 'manifests/val.jsonl', read_rows(old / 'prepared/manifests/val.jsonl', old))
    # 旧TEST已与官方训练名单重合，仅保留原文件，不复制进新的未见TEST。
    if len(rows) < plan['minimum']:
        raise RuntimeError(f'TartanAir未达最低组数：{len(rows)}')
    return {'train_groups': len(rows), 'minimum': plan['minimum'], 'root': str(root)}


def dynamic(plan):
    from prepare_dynamicreplica import camera_from_annotation, CV_FROM_RIG, decode_depth
    root = Path(plan['root'])
    raw = root / 'raw/train'
    raw.mkdir(parents=True, exist_ok=True)
    annotation_file = raw / 'frame_annotations_train.jgz'
    if not annotation_file.exists():
        entry = plan['annotation']
        archive_path = ROOT / 'temporary/train_000.zip'
        fetch(entry['url'], archive_path, max_bytes=entry['bytes'])
        digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if digest != entry['sha256']:
            raise ValueError('官方DynamicReplica相机标注SHA不符')
        with zipfile.ZipFile(archive_path) as archive:
            extract_members(archive, raw, [m for m in archive.infolist() if m.filename.endswith('.jgz')])
        archive_path.unlink()
    with gzip.open(annotation_file, 'rt') as stream:
        annotations = json.load(stream)
    lookup = collections.defaultdict(dict)
    for item in annotations:
        lookup[item['sequence_name']][(item['frame_number'], item['camera_name'])] = item
    rows = []
    official = set((PROJECT / plan['official_train_scans']).read_text().split())
    def download(entry):
        return selected_zip(entry['url'], entry['bytes'], raw,
                            lambda name: any(f'/{s}/' in name for s in ['images', 'depths', 'masks']),
                            'dynamic_' + entry['path'])
    # 不同分包含不同序列；并行I/O，不重复创建训练组或复制稠密浮点标签。
    # 每个分包仍独立检查200GiB余量，前四包24序列共7200组。
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(download, plan['archives'][:4]))
    for entry in plan['archives']:
        result = download(entry)
        sequences = sorted({m['name'].split('_source_')[0] for m in result['members']})
        for sequence in sequences:
            if sequence not in official:
                raise ValueError(f'原训练分包序列不在官方训练扫描名单：{sequence}')
            output = root / f'manifests/by_sequence/{sequence}.jsonl'
            if not output.exists():
                records = []
                group = lookup[sequence]
                for frame in sorted({k[0] for k in group}):
                    pair = [group[(frame, name)] for name in ['left', 'right']]
                    if abs(pair[0]['frame_timestamp'] - pair[1]['frame_timestamp']) > 1e-8:
                        raise ValueError('双目不同步')
                    k, e = zip(*(camera_from_annotation(a) for a in pair))
                    world_from_rig = np.linalg.inv(e[0]) @ CV_FROM_RIG
                    record = dict(sequence='dynamicreplica_' + sequence, source_sequence=sequence,
                                  frame_id=str(frame), timestamp=pair[0]['frame_timestamp'],
                                  source_split='train', reference_slot=0, num_views=2,
                                  geometry_units='scene_unit', sample_type='synchronized_stereo',
                                  synchronized_capture=True, cameras=[])
                    for i, annotation in enumerate(pair):
                        image_path = raw / annotation['image']['path']
                        depth_path = raw / annotation['depth']['path']
                        with Image.open(image_path) as im:
                            im.load()
                            if im.size[::-1] != tuple(annotation['image']['size']):
                                raise ValueError('RGB尺寸不一致')
                        depth = decode_depth(depth_path)
                        if depth.shape != tuple(annotation['image']['size']) or annotation['depth']['scale_adjustment'] != 1:
                            raise ValueError('深度尺寸或原始尺度不符')
                        record['cameras'].append(dict(camera_id=annotation['camera_name'], frame_id=str(frame),
                            timestamp=annotation['frame_timestamp'],
                            rgb=image_path.relative_to(root).as_posix(), K=k[i].tolist(),
                            T_cam_from_rig=(e[i] @ world_from_rig).tolist(),
                            targets=dict(frame_id=str(frame), depth=depth_path.relative_to(root).as_posix(),
                                depth_type='optical_z', depth_units='scene_unit', depth_scale=1.,
                                depth_encoding='dynamic_replica_float16_png', max_valid_depth=1000.)))
                    records.append(record)
                manifest(output, records)
            rows.extend(read_rows(output))
        manifest(root / 'manifests/train.jsonl', rows)
        write_json(root / 'progress.json', dict(groups=len(rows), target=plan['target']))
        print('DynamicReplica 有效独立组数', len(rows), flush=True)
        if len(rows) >= plan['target']:
            break
    old = Path(plan['old_root'])
    for split in ['val', 'test']:
        path = old / f'prepared/manifests/project_development_v1/{split}.jsonl'
        manifest(root / f'manifests/{split}.jsonl', read_rows(path, old))
    if len(rows) < plan['minimum']:
        raise RuntimeError(f'DynamicReplica未达最低组数：{len(rows)}')
    return {'train_groups': len(rows), 'minimum': plan['minimum'], 'root': str(root),
            'storage': '原PNG无损按float16位模式解码；不复制完整float32数组'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['hypersim', 'tartan', 'dynamic'])
    args = parser.parse_args()
    plan = json.loads((Path(__file__).parent / 'plan.json').read_text())[args.dataset]
    if not __import__('os').environ.get('TMUX'):
        raise RuntimeError('下载必须在tmux内运行')
    status = ROOT / (args.dataset + '_status.json')
    try:
        write_json(status, {'state': 'running', 'plan': plan})
        result = globals()[args.dataset](plan)
        write_json(status, {'state': 'ready', **result})
    except Exception:
        write_json(status, {'state': 'failed', 'error': traceback.format_exc()})
        raise
