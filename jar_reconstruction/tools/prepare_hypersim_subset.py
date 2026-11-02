"""将选定完整Hypersim场景转换为米制四视图；按场景隔离，不读GT选视角。"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_hypersim_planar import optical_axis_factor
from prepare_hypersim_sample import load_cameras, read_h5, project, check_depth_geometry
from download_public_dataset import require_space


def select_groups(poses, intrinsics, image_size):
    height, width = image_size
    u, v = np.meshgrid(np.linspace(0.1 * width, 0.9 * width - 1, 10),
                       np.linspace(0.1 * height, 0.9 * height - 1, 8))
    rays = np.stack((u, v, np.ones_like(u)), -1).reshape(-1, 3) @ np.linalg.inv(intrinsics).T
    probes = (rays[None] * np.array([0.5, 1, 2, 4, 8, 16])[:, None, None]).reshape(-1, 3)
    groups, rejected = [], []
    inverse = {index: np.linalg.inv(pose) for index, pose in poses.items()}
    for reference, pose in poses.items():
        candidates = []
        for index, other in poses.items():
            if reference == index:
                continue
            relative = inverse[index] @ pose
            baseline = float(np.linalg.norm(relative[:3, 3]))
            angle = float(np.arccos(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1)))
            if not 0.02 <= baseline <= 10 or angle > np.pi / 3:
                continue
            uv, z = project(probes, relative, intrinsics)
            overlap = ((z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
                       & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)).mean()
            if overlap >= 0.35:
                candidates.append((baseline + 2 * angle + 1 - overlap, index))
        chosen = []
        for _, index in sorted(candidates):
            if all(np.linalg.norm(poses[index][:3, 3] - poses[j][:3, 3]) >= 0.02 for j in chosen):
                chosen.append(index)
            if len(chosen) == 3:
                break
        if len(chosen) == 3:
            groups.append({'reference': reference, 'sources': chosen})
        else:
            rejected.append(reference)
    return groups, rejected


def prepare(config):
    root = Path(config['root']).resolve()
    splits = config['splits']
    scenes = [scene for values in splits.values() for scene in values]
    if len(set(scenes)) != len(scenes):
        raise ValueError('同一场景不能出现在不同划分')
    metadata = Path(config['metadata']).resolve()
    reference_slot = config.get('reference_slot', 2)
    cv_from_rig = np.eye(4)
    cv_from_rig[:3, :3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    output = root / config.get('manifest_directory', 'manifests')
    output.mkdir(parents=True, exist_ok=True)
    summary = {'config': config, 'geometry_units': 'meters', 'splits': {}, 'cameras': {}}
    for split, scene_list in splits.items():
        records = []
        for scene in scene_list:
            detail = root / 'raw' / scene / '_detail'
            for camera_folder in sorted(detail.glob('cam_*')):
                if not camera_folder.is_dir():
                    continue
                camera = camera_folder.name
                folder, size, matrix, k, scale, poses = load_cameras(
                    root, metadata, scene, camera, allow_uniform_scale=True)
                raw_bases = read_h5(camera_folder / 'camera_keyframe_orientations.hdf5')
                raw_indices = read_h5(camera_folder / 'camera_keyframe_frame_indices.hdf5').reshape(-1)
                basis_scales = np.linalg.norm(raw_bases, axis=(1, 2)) / np.sqrt(3)
                normalized_bases = {str(int(index)): float(value) for index, value in zip(raw_indices, basis_scales)
                                    if not np.isclose(value, 1, atol=1e-5)}
                if not poses:
                    continue
                if 'minimum_free_disk_gib' in config:
                    import shutil
                    available = shutil.disk_usage(root).free
                    needed = len(poses) * size[0] * size[1] * 4
                    if available < config['minimum_free_disk_gib'] * 2**30 + needed:
                        raise RuntimeError('Hypersim转换将超过配置的磁盘保留空间')
                else:
                    require_space(root, len(poses) * size[0] * size[1] * 4)
                groups, rejected = select_groups(poses, k, size)
                factor = optical_axis_factor(matrix, *size)
                v, u = np.mgrid[0:size[0]:8, 0:size[1]:8]
                rays = np.stack((u, v, np.ones_like(u)), -1) @ np.linalg.inv(k).T
                ray_error = float(np.abs(1 / np.linalg.norm(rays, axis=-1) - factor[::8, ::8]).max())
                if ray_error > 1e-5:
                    raise ValueError(f'{scene}/{camera} 距离转换与像素射线不一致')
                processed = root / 'processed' / scene / camera
                processed.mkdir(parents=True, exist_ok=True)
                paths, statistics = {}, []
                for index in poses:
                    stem = f'frame.{index:04d}'
                    rgb = folder / 'images' / f'scene_{camera}_final_preview' / f'{stem}.tonemap.jpg'
                    with Image.open(rgb) as image:
                        image.load()
                        if image.size[::-1] != size or image.mode != 'RGB':
                            raise ValueError(f'RGB尺寸/格式错误: {rgb}')
                    distance = read_h5(folder / 'images' / f'scene_{camera}_geometry_hdf5' / f'{stem}.depth_meters.hdf5')
                    if distance.shape != size:
                        raise ValueError(f'深度尺寸错误: {rgb}')
                    depth = np.asarray(distance * factor, dtype=np.float32)
                    valid = np.isfinite(depth) & (depth > 0)
                    depth[~valid] = np.nan
                    target = processed / f'{stem}.optical_z.npy'
                    with target.with_suffix('.tmp').open('wb') as stream:
                        np.save(stream, depth, allow_pickle=False)
                    target.with_suffix('.tmp').replace(target)
                    paths[index] = {'rgb': rgb.relative_to(root).as_posix(),
                                    'depth': target.relative_to(root).as_posix()}
                    statistics.append({'frame_id': index, 'valid_fraction': float(valid.mean())})
                diagnostics = []
                count = min(len(groups), int(config.get('geometry_groups_per_scene', 4)))
                for position in np.unique(np.linspace(0, len(groups) - 1, count).astype(int)):
                    group = groups[position]
                    order = [group['reference'], *group['sources']]
                    depths = {index: np.load(root / paths[index]['depth'], allow_pickle=False) for index in order}
                    diagnostics.extend(check_depth_geometry(depths, order, poses, k))
                for group in groups:
                    ordered = list(group['sources'])
                    ordered.insert(reference_slot, group['reference'])
                    tuple_id = f"reference_{group['reference']:04d}"
                    world_from_rig = poses[group['reference']] @ cv_from_rig
                    record = {'sequence': f'hypersim_{scene}_{camera}', 'frame_id': tuple_id,
                              'timestamp': None, 'sync_basis': 'frame_id', 'geometry_units': 'meters',
                              'sample_type': 'static_multiview_tuple', 'synchronized_capture': False,
                              'reference_slot': reference_slot, 'selection_uses_ground_truth': False, 'cameras': []}
                    for slot, index in enumerate(ordered):
                        record['cameras'].append({
                            'camera_id': f'view_{slot}', 'role': 'wide', 'frame_id': tuple_id,
                            'timestamp': None, 'source_frame_id': index, 'source_camera_id': camera,
                            'rgb': paths[index]['rgb'], 'K': k.tolist(),
                            'T_cam_from_rig': (np.linalg.inv(poses[index]) @ world_from_rig).tolist(),
                            'targets': {'frame_id': tuple_id, 'depth': paths[index]['depth'],
                                        'depth_type': 'optical_z', 'depth_encoding': 'metric_npy',
                                        'depth_scale': 1.0, 'max_valid_depth': 1000.0},
                        })
                    records.append(record)
                audit = {'split': split, 'frames_checked': len(poses), 'tuples': len(groups),
                         'positive_uniform_basis_scales_removed': normalized_bases,
                         'reference_frames_without_three_sources': rejected, 'meters_per_asset_unit': scale,
                         'ray_factor_max_error': ray_error, 'depth_statistics': statistics,
                         'geometry_diagnostics': diagnostics, 'selection_uses_ground_truth': False}
                (processed / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
                summary['cameras'][f'{scene}/{camera}'] = {
                    key: value for key, value in audit.items() if key not in ('depth_statistics', 'geometry_diagnostics')}
                print(f'{split} {scene}/{camera}: {len(poses)}帧，{len(groups)}组', flush=True)
        path = output / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(record) + '\n' for record in records))
        if not records:
            raise ValueError(f'{split}没有四视图，保留审计，不复制凑数')
        summary['splits'][split] = {'tuples': len(records), 'manifest': str(path), 'scenes': scene_list}
    summary_path = root / config.get('summary_file', 'preparation_summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    prepare(json.loads(Path(args.config).read_text()))
