"""ETH3D官方双目包转共同接口：只作外部测试，不读取GT决定输入范围。"""

import copy
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.data import build_dataset
from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import dataset_config


DATA = Path('/data/liuzonghao_data/dataset/ETH3D_two_view_paper')
OUTPUT = ROOT / 'runs/active/rulermvs_v6_20261101/full_training/paper_eth3d'


def pfm(path):
    with path.open('rb') as stream:
        if stream.readline().strip() != b'Pf':
            raise ValueError('GT应为单通道PFM')
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        if abs(scale) != 1:
            raise ValueError('未预期的PFM缩放')
        array = np.fromfile(stream, dtype='<f4' if scale < 0 else '>f4')
    return array.reshape(height, width)[::-1].copy()


def calibration(path):
    values = dict(line.strip().split('=', 1) for line in path.read_text().splitlines() if '=' in line)
    cameras = [np.array([[float(v) for v in row.split()] for row in values[f'cam{i}'].strip('[]').split(';')])
               for i in (0, 1)]
    baseline = float(values['baseline']) / 1000
    return cameras, baseline, float(values['doffs']), int(values['width']), int(values['height'])


def prepare():
    fixed = json.loads((DATA / 'pre_download_plan.json').read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prepared = DATA / 'prepared'
    prepared.mkdir(exist_ok=True)
    rows = {'all': [], 'nocc': []}
    checks = []
    for scene in sorted((DATA / 'raw').iterdir()):
        if not scene.is_dir():
            continue
        intrinsics, baseline, doffs, width, height = calibration(scene / 'calib.txt')
        disparity = pfm(scene / 'disp0GT.pfm')
        mask = np.asarray(Image.open(scene / 'mask0nocc.png').convert('L'))
        assert disparity.shape == mask.shape == (height, width)
        assert Image.open(scene / 'im0.png').size == Image.open(scene / 'im1.png').size == (width, height)
        assert abs(intrinsics[1][0, 2] - intrinsics[0][0, 2] - doffs) < 1e-4
        assert np.allclose(intrinsics[0][:2, :2], intrinsics[1][:2, :2])
        poses = [line.split() for line in (scene / 'images.txt').read_text().splitlines()
                 if line.strip() and not line.startswith('#')]
        assert len(poses) == 2
        assert np.allclose(np.array(poses[0][1:5], float), np.array(poses[1][1:5], float))
        translations = [np.array(p[5:8], float) for p in poses]
        np.testing.assert_allclose(translations[1] - translations[0], [-baseline, 0, 0], atol=5e-5, rtol=0)
        valid = np.isfinite(disparity) & (disparity + doffs > 0) & (mask > 0)
        nonoccluded = valid & (mask == 255)
        depth = np.zeros_like(disparity)
        depth[valid] = baseline * intrinsics[0][0, 0] / (disparity[valid] + doffs)
        np.save(prepared / f'{scene.name}.depth.npy', depth)
        np.save(prepared / f'{scene.name}.valid.npy', valid)
        np.save(prepared / f'{scene.name}.nocc.npy', nonoccluded)
        # ETH3D文档：像素原点在左上角；本项目整数坐标表示像素中心。
        intrinsics = [k.copy() for k in intrinsics]
        for k in intrinsics:
            k[:2, 2] -= .5
        ref_pose = np.array([[0., -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        source_pose = ref_pose.copy()
        source_pose[0, 3] = -baseline
        # 数值验证Z转换与双目精确投影得到同一disparity，含主点偏移。
        positions = np.flatnonzero(valid)[::max(1, int(valid.sum()) // 1000)]
        y, x = np.unravel_index(positions, valid.shape)
        rays = np.stack((x, y, np.ones_like(x)), 1) @ np.linalg.inv(intrinsics[0]).T
        points = rays * depth[y, x, None]
        points[:, 0] -= baseline
        pixels = points @ intrinsics[1].T
        projected = pixels[:, 0] / pixels[:, 2]
        error = np.max(np.abs((x - projected) - disparity[y, x]))
        assert error < 2e-5
        physical_scene = re.sub(r'_\d+[ls]$', '', scene.name)
        cameras = []
        for index, pose in enumerate((ref_pose, source_pose)):
            camera = {'camera_id': f'view_{index}', 'role': 'wide', 'frame_id': scene.name, 'timestamp': None,
                      'rgb': str((scene / f'im{index}.png').relative_to(DATA)),
                      'K': intrinsics[index].tolist(), 'T_cam_from_rig': pose.tolist(),
                      'image_size': [height, width]}
            if index == 0:
                camera['targets'] = {'frame_id': scene.name, 'depth': f'prepared/{scene.name}.depth.npy',
                                     'depth_valid': f'prepared/{scene.name}.valid.npy',
                                     'depth_type': 'optical_z', 'depth_encoding': 'metric_npy',
                                     'depth_scale': 1., 'max_valid_depth': 1e12}
            cameras.append(camera)
        row = {'sequence': f'eth3d_{physical_scene}', 'frame_id': scene.name, 'timestamp': None, 'sync_basis': 'frame_id',
               'reference_slot': 0, 'sample_type': 'rectified_stereo_pair', 'geometry_units': 'meters',
               'camera_selection': '官方全部双目对；无GT选视图', 'cameras': cameras}
        rows['all'].append(row)
        nocc = copy.deepcopy(row)
        nocc['cameras'][0]['targets']['depth_valid'] = f'prepared/{scene.name}.nocc.npy'
        rows['nocc'].append(nocc)
        checks.append({'pair': scene.name, 'physical_scene': physical_scene, 'baseline_m': baseline,
                       'gt_pixels_all': int(valid.sum()), 'gt_pixels_nocc': int(nonoccluded.sum()),
                       'roundtrip_max_disparity_error_px': float(error),
                       'calibration_sha256': digest(scene / 'calib.txt')})
    for kind, items in rows.items():
        (prepared / f'test_{kind}.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in items))
    config = json.loads((ROOT / 'runs/active/rulermvs_v6_20261101/full_training/evaluation_recovery/configs/public_main_test.json').read_text())
    config['output'] = str(OUTPUT / 'evaluation')
    config['gpus'] = [7]
    config['loader_workers'] = 0
    config['shard_size'] = 27
    first = copy.deepcopy(config['methods']['mvsanywhere_refined'])
    first['refinements'] = 0
    config['methods']['mvsanywhere_firstpass'] = first
    config['methods'].pop('v6_final')
    for name, method in config['methods'].items():
        key = {'v5_best': 'v5', 'v6_best': 'v6', 'mvsanywhere_firstpass': 'mvsanywhere',
               'mvsanywhere_refined': 'mvsanywhere'}.get(name)
        if key:
            assert digest(method['weights']) == fixed['weights_sha256'][key]
    config['datasets'] = [
        {'name': 'ETH3D_two_view_' + kind, 'root': str(DATA), 'num_views': 2, 'reference_index': 0,
         'geometry_units': 'meters', 'near': fixed['near'], 'far': fixed['far'], 'depth_scale': fixed['depth_scale'],
         'manifests': {'train': '', 'val': '', 'test': str(prepared / f'test_{kind}.jsonl')}}
        for kind in ('all', 'nocc')
    ]
    config['protocol_notes'] = [
        '真实外域固定权重测试；官方training包仅因其提供GT，在本项目全部封存作TEST。',
        '官方全部27pair属于16组名称/7场景，l/s不当独立场景。预下载计划文字16对计数已由实际包修正。',
        '双目，不复制四目；灰度按共享加载器转3通道；共同RGB256x448，匹配图仅放大。',
        '所有方法0.1–100m初始范围、depth_scale=1在下载GT前已固定；不删除范围外GT。',
        'all/nocc是相同图像两种GT掩码，不影响前向；所有方法使用同一掩码定义。',
        '本表是自定义共同深度/可见表面协议，不是官方disparity榜单或MVSAnywhere RMVD表。',
        '没有目标域训练或VAL选优；通用预训练原始图像曝光无法排除，不作绝对无曝光承诺。',
    ]
    write_json(OUTPUT / 'evaluation.json', config)
    loading = []
    for domain in config['datasets']:
        dataset = build_dataset(dataset_config(domain, config), 'test', load_targets=True)
        for index in (0, len(dataset) // 2, len(dataset) - 1):
            frame = dataset[index]
            assert tuple(frame['inputs']['rgb'].shape) == (2, 3, 256, 448)
            assert frame['targets']['depth_valid'][0].any()
            loading.append({'domain': domain['name'], 'index': index,
                            'gt_pixels': int(frame['targets']['depth_valid'][0].sum())})
    write_json(OUTPUT / 'preparation.json', {'completed': True, 'source_sha256': digest(__file__),
               'pairs': checks, 'loader_checks': loading, 'pre_download_plan': fixed,
               'download_bytes': sum(p.stat().st_size for p in (DATA / 'archives').glob('*.7z'))})
    print(json.dumps({'pairs': len(checks), 'scenes': len({c['physical_scene'] for c in checks}), 'checks': len(loading)}))


if __name__ == '__main__':
    prepare()
