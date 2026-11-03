"""ETH3D官方RMVD配组的零样本测试准备；网络只允许服务器eno5直连。"""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'research/data_expansion_20261102'))
import direct_http
import rmvd_subset

DATA = Path('/data/liuzonghao_data/dataset/V7_zero_shot_20261103')
OLD = rmvd_subset.DATA
SCENES = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'meadow',
          'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains']
INDOOR = {'delivery_area', 'kicker', 'office', 'pipes', 'relief', 'relief_2'}
direct_http.ROOT = DATA


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            result.update(block)
    return result.hexdigest()


def samples():
    with (OLD / 'protocol/eth3d.mvsanywhere.mvd.pickle').open('rb') as stream:
        return rmvd_subset.MetadataReader(stream).load()


def freeze():
    previous = PROJECT / 'runs/active/overnight_v6_v7_mvs_20261103/comparison'
    config = json.loads((previous / 'config.json').read_text())
    config.update(datasets=[], output=str(DATA / 'results/accuracy'), gpus=[0, 1, 2, 3],
                  loader_workers=0, cpu_threads=2)
    config['fast']['cache_dir'] = str(DATA / 'results/compile_cache')
    config['timing'].update(test_samples_per_domain=13)
    config['protocol_notes'] = [
        '固定公共VAL-best，不微调、不测试时适配、不使用GT确定范围或做尺度对齐。',
        'ETH3D官方RMVD MVSAnywhere配组原顺序前2/4/8视图，主结果四视图。',
        '12个新增场景主外测，courtyard历史复查另列；不是完整五域RMVD榜单。',
        'RGB256x448，全部正有限原尺寸GT，统一输入范围0.1–100米，内部尺度1。',
        '主报告场景等权MAE/RMSE/AbsRel，另报像素池化、delta1和coverage。',
        '配对bootstrap以场景为单位，20000次seed20261103；仅估计场景变化，不代表多训练seed。',
        '零样本指任务训练未用ETH3D；通用预训练曝光不能认证绝对未见。']
    write(DATA / 'results/base_config.json', config)
    plan = dict(completed=False, scenes=SCENES, new_scenes=[s for s in SCENES if s != 'courtyard'],
                views=[2, 4, 8], primary_views=4, indoor=sorted(INDOOR), seed=20261103,
                metric_regions={'near': [0, 5], 'mid': [5, 20], 'far': [20, 'inf']},
                weights_sha256={k: sha(Path(v['weights'])) for k, v in config['methods'].items()},
                reference_protocol_sha256=sha(OLD / 'protocol/eth3d.mvsanywhere.mvd.pickle'),
                remap_source_sha256=sha(OLD / 'protocol/compute_eth3d_undistorted.py'),
                network=dict(interface='eno5', local_ip='172.16.2.4', proxies=False, fallback=False),
                free_space_floor_gib=200, protocol=config['protocol_notes'])
    write(DATA / 'results/plan.json', plan)


def extract_selected(archive, destination, names):
    direct_http.reserve(2 * 2**30)
    members = subprocess.check_output(['bsdtar', '-tf', str(archive)], text=True).splitlines()
    missing = set(names) - set(members)
    if missing:
        raise ValueError(f'归档缺少所需成员: {sorted(missing)[:3]}')
    # 只提取固定官方配组所需文件；bsdtar验证被读取归档数据的校验。
    subprocess.run(['bsdtar', '-xf', str(archive), '-C', str(destination), *names], check=True)


def acquire(scene, chosen):
    temp = DATA / 'temporary' / scene
    temp.mkdir(parents=True, exist_ok=True)
    direct_http.reserve(8 * 2**30)
    records = []
    refs = sorted({Path(x.data['images'][0].path).name for x in chosen})
    images = sorted({x.base + '/' + image.path for x in chosen for image in x.data['images'][:8]})
    for kind in ['undistorted', 'jpg', 'depth']:
        archive = temp / f'{scene}_dslr_{kind}.7z'
        if not archive.exists():
            part = archive.with_suffix('.part')
            record = direct_http.fetch('https://www.eth3d.net/data/' + archive.name, part,
                                       max_bytes=2 * 2**30, timeout=3600)
            part.replace(archive)
        else:
            record = dict(resumed=True)
        record.update(file=archive.name, bytes=archive.stat().st_size, sha256=sha(archive))
        if kind == 'undistorted':
            wanted = images + [f'{scene}/dslr_calibration_undistorted/cameras.txt']
            extract_selected(archive, temp, wanted)
        elif kind == 'jpg':
            wanted = [f'{scene}/dslr_calibration_jpg/{x}.txt' for x in ['cameras', 'images']]
            wanted += [f'{scene}/images/dslr_images/{name}' for name in refs]
            extract_selected(archive, temp, wanted)
        else:
            wanted = [f'{scene}/ground_truth_depth/dslr_images/{name}' for name in refs]
            extract_selected(archive, temp, wanted)
        records.append(record)
        print(json.dumps(dict(event='downloaded', scene=scene, archive=archive.name,
                              bytes=archive.stat().st_size)), flush=True)
    write(DATA / 'audit' / f'{scene}_acquisition.json', records)
    return temp / scene


def prepare_scene(scene):
    ready = DATA / 'audit' / f'{scene}_ready.json'
    if ready.exists():
        return
    chosen = [x for x in samples() if x.base == scene]
    assert len(chosen) == 8 and all(x.data['keyview_idx'] == 0 for x in chosen)
    source = OLD / 'raw/courtyard' if scene == 'courtyard' else acquire(scene, chosen)
    official = rmvd_subset.load_official()  # 已在本地，禁止重新获取隐式依赖。
    calibration = official.read_cameras_txt(str(source / official.CALIB_DIR / 'cameras.txt'))
    image_ids = official.read_image_camera_ids(str(source / official.CALIB_DIR / 'images.txt'))
    target_cameras = {}
    for camera_id, (model, width, height, params) in calibration.items():
        assert model == 'THIN_PRISM_FISHEYE'
        w, h, p = official.undistort_camera(width, height, params)
        target_cameras[camera_id] = ('PINHOLE', w, h, p)
    official.verify_cameras(scene, target_cameras, str(source / official.CALIB_REF_DIR / 'cameras.txt'))
    result = dict(scene=scene, split='historical' if scene == 'courtyard' else 'new',
                  category='indoor' if scene in INDOOR else 'outdoor', images={}, depth={})
    refs_per_camera = defaultdict(list)
    for name in sorted({Path(s.data['images'][0].path).name for s in chosen}):
        refs_per_camera[image_ids[name]].append(name)
    for camera_id, names in refs_per_camera.items():
        _, width, height, params = calibration[camera_id]
        _, out_width, out_height, out_params = target_cameras[camera_id]
        mapping = official.compute_forward_map(width, height, params, out_params)
        roundtrip = official.check_maps_consistent(params, out_params, mapping)
        backward = official.compute_backward_map(params, out_width, out_height, out_params)
        original = cv2.imread(str(source / official.IMAGE_DIR / names[0]))
        warped_rgb = official.warp_image(original, backward)
        rgb_difference = official.compare_to_reference(warped_rgb, str(source / official.IMAGE_REF_DIR / names[0]))
        del backward, original, warped_rgb
        for name in names:
            direct_http.reserve(512 * 2**20)
            raw = np.fromfile(source / official.DEPTH_DIR / name, np.float32).reshape(height, width)
            depth, num_in, num_out, num_oob = official.warp_depth(raw, mapping, out_width, out_height)
            output = DATA / 'prepared/depth' / scene / (name + '.npz')
            output.parent.mkdir(parents=True, exist_ok=True)
            if scene == 'courtyard':
                old = np.fromfile(source / official.DEPTH_OUT_DIR / name, np.float32).reshape(depth.shape)
                assert np.array_equal(depth, old), '新适配深度必须与既有官方处理逐位相同'
            np.savez_compressed(output, depth=depth)
            result['depth'][name] = dict(path=str(output), shape=list(depth.shape), sha256=sha(output),
                                         input_points=num_in, output_points=num_out, out_of_bounds=num_oob,
                                         roundtrip_pixels=roundtrip, rgb_mean_max_difference=rgb_difference)
            del raw, depth
        del mapping
    rgb_dir = DATA / 'prepared/rgb' / scene
    rgb_dir.mkdir(parents=True, exist_ok=True)
    for image in {x.path for sample in chosen for x in sample.data['images'][:8]}:
        source_image = source / image
        output = rgb_dir / (Path(image).stem + '.png')
        with Image.open(source_image) as frame:
            native_shape = list(reversed(frame.size))
            frame.convert('RGB').resize((448, 256), Image.Resampling.BILINEAR).save(output)
        result['images'][Path(image).name] = dict(path=str(output), shape=native_shape, sha256=sha(output))
    cv_from_rig = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1.]])
    for views in [2, 4, 8]:
        rows = []
        for index, sample in enumerate(chosen):
            data = sample.data
            world_from_rig = np.linalg.inv(data['poses'][0]) @ cv_from_rig
            row = dict(sequence='eth3d_rmvd/' + scene, frame_id=str(index), timestamp=None,
                       sync_basis='frame_id', reference_slot=0, num_views=views, cameras=[],
                       synchronized_capture=False, sample_type='static_multiview', geometry_units='meters',
                       source_sample=sample.name, scene=scene, category=result['category'],
                       historical=scene == 'courtyard', source_view_rule='官方原顺序前N图，不读GT选图')
            for slot, image in enumerate(data['images'][:views]):
                name = Path(image.path).name
                info = result['images'][name]
                h, w = info['shape']
                k = np.array(data['intrinsics'][slot], dtype=np.float64).copy()
                p = target_cameras[image_ids[name]][3]
                assert np.allclose(k, [[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]], atol=1e-3)
                # 官方K采用像素角点原点；转整数中心并按半像素规则缩放。
                k[0] *= 448 / w
                k[1] *= 256 / h
                k[0, 2] -= .5
                k[1, 2] -= .5
                row['cameras'].append(dict(camera_id=str(slot), frame_id=str(index), timestamp=None,
                    rgb=info['path'], K=k.tolist(),
                    T_cam_from_rig=(data['poses'][slot] @ world_from_rig).tolist()))
            reference_name = Path(data['images'][0].path).name
            row['evaluation'] = dict(native_shape=result['depth'][reference_name]['shape'],
                native_depth=result['depth'][reference_name]['path'], format='npz',
                depth_type='optical_z', depth_units='meters')
            rows.append(row)
        manifest = DATA / 'manifests' / f'{scene}_{views}views.jsonl'
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    write(ready, result)
    if scene != 'courtyard':
        # 仅删除本脚本新建的该场景临时归档与解压物；准备结果和来源校验保留。
        shutil.rmtree(DATA / 'temporary' / scene)
    print(json.dumps(dict(event='ready', scene=scene, groups=8, free_gib=shutil.disk_usage(DATA).free / 2**30)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--scene', choices=SCENES)
    args = parser.parse_args()
    if args.freeze:
        freeze()
        return
    if not os.environ.get('TMUX'):
        raise RuntimeError('下载必须在tmux中运行，且仅走服务器eno5')
    assert (OLD / 'protocol/compute_eth3d_undistorted.py').exists()
    for scene in [args.scene] if args.scene else SCENES:
        prepare_scene(scene)
    write(DATA / 'preparation_status.json', dict(completed=all((DATA / 'audit' / f'{s}_ready.json').exists() for s in SCENES)))


if __name__ == '__main__':
    main()
