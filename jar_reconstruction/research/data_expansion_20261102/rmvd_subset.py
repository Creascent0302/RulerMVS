"""官方 ETH3D 去畸变与配组的薄适配；固定 courtyard，不参与训练。"""

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from PIL import Image

from direct_http import ROOT, fetch, reserve
from expand import manifest, write_json

DATA = ROOT.parent / 'ETH3D_rmvd_subset'


class Record:
    """只承载可信来源配组元数据，不执行原 pickle 类的方法。"""


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'rmvd.data.eth3d' and name in ('ETH3DSample', 'ETH3DImage', 'ETH3DDepth'):
            return Record
        if module == 'numpy' and name in ('ndarray', 'dtype'):
            return getattr(np, name)
        if module == 'numpy.core.multiarray' and name == '_reconstruct':
            return np.core.multiarray._reconstruct
        raise ValueError(f'配组文件含未允许对象：{module}.{name}')


def samples():
    with (DATA / 'protocol/eth3d.mvsanywhere.mvd.pickle').open('rb') as stream:
        values = MetadataReader(stream).load()
    return [item for item in values if item.base == 'courtyard']


def load_official():
    source = DATA / 'protocol/compute_eth3d_undistorted.py'
    if not source.exists():
        api = source.with_suffix('.api.json')
        fetch('https://api.github.com/repos/lmb-freiburg/robustmvd/contents/'
              'rmvd/data/scripts/compute_eth3d_undistorted.py', api, max_bytes=200000)
        response = json.loads(api.read_text())
        code = base64.b64decode(response['content'])
        blob = hashlib.sha1(b'blob ' + str(len(code)).encode() + b'\0' + code).hexdigest()
        if blob != response['sha']:
            raise ValueError('官方处理代码 Git blob 不符')
        source.write_bytes(code)
    spec = importlib.util.spec_from_file_location('official_eth3d_undistort', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    official = load_official()
    audit = DATA / 'undistortion_audit.json'
    if not audit.exists():
        reserve(5 * 2**30)
        stats = official.undistort_scene(str(DATA / 'raw'), 'courtyard')
        stats['source_sha256'] = hashlib.sha256(Path(official.__file__).read_bytes()).hexdigest()
        write_json(audit, stats)
    cv_from_rig = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1.]])
    resized = DATA / 'prepared/rgb'
    resized.mkdir(parents=True, exist_ok=True)
    counts = {}
    for views in (4, 8):
        rows = []
        for index, sample in enumerate(samples()):
            data = sample.data
            reference = data['keyview_idx']
            if reference != 0:
                raise ValueError('该子集预先规定 keyview=0')
            world_from_rig = np.linalg.inv(data['poses'][reference]) @ cv_from_rig
            row = dict(sequence='eth3d_rmvd/courtyard', frame_id=str(index),
                       timestamp=None, sync_basis='frame_id', reference_slot=0, num_views=views, cameras=[],
                       synchronized_capture=False, sample_type='static_multiview',
                       geometry_units='meters', source_sample=sample.name,
                       source_view_rule='官方配组原顺序前N幅，未按GT或模型结果筛选')
            for camera_index in range(views):
                item = data['images'][camera_index]
                path = DATA / 'raw' / sample.base / item.path
                output = resized / Path(item.path).with_suffix('.png').name
                with Image.open(path) as image:
                    width, height = image.size
                    if not output.exists():
                        image.convert('RGB').resize((448, 256), Image.Resampling.BILINEAR).save(output)
                k = np.array(data['intrinsics'][camera_index], dtype=np.float64).copy()
                # COLMAP 角点坐标先转整数像素中心，再按半像素规则缩放。
                k[0, 2] -= .5
                k[1, 2] -= .5
                k[0] *= 448 / width
                k[1] *= 256 / height
                k[0, 2] += .5 * 448 / width - .5
                k[1, 2] += .5 * 256 / height - .5
                row['cameras'].append(dict(camera_id=str(camera_index), frame_id=str(index),
                    timestamp=None, rgb=str(output), K=k.tolist(),
                    T_cam_from_rig=(data['poses'][camera_index] @ world_from_rig).tolist()))
            reference_image = data['images'][reference]
            with Image.open(DATA / 'raw' / sample.base / reference_image.path) as image:
                width, height = image.size
            # 原分辨率GT另存元数据，仅用于预测后的打分；不下采样稀疏真值。
            row['evaluation'] = dict(native_shape=[height, width],
                native_depth=str(DATA / 'raw' / sample.base / official.DEPTH_OUT_DIR /
                                 Path(reference_image.path).name),
                format='float32_raw', depth_type='optical_z', depth_units='meters')
            rows.append(row)
        manifest(DATA / f'manifests/test_{views}views.jsonl', rows)
        counts[str(views)] = len(rows)
    write_json(DATA / 'ADAPTER_READY.json', dict(groups=counts, train=False,
        input_size=[256, 448], evaluation='预测双线性放回原尺寸后，使用全部有效原尺寸GT打分',
        scope='courtyard 子集固定4/8视图，不等同官方完整RMVD基准协议与成绩'))


if __name__ == '__main__':
    if not __import__('os').environ.get('TMUX'):
        raise RuntimeError('包含网络下载，必须在 tmux 中执行')
    main()
