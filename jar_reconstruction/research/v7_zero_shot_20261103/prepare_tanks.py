"""按官方RMVD配组下载Tanks and Temples目标ZIP成员，避免下载整套深度包。"""

import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from prepare import DATA, write, sha, direct_http, rmvd_subset


class Reader(rmvd_subset.MetadataReader):
    def find_class(self, module, name):
        if module == 'rmvd.data.tanks_and_temples' and name in (
                'TanksAndTemplesSample', 'TanksAndTemplesImage', 'TanksAndTemplesDepth'):
            return rmvd_subset.Record
        return super().find_class(module, name)


def main():
    assert os.environ.get('TMUX')
    metadata = DATA / 'protocol/tanks_and_temples.robustmvd.mvd.pickle'
    with metadata.open('rb') as stream:
        samples = Reader(stream).load()
    assert len(samples) == 69
    assert {x.base for x in samples} == {'Barn', 'Church', 'Courthouse', 'Ignatius'}
    assert all(x.data['keyview_idx'] == 0 for x in samples)
    plan = dict(completed=False, groups=len(samples), scenes=sorted({s.base for s in samples}),
                selected_without_gt_or_predictions=True, views=[2, 4, 8], input_range=[.1, 100.],
                unit_scale=1., source_metadata_sha256=sha(metadata),
                scope='官方RMVD四场景69组的共同2/4/8视图协议；不使用官方GT导出的depth_range')
    write(DATA / 'results/tanks_plan.json', plan)
    sources = json.loads((DATA / 'tanks_probe.json').read_text())
    files = {'images': {f'{s.base}/{im.path}' for s in samples for im in s.data['images'][:8]},
             'depth': {f'{s.base}/{s.data["depth"].path}' for s in samples}}
    shapes = {}
    source_records = []
    for item, kind in zip(sources, ['images', 'depth']):
        remote = direct_http.RemoteZipReader(item['url'], item['bytes'], block_size=2 * 2**20)
        with zipfile.ZipFile(remote) as archive:
            infos = archive.infolist()
            for logical in sorted(files[kind]):
                candidates = [v for v in infos if v.filename == logical or v.filename.endswith('/' + logical)]
                if len(candidates) != 1:
                    raise ValueError(f'ZIP成员无法唯一定位: {logical}')
                info = candidates[0]
                destination = DATA / 'tanks' / kind / logical
                destination.parent.mkdir(parents=True, exist_ok=True)
                direct_http.reserve(2 * info.file_size)
                with archive.open(info) as stream:
                    raw = stream.read()  # 读完触发zipfile的CRC校验。
                if kind == 'images':
                    with Image.open(io.BytesIO(raw)) as image:
                        shapes[logical] = list(reversed(image.size))
                        destination = destination.with_suffix('.png')
                        image.convert('RGB').resize((448, 256), Image.Resampling.BILINEAR).save(destination)
                else:
                    destination.write_bytes(raw)
                source_records.append(dict(member=info.filename, uncompressed_bytes=info.file_size,
                                           crc32=f'{info.CRC:08x}', prepared_path=str(destination), sha256=sha(destination)))
                print(json.dumps(dict(kind=kind, member=logical)), flush=True)
    transform = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1.]])
    for scene in plan['scenes']:
        chosen = [s for s in samples if s.base == scene]
        for views in plan['views']:
            rows = []
            for index, sample in enumerate(chosen):
                data = sample.data
                shape = shapes[f'{scene}/{data["images"][0].path}']
                depth_path = DATA / 'tanks/depth' / scene / data['depth'].path
                with np.load(depth_path) as depth:
                    assert list(depth['arr_0'].shape) == shape, 'GT与RGB必须处于同一原始像素格'
                row = dict(sequence='tanks_and_temples/' + scene, frame_id=str(index), timestamp=None,
                    sync_basis='frame_id', reference_slot=0, num_views=views, cameras=[],
                    synchronized_capture=False, sample_type='static_multiview', geometry_units='meters',
                    source_sample=sample.name, scene='Tanks_' + scene, category='other', historical=False,
                    source_view_rule='官方顺序前N图，忽略GT范围',
                    evaluation=dict(native_shape=shape, native_depth=str(depth_path), format='npz',
                                    array_key='arr_0', depth_type='optical_z', depth_units='meters'))
                world_from_rig = np.linalg.inv(data['poses'][0]) @ transform
                for slot, image in enumerate(data['images'][:views]):
                    h, w = shapes[f'{scene}/{image.path}']
                    k = np.array(data['intrinsics'][slot], dtype=np.float64).copy()
                    k[0] *= 448 / w
                    k[1] *= 256 / h
                    k[0, 2] -= .5
                    k[1, 2] -= .5
                    row['cameras'].append(dict(camera_id=str(slot), frame_id=str(index), timestamp=None,
                        rgb=str((DATA / 'tanks/images' / scene / image.path).with_suffix('.png')), K=k.tolist(),
                        T_cam_from_rig=(data['poses'][slot] @ world_from_rig).tolist()))
                rows.append(row)
            path = DATA / 'manifests' / f'Tanks_{scene}_{views}views.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    plan.update(completed=True, selected_archive_members=source_records,
                network='仅eno5直连HTTPS Range；只认证已下载成员CRC，不宣称验证整ZIP')
    write(DATA / 'results/tanks_plan.json', plan)


if __name__ == '__main__':
    main()
