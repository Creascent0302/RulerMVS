"""新原生深度读取与传输/配组边界的针对性验证。"""
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).parent))
from jar.data import CarlaFrameDataset
import direct_http
from expand import manifest


def test_float16_png_matches_npy_and_label_isolation(tmp_path):
    values = np.array([[0., 1., 2., np.nan], [np.inf, 3.25, 999., 1000.]], dtype=np.float16)
    Image.fromarray(values.view(np.uint16)).save(tmp_path / 'depth.png')
    Image.new('RGB', (4, 2), color=(80, 120, 150)).save(tmp_path / 'rgb.png')
    np.save(tmp_path / 'depth.npy', values.astype(np.float32))
    row = dict(sequence='test', frame_id='0', timestamp=None, sync_basis='frame_id',
               geometry_units='scene_unit', cameras=[dict(frame_id='0', timestamp=None,
               rgb='rgb.png', K=[[3., 0, 1.5], [0, 3., .5], [0, 0, 1]],
               T_cam_from_rig=np.eye(4).tolist(), targets=dict(frame_id='0', depth='depth.png',
               depth_encoding='dynamic_replica_float16_png', depth_type='optical_z',
               depth_units='scene_unit', max_valid_depth=1000.))])
    path = tmp_path / 'manifest.jsonl'
    path.write_text(json.dumps(row) + '\n')
    png = CarlaFrameDataset(str(tmp_path), str(path), geometry_units='scene_unit')[0]
    row['cameras'][0]['targets'].update(depth='depth.npy', depth_encoding='scene_unit_npy')
    path.write_text(json.dumps(row) + '\n')
    npy = CarlaFrameDataset(str(tmp_path), str(path), geometry_units='scene_unit')[0]
    torch.testing.assert_close(png['targets']['depth'], npy['targets']['depth'], equal_nan=True)
    assert torch.equal(png['targets']['depth_valid'], npy['targets']['depth_valid'])
    row['cameras'][0]['targets']['depth'] = 'does_not_exist.png'
    path.write_text(json.dumps(row) + '\n')
    no_labels = CarlaFrameDataset(str(tmp_path), str(path), load_targets=False, geometry_units='scene_unit')[0]
    assert not no_labels['targets']
    for key, value in png['inputs'].items():
        if torch.is_tensor(value):
            assert torch.equal(value, no_labels['inputs'][key])


def test_remote_zip_read_boundaries_and_crc(tmp_path, monkeypatch):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as writer:
        writer.writestr('nested/data.bin', bytes(range(256)) * 100)
    payload = archive.getvalue()
    monkeypatch.setattr(direct_http, 'ROOT', tmp_path)
    (tmp_path / 'temporary').mkdir()
    def fake_fetch(url, output, byte_range, max_bytes, timeout):
        start, stop = byte_range
        assert stop - start + 1 == max_bytes
        Path(output).write_bytes(payload[start:stop + 1])
    monkeypatch.setattr(direct_http, 'fetch', fake_fetch)
    reader = direct_http.RemoteZipReader('https://example.org/test.zip', len(payload), block_size=97)
    with zipfile.ZipFile(reader) as source:
        assert source.read('nested/data.bin') == bytes(range(256)) * 100


def test_no_duplicate_training_groups(tmp_path):
    row = dict(sequence='sequence', frame_id='1')
    with pytest.raises(ValueError, match='重复'):
        manifest(tmp_path / 'train.jsonl', [row, row])


def test_archive_path_escape_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(direct_http, 'reserve', lambda *args: None)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as writer:
        writer.writestr('../escape.txt', 'unsafe')
    with zipfile.ZipFile(buffer) as reader, pytest.raises(ValueError, match='不安全'):
        direct_http.extract_members(reader, tmp_path / 'raw', reader.infolist())
