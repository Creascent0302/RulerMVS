"""固定选择courtyard，获取RMVD的ETH3D多视图素材及作者配组名单。"""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from direct_http import ROOT, fetch, reserve


def main():
    root = ROOT.parent / 'ETH3D_rmvd_subset'
    (root / 'archives').mkdir(parents=True, exist_ok=True)
    (root / 'raw').mkdir(exist_ok=True)
    sources = [('courtyard_dslr_undistorted.7z', 500990569),
               ('courtyard_dslr_depth.7z', 428289105),
               ('courtyard_dslr_jpg.7z', 425492376)]
    for name, size in sources:
        target = root / 'archives' / name
        if not target.exists():
            temporary = target.with_suffix('.part')
            fetch('https://www.eth3d.net/data/' + name, temporary, max_bytes=size, timeout=3600)
            if temporary.stat().st_size != size:
                raise ValueError('ETH3D下载长度不符')
            temporary.replace(target)
        marker = root / (name + '.extracted')
        if not marker.exists():
            reserve(5 * 2**30)
            subprocess.run(['bsdtar', '-xf', str(target), '-C', str(root / 'raw')], check=True)
            marker.write_text(hashlib.sha256(target.read_bytes()).hexdigest() + '\n')
    metadata = root / 'protocol'
    metadata.mkdir(exist_ok=True)
    for filename in ['eth3d.mvsanywhere.mvd.pickle', 'eth3d.robustmvd.mvd.pickle']:
        target = metadata / filename
        if not target.exists():
            response = metadata / (filename + '.api.json')
            fetch('https://api.github.com/repos/lmb-freiburg/robustmvd/contents/rmvd/data/sample_lists/' + filename,
                  response, max_bytes=2**20)
            obj = json.loads(response.read_text())
            content = base64.b64decode(obj['content'])
            blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
            if blob != obj['sha']:
                raise ValueError('RMVD官方配组Git blob不符')
            target.write_bytes(content)
    (root / 'acquisition.json').write_text(json.dumps({
        'download_completed': True, 'scene': 'courtyard',
        'selection': '下载前固定选择，未查看GT/模型结果；室外建筑场景',
        'training': False, 'evaluation_completed': False,
        'note': '只有一个官方场景，不能称完整RMVD成绩；后续须沿用其配组、尺度与畸变协议。'
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if not __import__('os').environ.get('TMUX'):
        raise RuntimeError('数据下载须在tmux中运行')
    main()
