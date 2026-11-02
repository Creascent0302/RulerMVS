"""固定8MiB比较官方原站与镜像的服务器直连；不修改正在运行的任务。"""
import hashlib
import json
import os

from direct_http import ROOT, fetch

if __name__ == '__main__':
    if not os.environ.get('TMUX'):
        raise RuntimeError('下载必须在tmux中执行')
    records = []
    for url in ('https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_001.zip',
                'https://hf-mirror.com/datasets/geyongtao/dynamic_replica/resolve/main/train_001.zip'):
        path = ROOT / 'temporary/probe_range.bin'
        try:
            record = fetch(url, path, (0, 8 * 2**20 - 1), 8 * 2**20, timeout=60)
            record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            records.append(record)
        except RuntimeError as exc:
            records.append({'url': url, 'error': str(exc)})
        path.unlink(missing_ok=True)
    (ROOT / 'direct_probe.json').write_text(json.dumps(records, indent=2))
