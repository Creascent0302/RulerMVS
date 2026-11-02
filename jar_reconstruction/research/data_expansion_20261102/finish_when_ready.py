"""只在三个扩容任务通过后发布训练配置；失败退出，不自动训练。"""
import json
from pathlib import Path
import subprocess
import sys
import time

from direct_http import ROOT
from expand import write_json


def main():
    started = time.monotonic()
    while time.monotonic() - started < 12 * 3600:
        states = {name: json.loads((ROOT / f'{name}_status.json').read_text())['state']
                  for name in ('hypersim', 'tartan', 'dynamic')}
        write_json(ROOT / 'finalizer_status.json', {'state': 'waiting', 'datasets': states})
        if 'failed' in states.values():
            raise RuntimeError(f'扩容失败，不发布READY：{states}')
        if all(state == 'ready' for state in states.values()):
            subprocess.run([sys.executable, str(Path(__file__).with_name('build_registry.py')),
                            '--finalize'], check=True)
            write_json(ROOT / 'finalizer_status.json', {'state': 'ready', 'training_started': False})
            return
        time.sleep(60)
    raise TimeoutError('12小时仍未完成，不发布READY')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write_json(ROOT / 'finalizer_status.json', {'state': 'failed', 'error': str(error)})
        raise
