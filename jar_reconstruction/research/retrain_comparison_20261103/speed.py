"""复用既有逐域同卡交替计时；与精度测试使用相同预测执行路径。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.paper_retest_20261102 import speed as previous
from research.retrain_comparison_20261103.models import Models
from tools.smoke_outdoor_datasets import offline_only

previous.Models = Models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--domain')
    parser.add_argument('--four', action='store_true')
    args = parser.parse_args()
    offline_only()
    config = json.loads(Path(args.config).read_text())
    output = Path(config['output']).parent / 'speed'
    if args.domain:
        domain = next(d for d in config['datasets'] if d['name'] == args.domain)
        target = output / (args.domain + ('_four' if args.four else '') + '.json')
        if target.exists() and json.loads(target.read_text()).get('completed'):
            return
        previous.benchmark(config, domain, config['timing'], target, args.four)
        return
    active = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('其他GPU任务仍在运行，不能开始独占计时：' + active)
    for domain in config['datasets']:
        command = [sys.executable, __file__, '--config', args.config, '--domain', domain['name']]
        subprocess.run(command, check=True, cwd=ROOT)
        if domain['name'] == 'CARLA_new_upper':
            subprocess.run(command + ['--four'], check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
