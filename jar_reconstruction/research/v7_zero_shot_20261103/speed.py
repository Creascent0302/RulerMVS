"""精度任务结束后，用17个场景各一组无标签输入独占计时。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from research.v7_zero_shot_20261103.prepare import DATA, write
from research.v7_zero_shot_20261103.run import SCENES, TANKS
from research.paper_retest_20261102 import speed as timer
from research.retrain_comparison_20261103.models import Models
from tools.smoke_outdoor_datasets import offline_only

timer.Models = Models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--views', type=int, choices=[2, 4, 8])
    args = parser.parse_args()
    offline_only()
    if args.views is None:
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if active:
            raise RuntimeError('计时前须等待其他GPU任务退出：' + active)
        for views in [2, 4, 8]:
            subprocess.run([sys.executable, __file__, '--views', str(views)], cwd=PROJECT, check=True)
        return
    views = args.views
    config = json.loads((DATA / 'results/base_config.json').read_text())
    config['timing']['test_samples_per_domain'] = 17
    rows = [(DATA / 'manifests' / f'{scene}_{views}views.jsonl').read_text().splitlines()[0] for scene in SCENES + TANKS]
    manifest = DATA / 'manifests' / f'timing_{views}views.jsonl'
    manifest.write_text('\n'.join(rows) + '\n')
    domain = dict(name=f'zero_shot_{views}views', root=str(DATA), num_views=views,
                  reference_index=0, geometry_units='meters', near=.1, far=100., depth_scale=1.,
                  manifests={'test': str(manifest)})
    target = DATA / 'results/speed' / f'{views}views.json'
    timer.benchmark(config, domain, config['timing'], target, False)


if __name__ == '__main__':
    main()
