"""边准备边评测固定场景清单；每种视图数量独占一个工作进程。"""

import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from research.v7_zero_shot_20261103.prepare import DATA, SCENES, INDOOR, write, sha
from research.v7_zero_shot_20261103.evaluate import run_part
from research.retrain_comparison_20261103.models import Models
from tools.compare_public_pretrained import configure
from tools.smoke_outdoor_datasets import offline_only

TANKS = ['Tanks_Barn', 'Tanks_Church', 'Tanks_Courthouse', 'Tanks_Ignatius']
FRESH = {'kicker', 'meadow', 'office', 'pipes', 'relief', 'relief_2'}


def domain(scene, views):
    return dict(name=f'{scene}_{views}views', root=str(DATA), num_views=views,
                reference_index=0, geometry_units='meters', near=.1, far=100., depth_scale=1.,
                manifests={'test': str(DATA / 'manifests' / f'{scene}_{views}views.jsonl')},
                scene=scene, dataset='TanksAndTemples' if scene in TANKS else 'ETH3D_DSLR',
                category='other' if scene in TANKS else 'indoor' if scene in INDOOR else 'outdoor',
                fresh_capture=scene != 'courtyard', fresh_physical_scene=scene in FRESH or scene in TANKS)


def worker(config, gpu, pending, finished, source_sha):
    offline_only()
    device = configure(config, gpu)
    models = Models(config, device)
    while (item := pending.get()) is not None:
        try:
            config['datasets'] = [item]
            manifest = Path(item['manifests']['test'])
            length = len(manifest.read_text().splitlines())
            protocol_data = dict(base=config, manifest_sha256=sha(manifest), source_sha256=source_sha)
            protocol = hashlib.sha256(json.dumps(protocol_data, sort_keys=True).encode()).hexdigest()
            result = run_part(config, (0, 0, list(range(length))), models, {}, device,
                              DATA / 'results/accuracy/full_test', protocol)
            result.update(manifest_sha256=sha(manifest), domain_config=item, protocol=protocol)
        except Exception:
            result = dict(event='failed', domain=item['name'], error=traceback.format_exc())
            write(DATA / 'results/failures' / (item['name'] + '.json'), result)
        finished.put(result)


def main():
    config = json.loads((DATA / 'results/base_config.json').read_text())
    config['split'] = 'test'
    plan = json.loads((DATA / 'results/plan.json').read_text())
    for name, item in config['methods'].items():
        assert sha(Path(item['weights'])) == plan['weights_sha256'][name]
    sources = list(Path(__file__).parent.glob('*.py'))
    sources += [PROJECT / 'research/retrain_comparison_20261103/models.py']
    source_sha = {str(p.relative_to(PROJECT)): sha(p) for p in sources}
    write(DATA / 'results/execution_protocol.json', dict(config=config, source_sha256=source_sha,
        selected_scenes=SCENES + TANKS, fresh_physical_scenes=sorted(FRESH) + TANKS,
        counts=dict(ETH3D_DSLR=104, TanksAndTemples=69), primary_views=4, views=[2, 4, 8]))
    context = mp.get_context('spawn')
    finished = context.Queue()
    pending = {views: context.Queue() for views in [2, 4, 8]}
    jobs = [context.Process(target=worker, args=(config, gpu, pending[views], finished, source_sha), daemon=True)
            for gpu, views in enumerate([2, 4, 8])]
    for job in jobs:
        job.start()
    issued, received, records = set(), set(), []
    required = len(SCENES + TANKS) * 3
    while len(received) < required:
        for scene in SCENES + TANKS:
            ready = (DATA / 'results/tanks_plan.json') if scene in TANKS else (DATA / 'audit' / f'{scene}_ready.json')
            if not ready.exists() or (scene in TANKS and not json.loads(ready.read_text())['completed']):
                continue
            for views in pending:
                item = domain(scene, views)
                if item['name'] not in issued:
                    pending[views].put(item)
                    issued.add(item['name'])
        try:
            result = finished.get(timeout=20)
        except queue.Empty:
            for file in ['preparation.exit_code.txt', 'tanks_prepare.exit_code.txt']:
                path = DATA / file
                if path.exists() and path.read_text().strip() != '0':
                    raise RuntimeError('数据准备失败：' + str(path))
            if any(not job.is_alive() for job in jobs):
                raise RuntimeError('评测工作进程异常退出')
            continue
        records.append(result)
        received.add(result['domain'])
        write(DATA / 'results/progress.json', dict(completed=False, issued=len(issued),
            received=len(received), required=required, records=records))
        print(json.dumps(dict(event=result['event'], domain=result['domain'], received=len(received),
                              required=required)), flush=True)
        if result['event'] == 'failed':
            raise RuntimeError(result['error'])
    for q in pending.values():
        q.put(None)
    for job in jobs:
        job.join()
        assert job.exitcode == 0
    for path, digest in source_sha.items():
        assert sha(PROJECT / path) == digest
    for name, item in config['methods'].items():
        assert sha(Path(item['weights'])) == plan['weights_sha256'][name]
    write(DATA / 'results/progress.json', dict(completed=True, records=records, predictions=173 * 3 * 4))


if __name__ == '__main__':
    main()
