"""冻结任务训练曝光和评测清单；单独标记历史访问过的物理场景。"""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from research.v7_zero_shot_20261103.prepare import DATA, SCENES, INDOOR, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = json.loads((DATA / 'results/base_config.json').read_text())
    old_two_view = Path('/home/liuzonghao/3Dscene/data/ETH3D_two_view_paper/prepared/test_all.jsonl')
    past_scenes = {json.loads(line)['sequence'].removeprefix('eth3d_') for line in old_two_view.read_text().splitlines()}
    audit = dict(task_train_domains={}, training_manifests={}, previously_analyzed_physical_scenes=sorted(past_scenes | {'courtyard'}),
                 fresh_physical_scenes=sorted(set(SCENES) - past_scenes - {'courtyard'}),
                 generic_pretraining_exposure='无法认证完整曝光；不作绝对从未见过承诺')
    for method in ['v6', 'v7', 'mvsanywhere']:
        train = json.loads((PROJECT / f'research/overnight_v6_v7_mvs_20261103/configs/{method}.json').read_text())
        audit['task_train_domains'][method] = [d['name'] for d in train['datasets']]
        assert all('eth3d' not in d['name'].lower() for d in train['datasets'])
        records = []
        for domain in train['datasets']:
            path = Path(domain['manifests']['train'])
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert all('eth3d' not in json.dumps(row).lower() for row in rows)
            records.append(dict(name=domain['name'], samples=len(rows), sha256=sha(path)))
        audit['training_manifests'][method] = records
    write(DATA / 'results/exposure_audit.json', audit)
    config['datasets'] = []
    selected = ['courtyard'] if args.smoke else SCENES
    for views in [2, 4, 8]:
        for scene in selected:
            manifest = DATA / 'manifests' / f'{scene}_{views}views.jsonl'
            assert manifest.exists(), manifest
            config['datasets'].append(dict(name=f'ETH3D_{scene}_{views}views', root=str(DATA),
                reference_index=0, num_views=views, geometry_units='meters', near=.1, far=100., depth_scale=1.,
                manifests={'test': str(manifest)}, scene=scene, category='indoor' if scene in INDOOR else 'outdoor',
                fresh_capture=scene != 'courtyard', fresh_physical_scene=scene in audit['fresh_physical_scenes']))
    config['shard_size'] = 8
    if args.smoke:
        config.update(output=str(DATA / 'results/smoke'), gpus=[0])
    destination = DATA / 'results' / ('smoke_config.json' if args.smoke else 'config.json')
    write(destination, config)
    plan = json.loads((DATA / 'results/plan.json').read_text())
    for name, method in config['methods'].items():
        assert sha(Path(method['weights'])) == plan['weights_sha256'][name]
    if not args.smoke:
        plan['fresh_physical_scenes'] = audit['fresh_physical_scenes']
        plan['fresh_capture_scenes'] = [s for s in SCENES if s != 'courtyard']
        plan['manifest_sha256'] = {d['name']: sha(Path(d['manifests']['test'])) for d in config['datasets']}
        plan['source_sha256'] = {str(p.relative_to(PROJECT)): sha(p) for p in Path(__file__).parent.glob('*.py')}
        write(DATA / 'results/plan.json', plan)
    print(destination)


if __name__ == '__main__':
    main()
