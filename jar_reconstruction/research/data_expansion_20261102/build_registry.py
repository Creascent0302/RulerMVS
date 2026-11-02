"""生成独立新版数据注册；全部达标前不生成可启动的训练配置。"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zlib

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'tools'))
from jar.data import build_dataset
from jar.pretraining.data import dataset_config
from research.v2_training_20261028.sampling import RebalancedFullPassSampler
from prepare_dynamicreplica import decode_depth
from direct_http import ROOT
from expand import manifest, read_rows, write_json
from audit_dynamic import audit as audit_dynamic

SOURCE = PROJECT / 'research/rulermvs_v6_20261101/configs/train_public.json'
SPLITS = PROJECT / 'official_methods/methods/mvsanywhere/upstream/data_splits'
MINIMUM = {'Hypersim_subset': 4536, 'TartanAir_forest': 9212, 'DynamicReplica_valid': 7010,
           'MatrixCity_street': 4011, 'MVS_Synth': 322, 'VKITTI2': 4016, 'BlendedMVS': 9711}
EXPANDED = {'Hypersim_subset': ('hypersim', 'Hypersim_expanded'),
            'TartanAir_forest': ('tartan', 'TartanAir_expanded'),
            'DynamicReplica_valid': ('dynamic', 'DynamicReplica_train')}


def excluded_scenes(name):
    paths = {'TartanAir_forest': 'tartanair/train_scans.txt', 'MVS_Synth': 'mvssynth/train_scans.txt',
             'BlendedMVS': 'blendedmvg/blendedmvg_train.txt',
             'DynamicReplica_valid': 'dynamic_replica/dynamic_replica_train.txt',
             'VKITTI2': 'vkitti/vkitti_train.txt'}
    if name in paths:
        return set((SPLITS / paths[name]).read_text().split())
    if name == 'Hypersim_subset':
        keys = json.loads((SPLITS / 'hypersim/bd_split/train_files_bd.json').read_text())
        return {k.split('/')[0] for k in keys}
    if name == 'MatrixCity_street':
        return {'big_city'}
    raise ValueError(name)


def scene_key(name, row):
    path = row['cameras'][0]['rgb'].split('/')
    if name == 'Hypersim_subset':
        return next(x for x in path if x.startswith('ai_'))
    if name == 'TartanAir_forest':
        index = path.index('raw')
        return '/'.join(path[index + 1:index + 4])
    if name == 'MVS_Synth':
        return path[path.index('GTAV_540') + 1]
    if name == 'BlendedMVS':
        return row['source_scene']
    if name == 'DynamicReplica_valid':
        return row['source_sequence']
    if name == 'VKITTI2':
        return row['sequence'].split('VKITTI2/')[1]
    return next(x for x in path if x in ['small_city', 'big_city'])


def unseen_tests(config, updated_domains=None):
    tests, audit = [], []
    updated = {d['name']: d for d in (updated_domains or [])}
    for domain in config['datasets']:
        name = domain['name']
        original = Path(domain['manifests']['val']).with_name('test.jsonl')
        rows = read_rows(original, domain['root'])
        forbidden = excluded_scenes(name)
        selected = [row for row in rows if scene_key(name, row) not in forbidden]
        entry = dict(name=name, original=len(rows), retained=len(selected),
                     excluded_scenes=sorted({scene_key(name, r) for r in rows} & forbidden),
                     rule='整个场景/轨迹排除，不只是删除官方tuple命中的参考帧')
        audit.append(entry)
        if selected:
            output = ROOT / 'unseen_tests' / (name + '.jsonl')
            manifest(output, selected)
            item = copy.deepcopy(updated.get(name, domain))
            item['manifests'] = {'test': str(output)}
            tests.append(item)
    result = dict(datasets=tests, audit=audit, image_size=[256, 448],
                  matching_image_size=[512, 896], input_mode='equal_pixels',
                  level='同数据集家族的场景留出，不能叫全新数据域',
                  exposure_definition='未在官方公开任务训练扫描名单中发现，不保证通用预训练或资产无曝光',
                  historical_tests_kept=True, rmvd='courtyard另列，未混入训练',
                  warning='Blended官方pair文件不完整；这些旧测试已被研究查看，不能称新的盲测')
    write_json(ROOT / 'heldout_scene_registry.json', result)
    base = PROJECT / 'runs/active/rulermvs_v6_20261101/full_training'
    eth = json.loads((base / 'paper_eth3d/evaluation.json').read_text())
    carla = json.loads((base / 'evaluation_recovery/configs/carla_new_main_test.json').read_text())
    external = []
    evaluation_configs = {}
    for group, source in [('ETH3D', eth), ('CARLA', carla)]:
        for entry in source['datasets']:
            item = copy.deepcopy(entry)
            item['manifests'] = {'test': entry['manifests']['test']}
            item['groups'] = len(read_rows(item['manifests']['test']))
            item['evaluation_scope'] = ('真实未见域；all/nocc为同一批图像的两种GT掩码' if group == 'ETH3D'
                else '对Hero为未见域；自研若加载CARLA适配权重则是目标域留出，非双方零样本')
            external.append(item)
        evaluation = copy.deepcopy(source)
        evaluation['methods'] = {key: value for key, value in source['methods'].items()
                                 if key in ('v6_best', 'mvsanywhere_refined')}
        if len(evaluation['methods']) != 2:
            raise ValueError('未找到冻结V6和Hero细化对照')
        for domain in evaluation['datasets']:
            domain['manifests'] = {'test': domain['manifests']['test']}
        evaluation['output'] = str(ROOT / 'evaluation' / group)
        evaluation['protocol_notes'].append('本轮外测注册：仍加载保留的V6权重；扩容后未训练。')
        destination = ROOT / 'evaluation_configs' / (group + '.json')
        write_json(destination, evaluation)
        evaluation_configs[group] = str(destination)
    write_json(ROOT / 'unseen_test_registry.json', dict(datasets=external,
        evaluation_configs=evaluation_configs,
        level='MVSAnywhere公开任务训练未包含的数据域', image_size=[256, 448],
        matching_image_size=[512, 896], input_mode='equal_pixels',
        source='官方README与八域任务训练配置；不排除通用预训练资产曝光',
        rmvd=dict(root=str(ROOT.parent / 'ETH3D_rmvd_subset'), scene='courtyard',
                  views=[4, 8], groups_per_setting=8, native_depth_evaluation=True,
                  command='official_methods/.venv/bin/python research/data_expansion_20261102/evaluate_rmvd_subset.py'),
        excluded_from_primary=['VKITTI2', 'TartanAir', 'Hypersim', 'MatrixCity', 'MVS-Synth',
                               'BlendedMVS', 'DynamicReplica'],
        supplemental_scene_tests=str(ROOT / 'heldout_scene_registry.json'),
        fairness='Hero权重固定；如果基线之后也使用CARLA微调，必须改称同数据监督比较。'))
    return result


def range_from_train(domain):
    rows = read_rows(domain['manifests']['train'])
    values = []
    for index in np.linspace(0, len(rows) - 1, min(512, len(rows)), dtype=int):
        for camera in rows[index]['cameras']:
            label = camera['targets']
            path = Path(domain['root']) / label['depth']
            if label['depth_encoding'] == 'dynamic_replica_float16_png':
                depth = decode_depth(path)[::16, ::16].astype('float64')
            else:
                depth = np.load(path, mmap_mode='r')[::16, ::16].astype('float64')
            depth *= label.get('depth_scale', 1.)
            valid = np.isfinite(depth) & (depth > 0) & (depth < label.get('max_valid_depth', 1000.))
            if label.get('depth_valid'):
                valid &= np.load(Path(domain['root']) / label['depth_valid'], mmap_mode='r')[::16, ::16].astype(bool)
            values.append(depth[valid])
    values = np.concatenate(values)
    low, median, high = np.quantile(values, [.0001, .5, .9995])
    previous = [domain['near'], domain['far']]
    # 新室内场景增多不能收缩旧TRAIN已支持的深度；只用旧/新TRAIN信息取并集。
    domain.update(near=min(previous[0], float(low * .5)),
                  far=max(previous[1], float(high * 1.25)), depth_scale=float(median / 10.))
    return dict(name=domain['name'], sampled_groups=min(512, len(rows)),
                quantiles=[float(low), float(median), float(high)], labels_used='TRAIN only',
                previous_train_bounds=previous, configured_bounds=[domain['near'], domain['far']],
                range_rule='旧TRAIN范围与扩容TRAIN分位范围取并集，不按TEST调节')


def verify_downloads():
    """最终核对已落盘成员CRC，覆盖中断续传和镜像/原站切换前留下的文件。"""
    files, size = 0, 0
    for marker in sorted((ROOT / 'completed_archives').glob('*.json')):
        if marker.name.startswith('tartan_'):
            raw = ROOT.parent / 'TartanAir_expanded/raw'
        elif marker.name.startswith('dynamic_'):
            raw = ROOT.parent / 'DynamicReplica_train/raw/train'
        else:
            raw = ROOT.parent / 'Hypersim_expanded/raw'
        for item in json.loads(marker.read_text())['members']:
            path = raw / item['name']
            crc = 0
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(2**20), b''):
                    crc = zlib.crc32(block, crc)
            if path.stat().st_size != item['bytes'] or crc != item['crc']:
                raise ValueError(f'落盘CRC不符：{path}')
            files += 1
            size += item['bytes']
    write_json(ROOT / 'integrity_audit.json', dict(files=files, bytes=size, member_crc_verified=True,
                                                 whole_archive_sha_verified=False))


def finalize(config):
    for task, directory in EXPANDED.values():
        status = json.loads((ROOT / (task + '_status.json')).read_text())
        if status['state'] != 'ready':
            raise RuntimeError(f'{task}尚未ready，不发布训练配置')
    verify_downloads()
    train = copy.deepcopy(config)
    train['output'] = str(ROOT / 'future_training')
    train['initial_model_checkpoint'] = str(PROJECT / 'runs/active/rulermvs_v6_20261101/full_training/public/best.pt')
    train['protocol_notes'] = ['扩充数据配方；尚未启动新的模型训练。',
        '新增Hypersim和Dynamic样本，降低旧配方的机械重放倍率。',
        'TEST独立见unseen_test_registry.json；历史重合TEST不进入新主表。',
        '从保留的正式V6 public best继续适配；不加载MVSAnywhere任务权重。',
        '数据配比是待验证配方；不是已证明提升，未启动本轮八卡训练。']
    weights = {'Hypersim_subset': 1.5, 'DynamicReplica_valid': 1.5,
               'BlendedMVS': 1.6, 'VKITTI2': 1.5}
    validation_path = PROJECT / ('runs/active/rulermvs_v6_20261101/full_training/'
                                 'evaluation_recovery/public_main/full_val/summary.json')
    validation = json.loads(validation_path.read_text())
    replay_basis = {}
    for name, values in validation['datasets'].items():
        ours = values['methods']['v6_best']['depth']['full']['MAE']
        baseline = values['methods']['mvsanywhere_refined']['depth']['full']['MAE']
        replay_basis[name] = dict(v6_mae=ours, baseline_mae=baseline,
                                  relative_gap=ours / baseline - 1.)
    write_json(ROOT / 'replay_basis.json', dict(source=str(validation_path),
        source_sha256=hashlib.sha256(validation_path.read_bytes()).hexdigest(), split='val',
        domains=replay_basis, multipliers=weights,
        note='先保全覆盖，再有限重放弱域；倍率为待训练验证的保守方案，不是实验最优值。'))
    statistics, checks = [], []
    torch.set_num_threads(2)
    for domain in train['datasets']:
        name = domain['name']
        if name in EXPANDED:
            root = ROOT.parent / EXPANDED[name][1]
            domain['root'] = str(root)
            domain['display_name'] = EXPANDED[name][1]
            domain['manifests'] = {s: str(root / f'manifests/{s}.jsonl') for s in ['train', 'val']}
        path = Path(domain['manifests']['train'])
        rows = read_rows(path)
        if len(rows) < MINIMUM[name]:
            raise ValueError(f'{name}未满足官方1/10最低数')
        train_ids = {r['sequence'] for r in rows}
        val_rows = read_rows(domain['manifests']['val'])
        if name != 'MatrixCity_street' and train_ids & {r['sequence'] for r in val_rows}:
            raise ValueError(f'{name}训练验证序列交叉')
        domain['train_manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        domain['train_multiplier'] = weights.get(name, 1.)
        if name in EXPANDED:
            statistics.append(range_from_train(domain))
        # 首中末真实样本，验证监督/无监督输入逐项相同，不依赖模型。
        cfg = dataset_config(domain, train)
        ds = build_dataset(cfg, 'train')
        infer = build_dataset(cfg, 'train', load_targets=False)
        for index in sorted({0, len(ds) // 2, len(ds) - 1}):
            sample, no_labels = ds[index], infer[index]
            assert not no_labels['targets']
            for key, value in sample['inputs'].items():
                if torch.is_tensor(value):
                    assert torch.equal(value, no_labels['inputs'][key]), (name, key)
                    if value.is_floating_point():
                        assert torch.isfinite(value).all()
            assert sample['targets']['depth_valid'].any()
        checks.append(dict(name=name, train=len(rows), val=len(val_rows), minimum=MINIMUM[name],
                           multiplier=domain['train_multiplier'], loader_checked=3,
                           expected_exposure_before_ddp_padding=len(rows) * domain['train_multiplier']))
    write_json(ROOT / 'train_range_statistics.json', statistics)
    write_json(ROOT / 'training_data_audit.json', checks)
    audit_dynamic(ROOT.parent / 'DynamicReplica_train')
    sampler = RebalancedFullPassSampler([item['train'] for item in checks],
        epoch=0, rank=0, world=8, batch_size=train['train']['batch_size'],
        accumulation=train['train']['accumulation'], seed=train['seed'],
        domain_multipliers=[item['multiplier'] for item in checks], domain_schedule='balanced_prefix')
    for index, item in enumerate(checks):
        seen = {sample for domain, batch in sampler.updates if domain == index for sample, _ in batch}
        if len(seen) != item['train']:
            raise ValueError(f"{item['name']}八卡采样计划未覆盖完整TRAIN")
    old_sampler = RebalancedFullPassSampler(
        [len(read_rows(d['manifests']['train'])) for d in config['datasets']],
        epoch=0, rank=0, world=8, batch_size=train['train']['batch_size'],
        accumulation=train['train']['accumulation'], seed=train['seed'],
        domain_multipliers=[d['train_multiplier'] for d in config['datasets']],
        domain_schedule='balanced_prefix')
    fractions = {}
    for i, item in enumerate(checks):
        before = old_sampler.domain_counts[i]['total_exposures'] / sum(
            v['total_exposures'] for v in old_sampler.domain_counts.values())
        after = sampler.domain_counts[i]['total_exposures'] / sum(
            v['total_exposures'] for v in sampler.domain_counts.values())
        fractions[item['name']] = dict(before=before, after=after)
        if item['name'] in ('Hypersim_subset', 'DynamicReplica_valid', 'BlendedMVS') and after <= before:
            raise ValueError('旧VAL弱域的实际曝光份额没有提高')
    write_json(ROOT / 'sampler_audit.json', dict(world_size=8,
        updates_per_epoch=len(sampler.updates), training_executed=False,
        exposure_fraction_comparison=fractions,
        domains={checks[i]['name']: counts for i, counts in sampler.domain_counts.items()}))
    network = [json.loads(line) for line in (ROOT / 'network.jsonl').read_text().splitlines()]
    if not all(item['local_ip'] == '172.16.2.4' and item['interface'] == 'eno5'
               and item['proxy'] is False for item in network):
        raise ValueError('下载连接记录不满足服务器直连约束')
    free_gib = shutil.disk_usage(ROOT).free / 2**30
    if free_gib < 200:
        raise RuntimeError('磁盘余量低于200GiB，不发布READY')
    write_json(ROOT / 'network_audit.json', dict(requests=len(network),
        local_ip='172.16.2.4', interface='eno5', proxy=False,
        remote_ips=sorted({item['remote_ip'] for item in network}),
        completed_response_bytes=sum(item['bytes'] for item in network),
        note='成功响应字节合计，包含重复区块；不包括失败请求和重试的中间传输',
        free_disk_gib=free_gib))
    write_json(ROOT / 'train_v6_expanded.json', train)
    # 补充域的搜索范围随新TRAIN更新，测试帧/标签与曝光过滤规则不变。
    unseen_tests(config, updated_domains=train['datasets'])
    write_json(ROOT / 'READY.json', {'ready': True, 'all_seven_domains_at_least_tenth': True,
                                   'training_started': False, 'datasets': checks})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    config = json.loads(SOURCE.read_text())
    published = ROOT / 'train_v6_expanded.json'
    updated = json.loads(published.read_text())['datasets'] if published.exists() else None
    test = unseen_tests(config, updated_domains=updated)
    print(json.dumps(test['audit'], ensure_ascii=False, indent=2))
    if args.finalize:
        finalize(config)
