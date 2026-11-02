"""冻结编码器的真实RGB单遍训练；增加独立样本，不保存特征缓存。

配置全部由JSON提供。正式协议：每域最多512个不同TRAIN、同域batch2、
epoch0原有参考轮换/亮度/缺源增强；每448更新及末尾复用固定8DEV。
smoke=true仅执行预登记调度前至多20步，不能当作完整训练结果。
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.models.position_cache import install_position_cache
from jar.pretraining.comparison import digest, write_json
from jar.pretraining.data import MixedDataset, collate_mixed
from jar.pretraining.losses import depth_objective
from jar.pretraining.model import PretrainingModel
from jar.runtime import move
from research.ruler_joint_20261027.model import make_decoder
from research.ruler_joint_20261027.run import compare, evaluate, metric_auxiliary
from tools.cache_architecture_screen import encoder_digest
from tools.refine_public_candidates import save_record
from tools.screen_public_architecture import balanced_sample


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def select_training(config, plan):
    """只读TRAIN清单元数据，保留原始选中行及其索引/摘要。"""
    seed = plan.get('seed', 20261027)
    requested = plan.get('train_samples_per_domain', 512)
    if requested != 512 and not plan.get('smoke', False):
        raise ValueError('正式协议每域最多512组；缩小样本只能用于显式smoke')
    if requested <= 0:
        raise ValueError('train_samples_per_domain须为正数')
    selections, raw_selected = {}, {}
    for domain_index, domain in enumerate(config['datasets']):
        manifest = Path(domain['manifests']['train'])
        manifest_hash = digest(manifest)
        if manifest_hash != domain['train_manifest_sha256']:
            raise ValueError(f"TRAIN清单改变：{domain['name']}")
        lines = [line for line in manifest.read_text().splitlines() if line.strip()]
        rows = [json.loads(line) for line in lines]
        groups = {}
        for index, row in enumerate(rows):
            sequence = str(row.get('sequence', row.get('source_sequence', 'single_sequence')))
            groups.setdefault(sequence, []).append(index)
        selected = balanced_sample(groups, min(requested, len(rows)), seed + domain_index)
        selected_text = ''.join(lines[index] + '\n' for index in selected)
        records = []
        for index in selected:
            row = rows[index]
            records.append({
                'index': index,
                'sequence': row.get('sequence', row.get('source_sequence')),
                'frame_id': row.get('frame_id'),
                'row_sha256': hashlib.sha256(lines[index].encode()).hexdigest(),
            })
        selections[domain['name']] = {
            'manifest': str(manifest), 'manifest_sha256': manifest_hash,
            'manifest_frames': len(rows), 'sequence_count': len(groups),
            'selected_count': len(selected), 'indices': selected, 'records': records,
            'selected_manifest_sha256': hashlib.sha256(selected_text.encode()).hexdigest(),
        }
        raw_selected[domain['name']] = selected_text
    return selections, raw_selected


def make_schedule(domains, selections, plan):
    """单遍、不补齐、不重复；不足两组的最后一个batch保留真实大小。"""
    if plan.get('batch_size', 2) != 2:
        raise ValueError('本轮固定同域batch_size=2')
    rng = random.Random(plan.get('seed', 20261027))
    batches = []
    for domain_index, domain in enumerate(domains):
        indices = list(selections[domain['name']]['indices'])
        rng.shuffle(indices)
        batches.extend([(domain_index, index, 0) for index in indices[start:start + 2]]
                       for start in range(0, len(indices), 2))
    rng.shuffle(batches)
    full_steps = len(batches)
    if plan.get('smoke', False):
        smoke_steps = int(plan.get('smoke_steps', 20))
        if not 1 <= smoke_steps <= 20:
            raise ValueError('smoke_steps须在1至20之间')
        batches = batches[:smoke_steps]
    elif 'steps' in plan:
        raise ValueError('正式更新数由唯一TRAIN样本单遍确定，不另设steps')
    return batches, full_steps


def trainable_names(plan):
    names = list(plan.get('trainable_modules',
                          ['relation'] if plan.get('relation', False) else ['fine_offset']))
    allowed = {'relation', 'fine_offset', 'fine_score', 'fine_inherit', 'upsample_mask'}
    if not names or len(set(names)) != len(names) or not set(names) <= allowed:
        raise ValueError('只允许显式训练细层模块，编码器和粗层保持冻结')
    if 'relation' in names and not plan.get('relation', False):
        raise ValueError('训练relation需要relation=true')
    return names


def run(plan):
    torch.set_num_threads(plan.get('threads', 2))
    seed = plan.get('seed', 20261027)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(plan['device'])
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    evaluate_every = int(plan.get('evaluate_every', 448))
    if evaluate_every <= 0:
        raise ValueError('evaluate_every须为正数')
    if evaluate_every != 448 and not plan.get('smoke', False):
        raise ValueError('正式协议每448更新及末尾评价')
    output = project_path(plan['output'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('研究目录非空，禁止覆盖')
    names = trainable_names(plan)
    checkpoint_path = project_path(plan['checkpoint'])
    index_path = project_path(plan['cache']) / 'index.json'
    index = json.loads(index_path.read_text())
    for name, paths in index['splits']['development'].items():
        index['splits']['development'][name] = [str(project_path(path)) for path in paths]
    if digest(checkpoint_path) != index['checkpoint_sha256']:
        raise ValueError('DEV缓存必须来自同一冻结编码器checkpoint')
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    encoder_hash = encoder_digest(saved['model'])
    if encoder_hash != index['encoder_sha256']:
        raise ValueError('DEV缓存编码器摘要不匹配')
    config = copy.deepcopy(saved['config'])
    config.update(seed=seed, augmentation=True)
    if config['datasets'] != index['domains']:
        raise ValueError('训练域、范围及单位必须与DEV缓存一致')
    selections, selected_lines = select_training(config, plan)
    schedule, full_steps = make_schedule(config['datasets'], selections, plan)
    schedule_payload = json.dumps(schedule, separators=(',', ':')).encode()
    schedule_hash = hashlib.sha256(schedule_payload).hexdigest()
    dataset = MixedDataset(config, 'train')
    for domain, source in zip(dataset.domains, dataset.datasets):
        if len(source) != selections[domain['name']]['manifest_frames']:
            raise ValueError('Dataset行号与原始TRAIN清单不一致')
    loader = DataLoader(dataset, batch_sampler=schedule, collate_fn=collate_mixed,
                        num_workers=plan.get('workers', 4), pin_memory=device.type == 'cuda',
                        generator=torch.Generator().manual_seed(seed))

    # 必须先冻结并eval整个原模型，再只开启新细头；不调用整体model.train()。
    model = PretrainingModel(saved['config'], initialize=False)
    model.load_state_dict(saved['model'], strict=True)
    model.requires_grad_(False).eval().to(device)
    install_position_cache(model.foundation.model)
    decoder, settings = make_decoder(saved, plan, device)
    model.decoder = decoder
    decoder.requires_grad_(False)
    for name in names:
        getattr(decoder, name).requires_grad_(True)
    frozen = {name: value.detach().cpu().clone() for name, value in decoder.named_parameters()
              if not value.requires_grad}
    groups = [{'params': list(getattr(decoder, name).parameters()),
               'lr': plan.get('relation_lr', plan['lr']) if name == 'relation' else plan['lr']}
              for name in names]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'plan.json', plan)
    write_json(output / 'selection.json', selections)
    write_json(output / 'schedule.json', {'sha256': schedule_hash, 'batches': schedule})
    for name, lines in selected_lines.items():
        destination = output / 'selected_train' / (name + '.jsonl')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(lines)
    sources = [Path(__file__), Path(__file__).with_name('model.py'),
               Path(__file__).with_name('run.py'), ROOT / 'jar/pretraining/data.py',
               ROOT / 'jar/pretraining/model.py', ROOT / 'jar/pretraining/losses.py']
    audit = {
        'settings': settings, 'seed': seed, 'precision': 'FP32 TF32 off',
        'checkpoint_sha256': index['checkpoint_sha256'], 'encoder_sha256': encoder_hash,
        'cache_index_sha256': digest(index_path), 'schedule_sha256': schedule_hash,
        'sources': {str(path.relative_to(ROOT)): digest(path) for path in sources},
        'trainable_modules': names,
        'trainable_parameters': sum(p.numel() for p in decoder.parameters() if p.requires_grad),
        'planned_unique_samples': sum(len(value['indices']) for value in selections.values()),
        'executed_schedule_samples': sum(len(batch) for batch in schedule),
        'steps': len(schedule), 'full_pass_steps': full_steps, 'epoch': 0,
        'smoke': bool(plan.get('smoke', False)), 'test_accessed': False,
        'encoder_eval_and_no_grad': True, 'coarse_parameters_frozen': True,
        'feature_cache_written': False,
        'augmentation': 'existing MixedDataset epoch0: labelled reference rotation, brightness, missing source',
        'selection_uses_gt': False,
        'reference_selection': 'existing training labelled-view availability only; no candidate/oracle selection',
        'gate': 'reuse compare(): full per-domain metric<=+2%, far25<=+3%, mean normalized risk<0',
    }
    write_json(output / 'audit.json', audit)
    initial = evaluate(decoder, index, device)
    reference = json.loads(project_path(plan['baseline_metrics']).read_text()) if plan.get('baseline_metrics') else initial
    initial_comparison = compare(initial, reference)
    write_json(output / 'step_0000.json', initial)
    write_json(output / 'comparison_0000.json', initial_comparison)
    best = min(0., initial_comparison['normalized_risk_change']) if initial_comparison['passes_development_gate'] else 0.
    best_step = 0

    def save(filename, step):
        save_record({'model': decoder.state_dict(), 'settings': settings, 'plan': plan,
                     'step': step, 'optimizer': optimizer.state_dict(),
                     'encoder_checkpoint': str(checkpoint_path), 'encoder_sha256': encoder_hash,
                     'schedule_sha256': schedule_hash}, output / filename)

    save('best.pt', 0)
    started = time.monotonic()
    exposures = 0
    with (output / 'train.jsonl').open('w') as stream:
        for step, batch in enumerate(loader, 1):
            keys = schedule[step - 1]
            domain = dataset.domains[keys[0][0]]
            batch = move(batch, device, non_blocking=True)
            assert not model.foundation.training and not model.matching.training
            with torch.no_grad():
                work = model.encode(batch['inputs'], domain, reference=0)
            scale = domain['depth_scale']
            decoder.set_range(domain['near'] / scale, domain['far'] / scale)
            decoder.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = decoder(work)
            prediction['unit_scale'] = scale
            losses = depth_objective(prediction, batch['inputs'], batch['targets'], domain, {}, 0.)
            metric = metric_auxiliary(prediction, batch)
            loss = losses['total'] + plan.get('metric_weight', 0.) * metric
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            exposures += len(keys)
            record = {'step': step, 'dataset': domain['name'], 'indices': [key[1] for key in keys],
                      'exposures': exposures, 'metadata': batch['metadata'],
                      'view_mask': batch['inputs']['view_mask'].tolist(),
                      'loss': float(loss.detach()), 'metric_auxiliary': float(metric.detach()),
                      'grad_norm': float(gradient), 'elapsed_s': time.monotonic() - started}
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            if step % 20 == 0 or step == len(schedule):
                stream.flush()
                print(json.dumps(record, ensure_ascii=False), flush=True)
            del prediction, work, losses, loss, metric, batch
            if step % evaluate_every == 0 or step == len(schedule):
                result = evaluate(decoder, index, device)
                comparison = compare(result, reference)
                write_json(output / f'step_{step:04d}.json', result)
                write_json(output / f'comparison_{step:04d}.json', comparison)
                if comparison['passes_development_gate'] and comparison['normalized_risk_change'] < best:
                    best, best_step = comparison['normalized_risk_change'], step
                    save('best.pt', step)
                save('last.pt', step)
                if plan.get('save_evaluation_checkpoints', False):
                    save(f'step_{step:04d}.pt', step)
                print(json.dumps({'event': 'validation', 'step': step, 'best_step': best_step,
                                  'normalized_risk_change': comparison['normalized_risk_change'],
                                  'gate_failures': comparison['failures']}), flush=True)
    for name, value in decoder.named_parameters():
        if name in frozen:
            assert torch.equal(value.detach().cpu(), frozen[name]), f'冻结参数改变：{name}'
    final_encoder = {name: value.detach().cpu() for name, value in model.state_dict().items()
                     if name.startswith(('foundation.', 'matching.'))}
    assert encoder_digest(final_encoder) == encoder_hash, '冻结编码器参数/状态改变'
    write_json(output / 'complete.json', {
        'steps': len(schedule), 'exposures': exposures, 'best_step': best_step,
        'best_normalized_risk_change': best, 'elapsed_s': time.monotonic() - started,
        'smoke': bool(plan.get('smoke', False)), 'full_pass_completed': len(schedule) == full_steps,
        'frozen_encoder_and_coarse_verified': True, 'test_accessed': False,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
