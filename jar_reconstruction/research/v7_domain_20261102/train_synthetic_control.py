"""仅删新增先验的V6减法对照：合成数据、联合保持与训练预算完全不变。"""

import argparse
from functools import partial
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.pretraining.comparison import digest, write_json
from research.v7_domain_20261102 import train
from research.v7_domain_20261102.retention_round2 import retention_penalty
from research.v7_domain_20261102.train_synthetic import SyntheticDataset, exposure_audit


def run(settings):
    if settings['variant'] != 'control' or settings['training_condition'] != 'joint_keep':
        raise ValueError('此入口仅用于V6 control＋合成＋联合保持减法对照')
    output = Path(settings['output'])
    sources = [Path(__file__)] + [Path(__file__).with_name(name + '.py') for name in
               ('train_synthetic', 'synthetic_pair_bilinear', 'synthetic_pair',
                'retention_round2', 'train', 'augmentation', 'model')]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    dataset = partial(SyntheticDataset, options=settings['synthetic_options'],
                      audit_folder=output / 'synthetic_exposures')
    with patch.object(train, 'MixedDataset', dataset), \
            patch.object(train, 'retention_penalty', retention_penalty):
        train.run(settings)
    saved = torch.load(output / 'final.pt', map_location='cpu', mmap=True, weights_only=False)
    assert not any(name.startswith(('decoder.context_prior.', 'decoder.center.')) for name in saved['model'])
    assert {group['lr'] for group in saved['optimizer']['param_groups']} == {settings['lr']}
    assert hashes == {str(path.relative_to(ROOT)): digest(path) for path in sources}
    for path in sources:
        (output / 'source_snapshot' / path.name).write_bytes(path.read_bytes())
    write_json(output / 'synthetic_control_audit.json', {
        'completed': True, 'test_accessed': False, 'architecture': saved['architecture'],
        'checkpoint_sha256': digest(output / 'final.pt'), 'source_sha256': hashes,
        'exposure': exposure_audit(output, settings), 'no_added_prior': True,
        'optimizer_lrs': sorted({group['lr'] for group in saved['optimizer']['param_groups']}),
        'scope': '同起点/数据/seed/预算/联合保持，仅删除新增prior；未修改原模块文件',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
