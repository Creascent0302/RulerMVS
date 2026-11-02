"""GT事后供给诊断：中心四组winner＋八邻格winner，共12个提案。"""

import csv
import json
from pathlib import Path
import shutil
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jar.pretraining.comparison import digest, write_json
from jar.pretraining.model import CalibratedDecoder
from jar.runtime import move
from research.near_mid_20261027.diagnose import distance_masks, local_plane_groups
from tools.probe_public_stages import edge_groups, error_statistics, sum_records, summarize


def proposal_oracles(output, scale, truth, valid):
    fine = output['fine']
    depth = fine['depth'] * scale
    batch, _, height, width = depth.shape
    grouped_scores = fine['scores'].reshape(batch, 4, 3, height, width)
    grouped_depth = fine['candidate_depth'].reshape_as(grouped_scores) * scale
    group_indices = grouped_scores.argmax(2, keepdim=True)
    group_depth = grouped_depth.gather(2, group_indices)[:, :, 0]
    group_valid = fine['valid'].reshape_as(grouped_scores).gather(2, group_indices)[:, :, 0]
    center_global = group_depth.gather(1, grouped_scores.amax(2).argmax(1, keepdim=True))
    torch.testing.assert_close(center_global, depth, rtol=1e-6, atol=1e-7)
    neighbors = F.unfold(depth, 3, padding=1).reshape(batch, 9, height, width)
    neighbor_valid = F.unfold(fine['ref_valid'].float(), 3, padding=1).reshape_as(neighbors) > .5
    adjacent = [0, 1, 2, 3, 5, 6, 7, 8]
    proposed = torch.cat((group_depth, neighbors[:, adjacent]), dim=1)
    proposed_valid = torch.cat((group_valid, neighbor_valid[:, adjacent]), dim=1)

    def expand(value):
        return value.repeat_interleave(4, -2).repeat_interleave(4, -1)

    def distance_oracle(candidates, legal):
        difference = (expand(candidates) - truth).abs().masked_fill(~expand(legal), torch.inf)
        error, index = difference.min(1, keepdim=True)
        has_candidate = expand(legal.any(1, keepdim=True))
        error = torch.where(has_candidate, error, (output['depth'] * scale - truth).abs())
        return error, index, has_candidate

    error9, _, has9 = distance_oracle(neighbors, neighbor_valid)
    error12, chosen12, has12 = distance_oracle(proposed, proposed_valid)
    assert torch.equal(has9, has12)
    tolerance = 1e-6 * (1 + truth.abs())
    if ((error12 > error9 + tolerance) & valid & has9).any():
        raise AssertionError('12提案应包含原9提案能力，离散oracle不能更差')
    low = expand(neighbors.masked_fill(~neighbor_valid, torch.inf).amin(1, keepdim=True))
    high = expand(neighbors.masked_fill(~neighbor_valid, -torch.inf).amax(1, keepdim=True))
    convex = torch.maximum(low - truth, truth - high).clamp_min(0)
    convex = torch.where(has9, convex, (output['depth'] * scale - truth).abs())
    return {
        'original_final': output['depth'] * scale,
        'nearest_fine_winner': expand(depth),
        'oracle_9_discrete': truth + error9,
        'oracle_12_discrete': truth + error12,
        'oracle_9_convex': truth + convex,
    }, {
        'improved': (error12 + tolerance < error9) & has12,
        'new_center_winner_chosen': (chosen12 < 4) & (error12 + tolerance < error9) & has12,
        'no_candidate': ~has12,
        'error_reduction': error9 - error12,
        'rescue_outside_original_hull': (convex > tolerance) & (error12 + tolerance < convex) & has12,
    }


@torch.inference_mode()
def run():
    torch.set_num_threads(2)
    torch.manual_seed(20261027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:1')
    torch.cuda.set_device(device)
    out = ROOT / 'runs/active/near_mid_20261027/proposal_supply'
    if out.exists():
        raise FileExistsError(out)
    if shutil.disk_usage(out.parent.resolve()).free < 204 * 2**30:
        raise RuntimeError('数据盘空闲不足204GiB')
    out.mkdir(parents=True)
    checkpoint = ROOT / 'runs/active/public_refinement_20261025/recommended/pretrained.pt'
    index_path = ROOT / 'runs/active/architecture_screen_20261026/features/index.json'
    index = json.loads(index_path.read_text())
    assert digest(checkpoint) == index['checkpoint_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    decoder = CalibratedDecoder(saved['config']['model'])
    decoder.load_state_dict({k.removeprefix('decoder.'): v for k, v in saved['model'].items() if k.startswith('decoder.')}, strict=True)
    decoder.to(device).eval().requires_grad_(False)
    assert decoder.fine_readout == 'winner'
    write_json(out / 'protocol.json', {
        'checkpoint_sha256': digest(checkpoint), 'cache_index_sha256': digest(index_path),
        'script_sha256': digest(__file__), 'test_accessed': False, 'training': False,
        'frames': 56, 'device': 'cuda:1', 'precision': 'FP32 TF32 off',
        'proposal': 'center four candidate groups: argmax over three scores per group, continuous depth; plus eight neighbors global winners. Center global winner not added twice, equal-depth proposals from different groups are not deduplicated.',
        'oracle': 'GT only chooses hindsight nearest depth among supplied hypotheses, never enters prediction or proposal selection. Convex oracle can invent cross-surface intermediate depths, optimistic lower bound.',
        'regions': 'meters0-10/10-40/>=40; scene_unit0-.5/.5-2/>=2 times TRAIN median=10*depth_scale',
        'plane': 'same definition as diagnostics_plane/protocol.json',
        'field': 'proposals constant within each4x4 ownership block; plane correction not modeled here',
    })
    summary, table = {}, []
    start = time.monotonic()
    for domain in index['domains']:
        scale = domain['depth_scale']
        decoder.set_range(domain['near'] / scale, domain['far'] / scale)
        records = []
        for path in index['splits']['development'][domain['name']]:
            row = move(torch.load(ROOT / path, map_location='cpu', weights_only=False), device)
            prediction = decoder(row['work'])
            truth = row['targets']['depth'].float()
            valid = row['targets']['depth_valid'] & row['inputs']['image_mask'][:, :1]
            valid &= row['inputs']['view_mask'][:, :1, None, None] & torch.isfinite(truth) & (truth > 0)
            outputs, counts = proposal_oracles(prediction, scale, truth, valid)
            ranges = distance_masks(truth, valid, domain)
            partitions = edge_groups(truth, valid)
            partitions.update(local_plane_groups(truth, valid, partitions['edge_5px']))
            partitions['all'] = valid
            statistics = {}
            for region, region_mask in ranges.items():
                statistics[region] = {}
                for part, part_mask in partitions.items():
                    mask = region_mask & part_mask
                    statistics[region][part] = {
                        'metrics': {name: error_statistics(estimate, truth, mask) for name, estimate in outputs.items()},
                        'counts': {name: float(value[mask].double().sum()) if name == 'error_reduction' else int((value & mask).sum())
                                   for name, value in counts.items()},
                    }
            records.append({'cache': path, 'statistics': statistics})
        total = summarize(sum_records([r['statistics'] for r in records]))
        summary[domain['name']] = {'units': domain['geometry_units'], 'statistics': total}
        write_json(out / (domain['name'] + '.json'), {'records': records, 'summary': summary[domain['name']]})
        for region, parts in total.items():
            for part, results in parts.items():
                row = {'dataset': domain['name'], 'units': domain['geometry_units'], 'region': region, 'part': part}
                row['count'] = results['metrics']['original_final']['count']
                row.update({name + '_mae': result['mae'] for name, result in results['metrics'].items()})
                row.update(results['counts'])
                table.append(row)
        print(domain['name'], 'done', flush=True)
    write_json(out / 'summary.json', summary)
    with (out / 'table.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_json(out / 'complete.json', {'frames': 56, 'elapsed_s': time.monotonic() - start, 'test_accessed': False})
    shutil.copy2(__file__, out / 'proposal_supply.py')


if __name__ == '__main__':
    run()
