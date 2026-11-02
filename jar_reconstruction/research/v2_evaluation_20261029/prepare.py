"""固定评测对象与新CARLA清单；不依据TEST深度或预测结果选择范围/权重。"""
from pathlib import Path
import copy
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from jar.pretraining.comparison import write_json


def main():
    folder = Path(__file__).parent
    output = ROOT / 'runs/active/v2_evaluation_20261029'
    public = json.loads((ROOT / 'research/v2_training_20261028/configs/train_public.json').read_text())
    carla = json.loads((ROOT / 'research/v2_training_20261028/configs/train_carla_new.json').read_text())
    previous = json.loads((ROOT / 'research/near_mid_evaluation_20261028/config.json').read_text())
    baseline = previous['methods']['mvsanywhere_refined']
    public_path = str(ROOT / 'runs/active/v2_training_20261028/public/pretrained.pt')
    carla_path = str(ROOT / 'runs/active/v2_training_20261028/carla_new/pretrained.pt')
    own = lambda weights, mode='native': {'kind': 'own', 'weights': weights, 'input_mode': mode,
                                        'inference': {'spatial_layout': True}}
    public_methods = {'v2_public': own(public_path), 'v2_public_equal': own(public_path, 'equal_pixels'),
                      'mvsanywhere_refined': baseline}
    carla_methods = {'v2_public': own(public_path), 'v2_carla_new': own(carla_path),
                     'v2_carla_new_equal': own(carla_path, 'equal_pixels'),
                     'mvsanywhere_refined': baseline}
    common = {k: previous[k] for k in ('image_size', 'matching_image_size', 'gpus', 'cpu_threads',
                                       'loader_workers', 'shard_size', 'surface_queries',
                                       'tf32', 'position_cache', 'fast_cost_volume')}
    common.update(training_config='research/v2_training_20261028/configs/train_public.json',
                  own_weights=public_path, seed=20261029)
    common['protocol_notes'] = [
        '固定VAL选择的导出权重；公共导出已回退旧V2，新CARLA导出为stage2 epoch1。TEST不选模。',
        '主图256x448，自研native匹配512x896；equal_pixels仅由共同256x448主图上采样，仍非相同计算预算。',
        '真实N=2或4；缺源图仅压缩实际存在视图交给MVSAnywhere，不复制图片。',
        'MVSAnywhere官方Hero细化1次，使用相同TRAIN固定搜索范围；无测试范围估计或GT输入。',
        '公共三维scene_unit不称米；原生系统精度差异不能单独归因架构。',
        '新CARLA微调V2与官方未做新CARLA微调的baseline训练经历不同；单独列公共V2迁移对照。',
        '同GT掩码，按像素汇总误差；缺失预测报告coverage，delta按全部GT分母；不删范围外预测。',
        '表面是单参考GT可见区深度反投影，不是全场景重建或语义占据。']
    groups = {'public': (public['datasets'], public_methods),
              'carla': ([dict(carla['datasets'][0], name='CARLA_new_upper')], carla_methods)}
    # 额外左右下参考用于检查参考轮换训练的泛化；主上参考仍用训练前固定TEST。
    refs = []
    source = ROOT / 'runs/active/v2_training_20261028/carla_prepared'
    split_counts = {}
    for reference, camera in enumerate(carla['datasets'][0]['camera']['wide_ids']):
        if reference == 2:
            continue
        domain = copy.deepcopy(carla['datasets'][0])
        domain.update(name='CARLA_new_' + camera, reference_index=reference)
        for split in ('val', 'test'):
            rows = [json.loads(l) for l in (source / (split + '_supervised.jsonl')).read_text().splitlines()]
            selected = [row for row in rows if camera in row['depth_supervised_camera_ids']
                        and row['valid_camera_count'] >= 2]
            path = output / 'manifests' / (split + '_' + camera + '.jsonl')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected))
            domain['manifests'][split] = str(path)
            split_counts[split + '_' + camera] = len(selected)
        refs.append(domain)
    groups['carla_other_refs'] = (refs, {k: v for k, v in carla_methods.items() if not k.endswith('_equal')})
    for name, (domains, methods) in groups.items():
        cfg = dict(common, output=str(output/name), datasets=domains, methods=methods)
        write_json(folder/(name+'.json'), cfg)
        if name != 'carla_other_refs':
            speed = {'data_config': str(folder/(name+'.json')), 'output': str(output/name/'latency'),
                     'own_weights': public_path, 'methods': methods, 'gpu': 0, 'cpu_threads': 2,
                     'seed': 20261029, 'warmup': 10, 'repetitions': 40, 'isolated_resources': True,
                     'position_cache': True, 'fast_cost_volume': True}
            write_json(folder/(name+'_speed.json'), speed)
    write_json(output/'preparation.json', {'reference_manifest_counts': split_counts,
                'test_used_for': '仅读取帧/相机标识生成参考可用清单，不读取深度内容、不改变范围或挑权重'})
    print(split_counts)


if __name__ == '__main__':
    main()
