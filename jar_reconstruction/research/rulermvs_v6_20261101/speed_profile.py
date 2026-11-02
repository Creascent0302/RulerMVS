"""V5无标签VAL分阶段耗时；原权重、分辨率和计算精度保持不变。"""

import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from jar.data import build_dataset, collate_frames
from jar.pretraining.comparison import digest, equal_pixel_inputs, write_json
from jar.pretraining.data import dataset_config
from jar.runtime import move
from research.rulermvs_v5_20261031.infer import load_model
from tools.profile_public_inference import StageRecorder, benchmark, install_stages, operator_report


def main():
    settings = json.loads((ROOT / 'research/baseline_full_comparison_20261101/configs/speed_data.json').read_text())
    output = ROOT / 'runs/active/rulermvs_v6_20261101/speed_profile'
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(20261101)
    domain = settings['datasets'][0]
    weights = settings['methods']['v5_best']['weights']
    model, _ = load_model(weights, 'cuda:0', inference={'spatial_layout': True})
    dataset = build_dataset(dataset_config(domain, settings), 'val', load_targets=False)
    index = len(dataset) // 2
    inputs = move(collate_frames([dataset[index]])['inputs'], 'cuda:0')
    reference = domain['reference_index']
    call = lambda: model(equal_pixel_inputs(inputs), domain, reference)
    with torch.inference_mode():
        native = benchmark(call, 15, 50)
        original = call()['depth'].clone()
        recorder = StageRecorder()
        install_stages(model, recorder)
        recorder.wrap(model, '_foundation_with_bridge', 'foundation_with_bridge')
        recorder.wrap(model.bridge, 'forward', 'cnn_token_bridge')
        try:
            runs = []
            for _ in range(15):
                current = call()
                runs.append(recorder.collect())
            stages = {name: {key: statistics.median(run[name][key] for run in runs)
                             for key in ('inclusive_ms', 'exclusive_ms', 'calls')}
                      for name in runs[0]}
            difference = float((current['depth'] - original).abs().max())
            recorder.events_enabled = False
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True) as profiler:
                for _ in range(3):
                    call()
                torch.cuda.synchronize()
            operators = operator_report(profiler, 3)
            write_json(output / 'operators.json', operators)
            write_json(output / 'shapes.json', recorder.shape_records)
            write_json(output / 'profile.json', {
                'completed': True, 'gpu': torch.cuda.get_device_name(), 'weights': weights,
                'checkpoint_sha256': digest(weights), 'index': index, 'split': 'val',
                'load_targets': False, 'image_size': settings['image_size'],
                'matching_image_size': settings['matching_image_size'], 'input_mode': 'equal_pixels',
                'native': native, 'stages': stages,
                'instrumentation_max_depth_difference': difference,
                'notes': 'FP32/TF32off，输入驻GPU。子阶段事件包含CPU排队空隙，不相加冒充无插桩延迟。',
            })
            print(json.dumps({'native_ms': native['wall_median_ms'], 'difference': difference,
                              'stages': stages}, ensure_ascii=False), flush=True)
        finally:
            recorder.uninstall()


if __name__ == '__main__':
    main()
