"""公共TEST的薄适配与统一深度评测；不训练、不改变官方预测算法。"""

import copy
import hashlib
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F

from jar.data import build_dataset
from jar.models.position_cache import install_position_cache
from jar.pretraining.data import dataset_config
from jar.pretraining.model import PretrainingModel
from official_methods.metrics import depth_statistics
from official_methods.postprocess import depth_on_image_grid
from tools.evaluate_public_depth import finish_statistics


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def test_dataset(config, domain):
    return build_dataset(dataset_config(domain, config), 'test', load_targets=True)


def equal_pixel_inputs(inputs):
    """保持已训练分支尺寸，但只能使用共同主图已有的像素信息。"""
    rgb = inputs['rgb']
    batch, views = rgb.shape[:2]
    matching = F.interpolate(rgb.flatten(0, 1), inputs['matching_rgb'].shape[-2:],
                             mode='bilinear', align_corners=False)
    return dict(inputs, matching_rgb=matching.unflatten(0, (batch, views)))


class Models:
    """每个GPU缓存已加载权重；数据域范围和参考号每次显式设置。"""

    def __init__(self, config, device):
        self.config, self.device = config, device
        self.loaded = {}

    def get(self, method, domain):
        settings = self.config['methods'][method]
        kind = settings['kind']
        views = domain['num_views']
        if settings.get('num_views', views) != views:
            raise ValueError('固定视图数的已训练MLP不兼容当前真实相机数')
        # MVSAnywhere的视图独立MLP可以共享权重，配置检查仍按真实N构造。
        key = (kind, settings.get('weights'), views if kind in ('mvsanywhere', 'simplerecon') else 0)
        if kind == 'own':
            key = (kind, settings.get('weights', self.config['own_weights']),
                   json.dumps(settings.get('inference', {}), sort_keys=True))
        if key in self.loaded:
            return self.loaded[key]
        height, width = self.config['image_size']
        if kind == 'own':
            saved = torch.load(key[1], map_location='cpu', weights_only=False, mmap=True)
            model = PretrainingModel(saved['config'], initialize=False)
            model.load_state_dict(saved['model'], strict=True)
            if self.config['position_cache']:
                install_position_cache(model.foundation.model)
        else:
            from official_methods.registry import build_adapter
            native = {'method': kind, 'model': {
                'num_views': views, 'image_height': height, 'image_width': width,
                'reference_indices': [domain['reference_index']], 'pretrained': False,
                'weights': settings['weights'] if kind != 'simplerecon' and not settings.get('project_checkpoint') else '',
                'depth_bins': 64, 'min_depth': domain['near'], 'max_depth': domain['far'],
                'fast_cost_volume': self.config['fast_cost_volume'],
                'depth_range': [domain['near'], domain['far']],
                'test_depth_planes': 96, 'normalization': 'consistent_standardized'},
                'evaluation': {'reference_indices': [domain['reference_index']]}}
            if kind == 'simplerecon' or settings.get('project_checkpoint'):
                native['_loading_shared_checkpoint'] = True
            model = build_adapter(native)
            if kind == 'simplerecon' or settings.get('project_checkpoint'):
                from jar.models.shared_initializer import _load_projective_state
                saved = torch.load(settings['weights'], map_location='cpu', weights_only=False, mmap=True)
                if saved['method'] != kind:
                    raise ValueError('checkpoint方法不符')
                _load_projective_state(model, saved['model'])
                if self.config['fast_cost_volume']:
                    model.native_model.cost_volume = model.native_model.cost_volume.to_fast()
            if kind == 'mvsanywhere':
                from jar.models.shared_initializer import PrimaryDepthHead
                model.native_model.depth_decoder = PrimaryDepthHead(model.native_model.depth_decoder)
                if self.config['position_cache']:
                    install_position_cache(model.native_model.encoder.model)
                    install_position_cache(model.native_model.cost_volume_net.model)
        model.requires_grad_(False)
        model.to(self.device).eval()
        if kind == 'own' and settings.get('inference'):
            from jar.pretraining.inference import prepare_inference
            prepare_inference(model, **settings['inference'])
        self.loaded[key] = model
        return model

    def forward(self, method, domain, inputs):
        settings = self.config['methods'][method]
        kind, reference = settings['kind'], domain['reference_index']
        if kind in ('mvsanywhere', 'fast_mvsnet') and settings.get('drop_missing_views'):
            if not bool(inputs['view_mask'].all()):
                compact, local_reference = compact_valid_views(inputs, reference)
                local_domain = dict(domain, num_views=compact['rgb'].shape[1], reference_index=local_reference)
                prediction = self.forward(method, local_domain, compact)
                # 深度与内参仍属于同一真实参考相机，仅恢复其公共槽位编号。
                return dict(prediction, reference_indices=[reference])
        model = self.get(method, domain)
        if kind == 'own':
            work = equal_pixel_inputs(inputs) if settings['input_mode'] == 'equal_pixels' else inputs
            output = model(work, domain, reference=reference)
            return {'depth': output['depth'] * output['unit_scale'], 'valid': output['valid'],
                    'reference_indices': [reference]}
        if kind == 'fast_mvsnet':
            model.depth_min, model.depth_max = domain['near'], domain['far']
            model.options['reference_indices'] = [reference]
            model.config['evaluation']['reference_indices'] = [reference]
        else:
            model.reference_indices = [reference]
            if kind == 'mvsanywhere':
                model.settings['range_refinement_steps'] = settings['refinements']
                if settings.get('shared_train_range', False):
                    # 额外公平性对照：使用与自研相同的TRAIN全局范围。
                    # 调用官方显式范围接口；不读取测试深度或修改模型参数。
                    current, sources, _ = model.prepare_native(inputs, reference)
                    current['min_depth'] = domain['near']
                    current['max_depth'] = domain['far']
                    native = model.native_model(
                        'test', current, sources, unbatched_matching_encoder_forward=False,
                        return_mask=True, estimate_depth_range=False,
                        num_refinement_steps=settings['refinements'])
                    depth = native['depth_pred_s0_b1hw']
                    return {'depth': depth, 'valid': torch.isfinite(depth) & (depth > 0),
                            'reference_indices': [reference]}
            else:
                model.native_model.run_opts.min_matching_depth = domain['near']
                model.native_model.run_opts.max_matching_depth = domain['far']
        context = (torch.backends.cudnn.flags(deterministic=True, benchmark=False, allow_tf32=False)
                   if kind == 'fast_mvsnet' else nullcontext())
        with context:
            return model(inputs)

    def convert(self, prediction, inputs):
        if 'depth_K' in prediction:
            converted = depth_on_image_grid(prediction, inputs)
            return converted['depth'], converted['depth_valid']
        depth = prediction['depth']
        valid = prediction['valid'] & torch.isfinite(depth) & (depth > 0)
        if depth.shape[-2:] != inputs['rgb'].shape[-2:]:
            raise ValueError('没有depth_K的预测必须位于共同主图格点')
        return depth, valid


def compact_valid_views(inputs, reference):
    """单样本评测只向原生可变视图模型提供真实有效相机，不补图。"""
    if inputs['rgb'].shape[0] != 1:
        raise ValueError('缺相机薄适配仅用于batch1评测')
    active = inputs['view_mask'][0].nonzero().flatten().tolist()
    if reference not in active or len(active) < 2:
        raise ValueError('原生多目方法至少需要参考相机和一个有效源相机')
    views = inputs['rgb'].shape[1]
    result = {key: value[:, active] if torch.is_tensor(value) and value.ndim >= 2
              and value.shape[:2] == (1, views) else value for key, value in inputs.items()}
    return result, active.index(reference)


def measure(depth, valid, batch, reference, depth_range=None):
    target, inputs = batch['targets'], batch['inputs']
    gt = target['depth'][:, reference:reference + 1].double()
    mask = target['depth_valid'][:, reference:reference + 1]
    mask = mask & inputs['image_mask'][:, reference:reference + 1]
    mask = mask & inputs['view_mask'][:, reference:reference + 1, None, None]
    mask = mask & torch.isfinite(gt) & (gt > 0)
    # 不用模型置信度、候选范围或预测误差筛选GT。
    near, far = depth_range if depth_range is not None else (0., math.inf)
    return depth_statistics(depth.double(), gt, mask, valid, near, far)


def add_statistics(total, item):
    for key, value in item.items():
        total[key] = total.get(key, 0) + value


def metrics(total, units):
    result = finish_statistics(total, units)
    # 缺失预测时绝不将条件误差伪装成全GT的有限MAE。
    result['full_gt_MAE'] = result['MAE'] if total['gt_pixels'] == total['predicted_pixels'] else None
    return result
