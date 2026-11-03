"""本轮固定权重适配；复用已验证的部署优化，不修改训练模型。"""

import torch

from jar.models.position_cache import install_position_cache
from jar.models.shared_initializer import PrimaryDepthHead
from jar.pretraining.comparison import Models as BaseModels, equal_pixel_inputs
from research.overnight_v6_v7_mvs_20261103.train import build_model
from research.rulermvs_v6_20261101.inference import prepare_fast_inference


class Models(BaseModels):
    def __init__(self, config, device, fixed_random_stream=True):
        super().__init__(config, device)

    def get(self, method, domain):
        settings = self.config['methods'][method]
        if not settings.get('retrained'):
            return super().get(method, domain)
        key = (method, settings['weights'])
        if key not in self.loaded:
            saved = torch.load(settings['weights'], map_location='cpu', weights_only=False, mmap=True)
            with torch.inference_mode(False), torch.no_grad():
                model = build_model(saved['config'], initialize=False)
                model.load_state_dict(saved['model'], strict=True)
                model.requires_grad_(False).to(self.device).eval()
                if settings['kind'] == 'own':
                    install_position_cache(model.foundation.model)
                    prepare_fast_inference(model, self.config['fast'])
                else:
                    native = model.native_model
                    native.cost_volume = native.cost_volume.to_fast().to(self.device)
                    native.depth_decoder = PrimaryDepthHead(native.depth_decoder)
                    install_position_cache(native.encoder.model)
                    install_position_cache(native.cost_volume_net.model)
                    model.config['mvsanywhere']['refinements'] = settings['refinements']
                    model.config['mvsanywhere']['estimate_depth_range'] = False
                    model.requires_grad_(False).eval()
            self.loaded[key] = model
        return self.loaded[key]

    def forward(self, method, domain, inputs):
        settings = self.config['methods'][method]
        if not settings.get('retrained'):
            return super().forward(method, domain, inputs)
        model = self.get(method, domain)
        work = equal_pixel_inputs(inputs) if settings['kind'] == 'own' else inputs
        result = model(work, domain, reference=domain['reference_index'])
        return {'depth': result['depth'] * result['unit_scale'], 'valid': result['valid'],
                'reference_indices': [domain['reference_index']]}
