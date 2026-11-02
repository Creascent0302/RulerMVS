"""本轮重测注册：复用原指标，增加同权重 V6 快速执行。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.rulermvs_v6_20261101.evaluation import Models as BaseModels
from research.rulermvs_v6_20261101.speed_final import load
from research.near_mid_evaluation_20261028 import evaluate


class Models(BaseModels):
    def get(self, method, domain):
        settings = self.config['methods'][method]
        if not settings.get('deployment_fast'):
            return super().get(method, domain)
        key = ('v6_fast', settings['weights'])
        if key not in self.loaded:
            self.loaded[key] = load({
                'v6_checkpoint': settings['weights'],
                'gpu': self.device.index,
                'fast': self.config['fast'],
            }, 'v6_fast')
        return self.loaded[key]


evaluate.Models = Models

if __name__ == '__main__':
    evaluate.main()
