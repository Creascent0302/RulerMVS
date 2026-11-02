"""ZNCC 工程对照；仅固定共同扫描器使用的匹配函数。"""

from ..classical import ClassicalMVS


class Adapter(ClassicalMVS):
    def __init__(self, cfg):
        super().__init__(cfg, method="zncc")
