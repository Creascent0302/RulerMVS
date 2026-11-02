"""按需加载，避免不同官方仓库的依赖互相污染。"""

import importlib

METHODS = (
    "fast_mvsnet", "simplerecon", "mvsanywhere", "mvsformerpp",
    "patchmatchnet", "diffmvs", "octreeocc", "gaussianformer2",
)


def build_adapter(config):
    name = config["method"]
    if name not in METHODS:
        raise ValueError(f"未知方法 {name}; 支持 {METHODS}")
    module = importlib.import_module(f"official_methods.methods.{name}")
    if hasattr(module, "create_adapter"):
        return module.create_adapter(config)
    return module.Adapter(config)
