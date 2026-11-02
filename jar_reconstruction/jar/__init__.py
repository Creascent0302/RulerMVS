"""MF-ET-JAR：可独立消融的强标定多视角重建。"""

from .config import load_config


def build_model(cfg, checkpoint_state=None):
    if cfg["model"]["name"] == "ruler_direct":
        from .models.ruler_direct_model import RulerDirectModel

        return RulerDirectModel(cfg, model_state=checkpoint_state)
    if cfg["model"].get("shared_depth", {}).get("enabled", False):
        from .models.shared_depth import SharedGeometryDepthModel

        return SharedGeometryDepthModel(cfg, checkpoint_state=checkpoint_state)
    from .model import ReconstructionModel

    return ReconstructionModel(cfg, checkpoint_state=checkpoint_state)
