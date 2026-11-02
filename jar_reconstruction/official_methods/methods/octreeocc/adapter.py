"""OctreeOcc 官方发布完整性审计；缺失核心实现时不构造替代网络。"""

from __future__ import annotations

import ast
from pathlib import Path

UPSTREAM = Path(__file__).resolve().parent / "upstream"
COMMIT = "54afdd64b210fa45a0b1e1f255e4f88739b4e5a8"


def audit_repository(root: Path = UPSTREAM) -> dict:
    """只读审计可复现的发布问题，不将本审计当作模型运行测试。"""
    cfg = root / "projects/configs/OctreeOcc/octreeocc_base.py"
    findings = []
    try:
        compile(cfg.read_text(), str(cfg), "exec")
    except SyntaxError as exc:
        findings.append(
            {
                "kind": "invalid_official_config",
                "path": str(cfg.relative_to(root)),
                "line": exc.lineno,
                "detail": exc.msg,
            }
        )
    modules = root / "projects/mmdet3d_plugin/bevformer/modules"
    missing = modules / "temporal_self_attention.py"
    if not missing.exists():
        findings.append(
            {
                "kind": "missing_import",
                "path": str(missing.relative_to(root)),
                "detail": "octree_transformer.py imports this unpublished module",
            }
        )
    detector = root / "projects/mmdet3d_plugin/bevformer/detectors/octree_occ.py"
    names = [
        node.name
        for node in ast.walk(ast.parse(detector.read_text()))
        if isinstance(node, ast.ClassDef)
    ]
    if "OctreeOcc_pred" not in names:
        findings.append(
            {
                "kind": "unregistered_config_type",
                "detail": "config selects OctreeOcc_pred; published detector is OctreeOcc",
            }
        )
    findings.append(
        {
            "kind": "unpublished_semantic_initializer",
            "path": "projects/mmdet3d_plugin/datasets/pipelines/loading.py:LoadSegPriorFromFile",
            "detail": "RGB-only inference requires UNet-generated seg_structure_l1/l2; released code only loads offline NPZ, without its generation pipeline, trained UNet or training configuration",
        }
    )
    findings.append(
        {
            "kind": "paper_code_temporal_difference",
            "detail": "paper uses temporal self-attention; official config has queue_length=16 but layer operation_order omits self_attn, and forward_test supplies prev_bev=None",
        }
    )
    return {
        "method": "octreeocc",
        "commit": COMMIT,
        "status": "blocked_upstream_incomplete",
        "full_model_forward_backward_tested": False,
        "findings": findings,
    }


def require_available() -> None:
    report = audit_repository()
    raise RuntimeError(
        "OctreeOcc 官方发布不完整，当前没有可声称忠实复现的 RGB+标定四目模型。"
        "见 official_methods/docs/octreeocc.md；具体阻塞："
        + "; ".join(item["kind"] for item in report["findings"])
    )


def create_adapter(config):
    """注册入口明确报告未完成；没有用其他网络冒充 OctreeOcc。"""
    require_available()
