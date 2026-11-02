"""独立 baseline 入口；共享扫描实现保持历史导入兼容。"""

from .classical import ClassicalMVS, build_baseline

__all__ = ["ClassicalMVS", "build_baseline"]
