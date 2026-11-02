"""MF-ET-JAR 的相机、附加模块与基础解码消融配置。"""

from __future__ import annotations

import copy

SWITCHES = (
    "tele_initialization",
    "tele_birth",
    "scale_routing",
    "observability",
    "tele_refinement",
    "resolution_fusion",
)
CORE_SWITCHES = {
    "encoder_queries": True,
    "calibration_step": True,
    "detail_births": True,
    "global_search": True,
    "photo_matching": True,
    "query_conditioned_matching": False,
    "parent_children": False,
    "edge_queries": False,
    "edge_ruler": False,
    "protect_parent": False,
    "infinity_guard": False,
    "shared_view_weights": False,
}
DEFAULT_CAMERA = {
    "mode": "wide4",
    "wide_ids": ["wide_left", "wide_right", "wide_up", "wide_down"],
    "tele_ids": ["tele_left", "tele_right"],
    "allow_missing": False,
    "disabled_ids": [],
}


def configure(cfg):
    """不强制通用 RGB/K/T API 使用固定槽；显式 camera 配置才验证身份。"""
    archived = ("et_multi_hypothesis", "et_reverse_verification", "et_exact_patch_matching")
    if any(cfg["model"].get(key, False) for key in archived):
        raise ValueError(
            "此配置启用了未采纳的研究分支，请在 "
            "runs/carla_research_20261011/research_archive 中复跑，不能在主线静默忽略。"
        )
    mf = cfg["model"].setdefault("multifocal", {})
    mf.setdefault("enabled", True)
    for key in ("enabled", *SWITCHES):
        mf.setdefault(key, True)
        if not isinstance(mf[key], bool):
            raise ValueError("model.multifocal." + key + " 必须是 bool")
    for key in CORE_SWITCHES:
        value = cfg["model"].get("et_" + key, CORE_SWITCHES[key])
        if not isinstance(value, bool):
            raise ValueError("model.et_" + key + " 必须是 bool")
    if "camera" not in cfg:
        return cfg
    camera = {**copy.deepcopy(DEFAULT_CAMERA), **cfg["camera"]}
    cfg["camera"] = camera
    if camera["mode"] not in ("wide4", "wide4_tele2"):
        raise ValueError("camera.mode 必须是 wide4 或 wide4_tele2")
    if len(camera["wide_ids"]) != 4 or len(camera["tele_ids"]) != 2:
        raise ValueError("必须配置四个广角 ID 和两个长焦 ID")
    ids = camera["wide_ids"] + camera["tele_ids"]
    if not all(isinstance(value, str) and value for value in ids) or len(set(ids)) != 6:
        raise ValueError("六个 camera ID 必须是非空且互不重复的字符串")
    if not isinstance(camera["allow_missing"], bool):
        raise ValueError("camera.allow_missing 必须是 bool")
    disabled = camera["disabled_ids"]
    if not isinstance(disabled, list) or not all(isinstance(value, str) for value in disabled):
        raise ValueError("camera.disabled_ids 必须是相机 ID 字符串列表")
    selected = camera_ids(cfg)
    if len(set(disabled)) != len(disabled) or set(disabled) - set(selected):
        raise ValueError("disabled_ids 不能重复，也不能包含当前相机模式之外的 ID")
    if len(disabled) == len(selected):
        raise ValueError("不能主动禁用全部相机")
    return cfg


def camera_ids(cfg):
    camera = cfg["camera"]
    return list(camera["wide_ids"]) + (
        list(camera["tele_ids"]) if camera["mode"] == "wide4_tele2" else []
    )


def effective_mode_metadata(cfg):
    camera = cfg.get("camera")
    mf = cfg["model"].get("multifocal", {})
    shared_depth = cfg["model"].get("shared_depth", {}).get("enabled", False)
    direct = cfg["model"]["name"] == "ruler_direct"
    active = bool(
        not shared_depth and camera and camera["mode"] == "wide4_tele2"
        and mf.get("enabled", True)
    )
    return {
        "model": cfg["model"]["name"],
        "camera_mode": camera["mode"] if camera else "generic_wide",
        "camera_ids": camera_ids(cfg) if camera else None,
        "disabled_camera_ids": list(camera.get("disabled_ids", [])) if camera else [],
        "multifocal_active": active,
        "switches_effective": {key: bool(active and mf.get(key, True)) for key in SWITCHES},
        "core_switches": {
            key: not (shared_depth or direct) and cfg["model"].get("et_" + key, default)
            for key, default in CORE_SWITCHES.items()
        },
        "output_heads": {
            "sparse": not shared_depth and cfg["model"].get("sparse_decoder", {}).get("enabled", True),
            "dense": (
                cfg["model"].get("dense_decoder", {}).get("enabled", False)
                or shared_depth
            ),
        },
        "missing_camera_policy": (
            "masked_degradation" if camera and camera["allow_missing"] else "error"
        ),
        "direct_reference_indices": (
            cfg["direct"].get("reference_indices", [cfg["direct"].get("reference_index", 2)])
            if direct else None
        ),
    }
