"""目录重组后的公共导入、官方代码位置和旧配置兼容性。"""

import importlib
import json
from pathlib import Path

import pytest

from official_methods.registry import METHODS


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", METHODS)
def test_method_package_and_legacy_source_path(name):
    package = ROOT / "methods" / name
    module = importlib.import_module(f"official_methods.methods.{name}")
    adapter_module = importlib.import_module(f"official_methods.methods.{name}.adapter")

    assert Path(module.__file__).resolve() == package / "__init__.py"
    assert (package / "README.md").is_file()
    if name == "octreeocc":
        assert module.create_adapter is adapter_module.create_adapter
        assert module.audit_repository()["status"] == "blocked_upstream_incomplete"
    else:
        assert module.Adapter is adapter_module.Adapter

    upstream_name = "gaussianformer" if name == "gaussianformer2" else name
    legacy = ROOT / "upstream" / upstream_name
    assert legacy.is_symlink()
    assert not (package / "upstream").is_symlink()
    assert legacy.resolve() == package / "upstream"
    if name in ("fast_mvsnet", "gaussianformer2", "octreeocc"):
        assert module.UPSTREAM == package / "upstream"
    if name == "gaussianformer2":
        assert module.ROOT == ROOT


@pytest.mark.parametrize("name", METHODS)
def test_config_and_patch_aliases_use_one_physical_file(name):
    package = ROOT / "methods" / name
    for legacy in (ROOT / "configs").glob(f"{name}_*.json"):
        config = package / "configs" / legacy.name.removeprefix(f"{name}_")
        assert legacy.is_symlink()
        assert legacy.resolve() == config
        assert json.loads(config.read_text())["method"] == name
    for profile in ("carla", "official"):
        assert (package / "configs" / f"{profile}.json").is_file()
    for patch in (package / "patches").glob("*.patch"):
        legacy = ROOT / "patches" / patch.name
        assert legacy.is_symlink()
        assert legacy.resolve() == patch


def test_shared_dependencies_and_niantic_geometry_keep_their_location():
    from official_methods.methods.niantic_common import UPSTREAM

    assert UPSTREAM == ROOT / "upstream"
    for name in ("dinov2", "pointcept"):
        assert (UPSTREAM / name).is_dir()
        assert not (UPSTREAM / name).is_symlink()
