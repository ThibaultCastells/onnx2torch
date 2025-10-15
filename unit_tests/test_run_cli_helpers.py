from __future__ import annotations

from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType
import sys

import pytest


def _write_cfg(path: Path) -> None:
    path.write_text("inputs:\n  - path: data\n", encoding="utf-8")


def _load_runner_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "run.py"
    spec = importlib_util.spec_from_file_location("onnx2torch_run", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load onnx2torch/run.py for testing.")
    module = importlib_util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner_module()


def test_discover_config_paths_directory(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    first = cfg_dir / "a.yaml"
    second = cfg_dir / "b.yml"
    other = cfg_dir / "ignore.txt"
    _write_cfg(first)
    _write_cfg(second)
    other.write_text("", encoding="utf-8")

    resolved_root, configs = runner._discover_config_paths(cfg_dir)

    assert resolved_root == cfg_dir.resolve()
    assert configs == [first.resolve(), second.resolve()]


def test_discover_config_paths_file(tmp_path: Path) -> None:
    cfg_file = tmp_path / "single.yaml"
    _write_cfg(cfg_file)

    resolved_root, configs = runner._discover_config_paths(cfg_file)

    assert resolved_root == cfg_file.resolve()
    assert configs == [cfg_file.resolve()]


def test_discover_config_paths_missing(tmp_path: Path) -> None:
    missing = tmp_path / "absent.yaml"

    with pytest.raises(FileNotFoundError):
        runner._discover_config_paths(missing)


def test_discover_config_paths_invalid_extension(tmp_path: Path) -> None:
    invalid = tmp_path / "config.json"
    invalid.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        runner._discover_config_paths(invalid)
