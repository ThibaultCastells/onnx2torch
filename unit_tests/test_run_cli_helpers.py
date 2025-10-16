from __future__ import annotations

from datetime import UTC, datetime
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


def _freeze_datetime(monkeypatch: pytest.MonkeyPatch, value: datetime) -> None:
    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz: object | None = None) -> datetime:
            if tz is not None:
                return value.astimezone(tz)  # type: ignore[call-arg]
            return value

    monkeypatch.setattr(runner, "datetime", _FrozenDateTime)


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


def test_prepare_run_directory_uses_config_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen_time = datetime(2025, 10, 15, 7, 55, 59, tzinfo=UTC)
    _freeze_datetime(monkeypatch, frozen_time)
    monkeypatch.setattr(runner, "RUN_LOG_ROOT", tmp_path / "run_logs")

    config_path = tmp_path / "cfg" / "sample-config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("inputs: []\n", encoding="utf-8")

    run_dir = runner._prepare_run_directory(config_path)

    assert run_dir == tmp_path / "run_logs" / "20251015_075559_sample-config"
    assert run_dir.is_dir()


def test_prepare_run_directory_strips_yml_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen_time = datetime(2025, 10, 15, 7, 55, 59, tzinfo=UTC)
    _freeze_datetime(monkeypatch, frozen_time)
    monkeypatch.setattr(runner, "RUN_LOG_ROOT", tmp_path / "logs")

    config_path = tmp_path / "cfg" / "experiment.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("inputs: []\n", encoding="utf-8")

    run_dir = runner._prepare_run_directory(config_path)

    assert run_dir == tmp_path / "logs" / "20251015_075559_experiment"
    assert run_dir.is_dir()
