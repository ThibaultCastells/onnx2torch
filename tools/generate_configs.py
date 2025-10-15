#!/usr/bin/env python3
"""Generate onnx2torch config stubs for shape-inferred ONNX models."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import yaml

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RepoConfig:
    """Resolved paths required to generate a config file."""

    name: str
    data_dir: Path
    config_path: Path


def _iter_repo_configs(data_root: Path, cfg_root: Path) -> Iterator[RepoConfig]:
    """Yield config descriptors for every repository under *data_root*."""

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir():
            continue
        yield RepoConfig(entry.name, entry, cfg_root / f"{entry.name}.yaml")


def _relpath(path: Path, start: Path) -> str:
    return os.path.relpath(path, start=start)


def _build_config_payload(repo: RepoConfig) -> dict[str, object]:
    """Create the YAML payload for *repo*."""

    config_dir = repo.config_path.parent
    project_root = config_dir.parent
    input_path = Path(_relpath(repo.data_dir, project_root)).as_posix()
    return {
        "inputs": {
            "path": input_path,
            "recursive": True,
            "pattern": "*.onnx",
        },
        "output_dir": f"data/executorch/{repo.name}",
        "conversion": {
            "save_input_names": False,
            "attach_onnx_mapping": False,
        },
        "device": "cpu",
    }


def _write_config(repo: RepoConfig, *, overwrite: bool) -> bool:
    """Materialise the config file for *repo*.

    Returns ``True`` if a file was written, ``False`` when skipped (existing file).
    """

    payload = _build_config_payload(repo)

    if repo.config_path.exists() and not overwrite:
        try:
            existing_data = yaml.safe_load(
                repo.config_path.read_text(encoding="utf-8")
            )
        except yaml.YAMLError:  # pragma: no cover - defensive guard
            existing_data = None

        if isinstance(existing_data, dict):
            inputs_section = existing_data.get("inputs")
            if isinstance(inputs_section, dict):
                current_path = str(inputs_section.get("path", ""))
                if current_path == payload["inputs"]["path"]:
                    LOGGER.debug("Skipping %s (config already exists)", repo.name)
                    return False

    repo.config_path.parent.mkdir(parents=True, exist_ok=True)
    with repo.config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)

    LOGGER.info("Generated %s", repo.config_path)
    return True


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../onnx_shape_inference/data/shape_inferred"),
        help="Directory containing shape-inferred ONNX repositories.",
    )
    parser.add_argument(
        "--cfg-root",
        type=Path,
        default=Path("cfg"),
        help="Destination directory for generated YAML configs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite configs that already exist.",
    )
    parser.add_argument(
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be specified multiple times).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)

    log_level = logging.WARNING - min(args.verbose, 2) * 10
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    data_root = args.data_root.resolve()
    cfg_root = args.cfg_root.resolve()

    LOGGER.debug("Resolved data_root=%s", data_root)
    LOGGER.debug("Resolved cfg_root=%s", cfg_root)

    repos = list(_iter_repo_configs(data_root, cfg_root))
    if not repos:
        LOGGER.warning("No repositories found under %s", data_root)
        return

    generated = 0
    for repo in repos:
        if _write_config(repo, overwrite=args.overwrite):
            generated += 1

    LOGGER.info(
        "Finished. %d configs generated out of %d repositories.", generated, len(repos)
    )


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    main()
