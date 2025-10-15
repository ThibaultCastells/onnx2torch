#!/usr/bin/env python3
"""Report run log folders that contain failed conversions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable


def _ensure_project_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_on_path()
from onnx2torch.run_log_detection import (  # noqa: E402
    RunFailure,
    scan_run_logs_dir,
)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect onnx2torch run log folders that failed."
    )
    parser.add_argument(
        "--run-logs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "run_logs",
        help="Path to the run logs directory (default: ./run_logs relative to this script).",
    )
    parser.add_argument(
        "--exit-zero",
        action="store_true",
        help="Always exit with status code 0 even when failures are found.",
    )
    return parser.parse_args(list(argv))


def _print_failures(failures: Iterable[RunFailure]) -> None:
    for failure in failures:
        print(f"{failure.name}: {failure.reason}")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run_logs_dir = args.run_logs_dir.expanduser()

    if not run_logs_dir.exists():
        print(
            f"[ERROR] Run logs directory does not exist: {run_logs_dir}",
            file=sys.stderr,
        )
        return 1

    if not run_logs_dir.is_dir():
        print(
            f"[ERROR] Provided path is not a directory: {run_logs_dir}",
            file=sys.stderr,
        )
        return 1

    failures = scan_run_logs_dir(run_logs_dir)

    if failures:
        _print_failures(failures)
        print(f"\nDetected {len(failures)} failed run(s) in {run_logs_dir}")
        return 0 if args.exit_zero else 1

    print(f"No failed runs detected in {run_logs_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
