"""Utilities to audit onnx2torch run logs for failed conversions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

_ERROR_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bERROR\b", re.IGNORECASE),
    re.compile(r"\bCRITICAL\b", re.IGNORECASE),
    re.compile(r"Failed to convert", re.IGNORECASE),
    re.compile(r"Failed to export", re.IGNORECASE),
    re.compile(r"Warm-up execution failed", re.IGNORECASE),
)


@dataclass(frozen=True)
class RunFailure:
    """Represents a run directory flagged as failed."""

    name: str
    directory: Path
    reason: str


def iter_run_dirs(run_logs_dir: Path) -> Iterator[Path]:
    """Yield run directories inside ``run_logs_dir`` sorted by name."""
    for entry in sorted(run_logs_dir.iterdir()):
        if entry.is_dir():
            yield entry


def _list_failure_artifacts(run_dir: Path) -> List[Path]:
    failures_dir = run_dir / "failures"
    if not failures_dir.exists() or not failures_dir.is_dir():
        return []
    return [
        path
        for path in sorted(failures_dir.iterdir())
        if path.is_file()
    ]


def _format_failure_artifacts(paths: Sequence[Path]) -> str:
    if not paths:
        return ""
    first = paths[0].name
    tail_count = len(paths) - 1
    if tail_count <= 0:
        return first
    return f"{first} (+{tail_count} more)"


def inspect_run(run_dir: Path) -> RunFailure | None:
    """Inspect a run directory and return a ``RunFailure`` if it failed."""
    run_log = run_dir / "run.log"
    if not run_log.is_file():
        return RunFailure(run_dir.name, run_dir, "missing run.log")

    artifacts = _list_failure_artifacts(run_dir)
    if artifacts:
        return RunFailure(
            run_dir.name,
            run_dir,
            f"failure artifacts present: {_format_failure_artifacts(artifacts)}",
        )

    try:
        with run_log.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_number, line in enumerate(handle, start=1):
                if _matches_failure_line(line):
                    return RunFailure(
                        run_dir.name,
                        run_dir,
                        f"log line {line_number}: {line.strip()}",
                    )
    except OSError as error:
        return RunFailure(
            run_dir.name,
            run_dir,
            f"unable to read run.log ({error})",
        )

    return None


def _matches_failure_line(line: str) -> bool:
    return any(pattern.search(line) for pattern in _ERROR_PATTERNS)


def collect_failures(run_dirs: Iterable[Path]) -> List[RunFailure]:
    """Aggregate ``RunFailure`` entries for the provided ``run_dirs``."""
    failures: List[RunFailure] = []
    for run_dir in run_dirs:
        failure = inspect_run(run_dir)
        if failure is not None:
            failures.append(failure)
    return failures


def scan_run_logs_dir(run_logs_dir: Path) -> List[RunFailure]:
    """Scan the ``run_logs_dir`` directory for failed runs."""
    return collect_failures(iter_run_dirs(run_logs_dir))


__all__ = [
    "RunFailure",
    "iter_run_dirs",
    "inspect_run",
    "collect_failures",
    "scan_run_logs_dir",
]
