from __future__ import annotations

from pathlib import Path

from onnx2torch.run_log_detection import (
    collect_failures,
    inspect_run,
    scan_run_logs_dir,
)


def test_inspect_run_missing_log(tmp_path: Path) -> None:
    run_dir = tmp_path / "20251015_000000"
    run_dir.mkdir()

    failure = inspect_run(run_dir)

    assert failure is not None
    assert failure.reason == "missing run.log"


def test_inspect_run_failure_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "20251015_000100"
    failures_dir = run_dir / "failures"
    failures_dir.mkdir(parents=True)
    (run_dir / "run.log").write_text("INFO: start\n", encoding="utf-8")
    (failures_dir / "model_a.log").write_text("traceback\n", encoding="utf-8")
    (failures_dir / "model_b.log").write_text("traceback\n", encoding="utf-8")

    failure = inspect_run(run_dir)

    assert failure is not None
    assert "failure artifacts" in failure.reason
    assert "model_a.log" in failure.reason
    assert "+1 more" in failure.reason


def test_inspect_run_error_line(tmp_path: Path) -> None:
    run_dir = tmp_path / "20251015_000200"
    run_dir.mkdir()
    (run_dir / "run.log").write_text(
        "INFO: ok\n2025-10-15 12:00:00,000 ERROR something bad happened\n",
        encoding="utf-8",
    )

    failure = inspect_run(run_dir)

    assert failure is not None
    assert "log line 2" in failure.reason
    assert "ERROR" in failure.reason


def test_collect_failures(tmp_path: Path) -> None:
    ok_dir = tmp_path / "ok"
    ok_dir.mkdir()
    (ok_dir / "run.log").write_text("INFO: success\n", encoding="utf-8")

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "run.log").write_text("CRITICAL failure\n", encoding="utf-8")

    failures = collect_failures([ok_dir, bad_dir])

    assert len(failures) == 1
    assert failures[0].name == bad_dir.name


def test_scan_run_logs_dir(tmp_path: Path) -> None:
    ok_dir = tmp_path / "20251015_000300"
    ok_dir.mkdir()
    (ok_dir / "run.log").write_text("INFO: success\n", encoding="utf-8")

    fail_dir = tmp_path / "20251015_000400"
    fail_dir.mkdir()
    (fail_dir / "run.log").write_text("Failed to export model\n", encoding="utf-8")

    failures = scan_run_logs_dir(tmp_path)

    assert {failure.name for failure in failures} == {fail_dir.name}
