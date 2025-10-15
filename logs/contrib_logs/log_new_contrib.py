#!/usr/bin/env python3
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    """Append a new contrib log entry stub using the current UTC timestamp."""
    now = datetime.now(timezone.utc)
    date_heading = now.strftime("%Y-%m-%d")
    time_heading = now.strftime("%Y-%m-%d %H:%M UTC")

    log_dir = Path(__file__).resolve().parent
    log_dir.mkdir(parents=True, exist_ok=True)

    monthly_filename = now.strftime("%y%m.md")
    log_file = log_dir / monthly_filename
    log_file.touch(exist_ok=True)

    existing_text = log_file.read_text(encoding="utf-8")
    has_day_heading = any(
        line.strip() == f"## {date_heading}" for line in existing_text.splitlines()
    )

    append_chunks: list[str] = []
    if existing_text and not existing_text.endswith("\n"):
        append_chunks.append("\n")

    if not has_day_heading:
        if existing_text.strip():
            append_chunks.append("\n")
        append_chunks.append(f"## {date_heading}\n\n")
    elif existing_text.strip():
        append_chunks.append("\n")

    append_chunks.append(f"### {time_heading}\n")
    append_chunks.append("- Summary: \n")
    append_chunks.append("- Files: \n")

    with log_file.open("a", encoding="utf-8") as log_handle:
        log_handle.write("".join(append_chunks))

    display_path = os.path.relpath(log_file, Path.cwd())
    print(f"Added contrib log stub at {time_heading} in {display_path}")


if __name__ == "__main__":
    main()
