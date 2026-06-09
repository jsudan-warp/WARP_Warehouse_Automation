"""Append measured numbers to RESULTS.md (the project's running ledger)."""
from __future__ import annotations

from pathlib import Path


def append_section(results_path: Path, title: str, body: str) -> None:
    """Append a titled section. Creates the file with a header if missing."""
    header = "# RESULTS — LAX-9 Stage 1\n\nMeasured numbers per phase (see PLAN.md acceptance checks).\n"
    if not results_path.exists():
        results_path.write_text(header)
    with open(results_path, "a") as f:
        f.write(f"\n## {title}\n\n{body.rstrip()}\n")
