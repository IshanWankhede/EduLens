"""Explicit command-line export of the cleaned Portuguese course dataset."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from src.data_loader import load_uci_dataset
from src.validation import DataValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "student-por-cleaned.csv"


@dataclass(frozen=True)
class CleanedExportResult:
    """Location and row-level details of a completed cleaned-data export."""

    output_path: Path
    row_count: int
    column_count: int
    g3_zero_count: int


def export_cleaned_dataset(
    *,
    raw_dir: Path | None = None,
    output_path: Path | None = None,
) -> CleanedExportResult:
    """Export the Phase 2 cleaned Portuguese data without modifying the raw source file.

    Loading and cleaning are delegated to ``load_uci_dataset``. Parent directories are created
    for the explicitly requested output location; CSV rows preserve the cleaned frame's order and
    columns, including ``G3_zero_flag``.
    """
    bundle = load_uci_dataset("por", data_dir=raw_dir)
    destination = output_path or DEFAULT_OUTPUT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle.clean.to_csv(destination, index=False, encoding="utf-8", lineterminator="\n")
    return CleanedExportResult(
        output_path=destination.resolve(),
        row_count=len(bundle.clean),
        column_count=len(bundle.clean.columns),
        g3_zero_count=int(bundle.clean["G3_zero_flag"].fillna(False).sum()),
    )


def main() -> int:
    """Run the export command and report friendly errors without a traceback."""
    try:
        result = export_cleaned_dataset(
            raw_dir=DEFAULT_RAW_DIR,
            output_path=DEFAULT_OUTPUT_PATH,
        )
    except (DataValidationError, OSError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1

    print(f"Output: {result.output_path}")
    print(f"Rows: {result.row_count}")
    print(f"Columns: {result.column_count}")
    print(f"G3_zero_flag count: {result.g3_zero_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
