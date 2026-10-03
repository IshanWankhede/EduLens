"""Tests for explicitly exporting the Phase 2 cleaned Portuguese dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src import export_cleaned
from src.data_loader import load_uci_dataset


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_matches_cleaned_data_and_includes_g3_zero_flag(tmp_path: Path) -> None:
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    expected = load_uci_dataset("por", data_dir=raw_dir)
    output_path = tmp_path / "processed" / "student-por-cleaned.csv"

    result = export_cleaned.export_cleaned_dataset(raw_dir=raw_dir, output_path=output_path)
    exported = pd.read_csv(output_path)

    assert result.row_count == 649
    assert result.column_count == len(expected.clean.columns)
    assert result.g3_zero_count == 15
    assert exported.shape == (649, len(expected.clean.columns))
    assert exported.columns.tolist() == expected.clean.columns.tolist()
    assert "G3_zero_flag" in exported.columns
    assert int(exported["G3_zero_flag"].sum()) == 15


def test_export_overwrites_deterministically(tmp_path: Path) -> None:
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    output_path = tmp_path / "processed" / "student-por-cleaned.csv"
    first = export_cleaned.export_cleaned_dataset(raw_dir=raw_dir, output_path=output_path)
    first_content = output_path.read_bytes()
    output_path.write_text("existing output must be replaced\n", encoding="utf-8")

    second = export_cleaned.export_cleaned_dataset(raw_dir=raw_dir, output_path=output_path)

    assert second == first
    assert output_path.read_bytes() == first_content


def test_export_does_not_modify_raw_file(tmp_path: Path) -> None:
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    raw_path = raw_dir / "student-por.csv"
    original_digest = _sha256(raw_path)

    export_cleaned.export_cleaned_dataset(
        raw_dir=raw_dir,
        output_path=tmp_path / "student-por-cleaned.csv",
    )

    assert _sha256(raw_path) == original_digest


def test_cli_reports_missing_raw_file_without_traceback(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    missing_raw_dir = tmp_path / "missing-raw"
    monkeypatch.setattr(export_cleaned, "DEFAULT_RAW_DIR", missing_raw_dir)
    monkeypatch.setattr(
        export_cleaned,
        "DEFAULT_OUTPUT_PATH",
        tmp_path / "processed" / "student-por-cleaned.csv",
    )

    exit_code = export_cleaned.main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Export failed:" in captured.err
    assert "student-por.csv" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "processed" / "student-por-cleaned.csv").exists()
