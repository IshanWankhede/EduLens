from __future__ import annotations

import pandas as pd
import pytest

from src.validation import (
    DataValidationError,
    InsufficientDataError,
    parse_csv,
    validate_dataframe,
)


def test_parse_csv_detects_semicolon_and_utf8() -> None:
    parsed = parse_csv(
        b"name;score\nAna;10\nLuis;12\n",
        min_rows=2,
    )

    assert parsed.delimiter == ";"
    assert parsed.encoding == "utf-8"
    assert parsed.dataframe.to_dict("records") == [
        {"name": "Ana", "score": 10},
        {"name": "Luis", "score": 12},
    ]


def test_parse_csv_supports_windows_1252_fallback() -> None:
    parsed = parse_csv(
        "name,city\nJosé,São Paulo\nAna,Porto\n".encode("cp1252"),
        min_rows=2,
    )

    assert parsed.encoding == "cp1252"
    assert parsed.delimiter == ","
    assert parsed.dataframe.loc[0, "name"] == "José"


@pytest.mark.parametrize("content", [b"", b"  \n", b"\x00\x01"])
def test_parse_csv_rejects_empty_or_binary_input(content: bytes) -> None:
    with pytest.raises(DataValidationError):
        parse_csv(content, min_rows=1)


def test_parse_csv_rejects_wrong_expected_delimiter() -> None:
    with pytest.raises(DataValidationError, match="Expected a"):
        parse_csv(
            b"one;two\n1;2\n",
            min_rows=1,
            expected_delimiter=",",
        )


def test_parse_csv_rejects_single_column_wrong_delimiter() -> None:
    with pytest.raises(DataValidationError, match="delimited"):
        parse_csv(b"one;two\n1;2\n", min_rows=1, expected_delimiter=",")


def test_dataframe_validation_checks_minimum_required_columns_and_numeric_types() -> None:
    frame = pd.DataFrame({"G3": [10, 11], "age": ["16", "invalid"]})

    with pytest.raises(InsufficientDataError, match="at least 3"):
        validate_dataframe(frame, min_rows=3)
    with pytest.raises(DataValidationError, match="Required column"):
        validate_dataframe(frame, min_rows=2, required_columns=("student_id",))
    with pytest.raises(DataValidationError, match="must contain numeric"):
        validate_dataframe(frame, min_rows=2, numeric_columns=("age",))


def test_upload_minimum_defaults_to_twenty_rows() -> None:
    records = "score\n" + "\n".join(str(value) for value in range(19)) + "\n"

    with pytest.raises(InsufficientDataError, match="at least 20"):
        parse_csv(records.encode("utf-8"))
