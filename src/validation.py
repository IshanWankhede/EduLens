"""Validation helpers for local and uploaded CSV datasets."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from io import StringIO

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


class DataValidationError(ValueError):
    """Raised when tabular input is unreadable or does not match its expected schema."""


class InsufficientDataError(DataValidationError):
    """Raised when a valid table contains fewer rows than the configured minimum."""


@dataclass(frozen=True)
class ParsedCsv:
    """CSV parsing result with its detected source format."""

    dataframe: pd.DataFrame
    encoding: str
    delimiter: str


def _decode_csv(content: bytes) -> tuple[str, str]:
    """Decode CSV bytes using UTF-8 first and Windows-1252 as a documented fallback."""
    if not content or not content.strip():
        raise DataValidationError("The CSV file is empty.")
    if b"\x00" in content:
        raise DataValidationError("The file contains binary data and is not a supported text CSV.")

    if content.startswith(b"\xef\xbb\xbf"):
        try:
            return content.decode("utf-8-sig"), "utf-8-sig"
        except UnicodeDecodeError as exc:
            raise DataValidationError("The CSV encoding is not valid UTF-8.") from exc

    try:
        return content.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        try:
            return content.decode("cp1252"), "cp1252"
        except UnicodeDecodeError as exc:
            raise DataValidationError(
                "The CSV encoding is unsupported. Save the file as UTF-8 and try again."
            ) from exc


def _detect_delimiter(text: str) -> str:
    """Detect a supported delimiter and reject a likely wrong-delimiter single column."""
    sample = "\n".join(line for line in text.splitlines()[:20] if line.strip())
    if not sample:
        raise DataValidationError("The CSV file does not contain a header row.")

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        header = sample.splitlines()[0]
        delimiter = max((";", ",", "\t", "|"), key=header.count)
        if header.count(delimiter) == 0:
            return ","

    try:
        headers = next(csv.reader([sample.splitlines()[0]], delimiter=delimiter))
    except (csv.Error, StopIteration) as exc:
        raise DataValidationError("The CSV header could not be parsed.") from exc
    if len(headers) < 2:
        raise DataValidationError(
            "The CSV appears to use an unsupported delimiter or has only one column."
        )
    if any(not header.strip() for header in headers):
        raise DataValidationError("Every CSV column must have a non-empty header.")
    if len(headers) != len(set(headers)):
        raise DataValidationError("The CSV contains duplicate column names.")
    return delimiter


def validate_dataframe(
    dataframe: pd.DataFrame,
    *,
    min_rows: int = 20,
    required_columns: Sequence[str] = (),
    numeric_columns: Sequence[str] = (),
) -> None:
    """Validate table shape, required columns, minimum rows, and specified numeric columns.

    Numeric checks require non-missing values to be parseable as numbers; this function never
    coerces or mutates the supplied DataFrame.
    """
    if min_rows < 1:
        raise ValueError("min_rows must be at least 1.")
    if dataframe.empty or dataframe.shape[1] == 0:
        raise DataValidationError("The dataset has no rows or columns.")
    if len(dataframe) < min_rows:
        raise InsufficientDataError(
            f"The dataset has {len(dataframe)} rows; at least {min_rows} are required."
        )

    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        names = ", ".join(missing_columns)
        raise DataValidationError(f"Required column(s) missing: {names}.")

    for column in numeric_columns:
        if column not in dataframe.columns:
            continue
        values = dataframe[column]
        converted = pd.to_numeric(values, errors="coerce")
        invalid = values.notna() & converted.isna()
        if invalid.any():
            raise DataValidationError(
                f"Column '{column}' must contain numeric values; "
                f"{int(invalid.sum())} value(s) could not be parsed."
            )


def parse_csv(
    content: bytes,
    *,
    min_rows: int = 20,
    required_columns: Sequence[str] = (),
    numeric_columns: Sequence[str] = (),
    expected_delimiter: str | None = None,
) -> ParsedCsv:
    """Decode, parse, and validate CSV bytes without writing the uploaded data to disk."""
    text, encoding = _decode_csv(content)
    detected_delimiter = _detect_delimiter(text)
    delimiter = expected_delimiter or detected_delimiter
    if delimiter not in {";", ",", "\t", "|"}:
        raise ValueError("expected_delimiter must be comma, semicolon, tab, or pipe.")
    if expected_delimiter is not None and detected_delimiter != expected_delimiter:
        raise DataValidationError(
            f"Expected a {expected_delimiter!r}-delimited CSV, but detected "
            f"{detected_delimiter!r}."
        )

    try:
        dataframe = pd.read_csv(StringIO(text), sep=delimiter)
    except (EmptyDataError, ParserError, UnicodeDecodeError, csv.Error) as exc:
        raise DataValidationError(
            "The CSV could not be parsed. Check its encoding, delimiter, and row formatting."
        ) from exc

    validate_dataframe(
        dataframe,
        min_rows=min_rows,
        required_columns=required_columns,
        numeric_columns=numeric_columns,
    )
    return ParsedCsv(dataframe=dataframe, encoding=encoding, delimiter=delimiter)
