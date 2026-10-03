"""Dataset loading and metadata containers; uploaded data is handled in memory only."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from src import config
from src.preprocessing import CleaningReport, clean_dataset
from src.validation import DataValidationError, parse_csv

COURSE_MATCH_COLUMNS = (
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "nursery",
    "internet",
)


@dataclass(frozen=True)
class DatasetMetadata:
    """Source identity and CSV facts about a loaded table."""

    source: str
    file_name: str
    encoding: str
    delimiter: str
    row_count: int
    column_count: int
    course: str | None = None


@dataclass(frozen=True)
class DatasetBundle:
    """Raw and non-destructively cleaned frames with schema metadata and a quality report."""

    raw: pd.DataFrame
    clean: pd.DataFrame
    metadata: DatasetMetadata
    column_roles: Mapping[str, list[str]]
    cleaning_report: CleaningReport


@dataclass(frozen=True)
class CourseOverlapReport:
    """Summary of profile-key matches; not a count of uniquely identified people."""

    match_columns: tuple[str, ...]
    common_key_groups: int
    joined_rows: int
    math_rows_with_common_key: int
    portuguese_rows_with_common_key: int
    one_to_one_key_groups: int
    non_unique_key_groups: int


def _column_roles(dataframe: pd.DataFrame) -> dict[str, list[str]]:
    """Return recognized column-role lists restricted to columns present in this dataset."""
    return {
        "numeric": [column for column in config.NUMERIC_COLUMNS if column in dataframe.columns],
        "ordinal": [column for column in config.ORDINAL_COLUMNS if column in dataframe.columns],
        "nominal": [column for column in config.NOMINAL_COLUMNS if column in dataframe.columns],
        "binary": [column for column in config.BINARY_COLUMNS if column in dataframe.columns],
    }


def _bundle(
    dataframe: pd.DataFrame,
    metadata: DatasetMetadata,
) -> DatasetBundle:
    """Build the common bundle while preserving all input observations."""
    clean, report = clean_dataset(dataframe)
    return DatasetBundle(
        raw=dataframe.copy(),
        clean=clean,
        metadata=metadata,
        column_roles=_column_roles(dataframe),
        cleaning_report=report,
    )


def load_uci_dataset(
    course: Literal["por", "mat"] = "por",
    *,
    data_dir: Path | None = None,
) -> DatasetBundle:
    """Load one local UCI course file without combining it with the other course.

    UCI files are expected to be UTF-8, semicolon-delimited, contain the documented 33 columns,
    and have at least 20 rows. The course file is validated but never modified on disk.
    """
    files = {"por": "student-por.csv", "mat": "student-mat.csv"}
    if course not in files:
        raise ValueError("course must be 'por' or 'mat'.")
    raw_dir = data_dir or Path(__file__).resolve().parents[1] / "data" / "raw"
    path = raw_dir / files[course]
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise DataValidationError(
            f"Could not read '{path.name}' from data/raw. Place the UCI course file there and try again."
        ) from exc

    expected_columns = (
        config.NUMERIC_COLUMNS
        + config.ORDINAL_COLUMNS
        + config.NOMINAL_COLUMNS
        + config.BINARY_COLUMNS
    )
    parsed = parse_csv(
        content,
        min_rows=20,
        required_columns=expected_columns,
        numeric_columns=config.NUMERIC_COLUMNS + config.ORDINAL_COLUMNS,
        expected_delimiter=";",
    )
    metadata = DatasetMetadata(
        source="UCI Student Performance",
        file_name=path.name,
        encoding=parsed.encoding,
        delimiter=parsed.delimiter,
        row_count=len(parsed.dataframe),
        column_count=len(parsed.dataframe.columns),
        course="Portuguese" if course == "por" else "Mathematics",
    )
    return _bundle(parsed.dataframe, metadata)


def load_uploaded_csv(
    content: bytes,
    file_name: str = "uploaded.csv",
    *,
    min_rows: int = 20,
) -> DatasetBundle:
    """Load uploaded CSV bytes in memory and return their bundle without persisting the upload."""
    parsed = parse_csv(content, min_rows=min_rows)
    metadata = DatasetMetadata(
        source="user upload",
        file_name=Path(file_name).name,
        encoding=parsed.encoding,
        delimiter=parsed.delimiter,
        row_count=len(parsed.dataframe),
        column_count=len(parsed.dataframe.columns),
    )
    return _bundle(parsed.dataframe, metadata)


def inspect_course_overlap(
    math_dataframe: pd.DataFrame,
    portuguese_dataframe: pd.DataFrame,
) -> CourseOverlapReport:
    """Count shared profile-key groups following student-merge.R without merging course data.

    The key fields are not guaranteed unique identifiers. Join-row counts therefore describe
    matching profiles, not verified distinct people.
    """
    missing = [
        column
        for column in COURSE_MATCH_COLUMNS
        if column not in math_dataframe.columns or column not in portuguese_dataframe.columns
    ]
    if missing:
        raise ValueError(f"Course overlap key column(s) missing: {', '.join(missing)}.")

    math_counts = math_dataframe.groupby(list(COURSE_MATCH_COLUMNS), dropna=False).size()
    portuguese_counts = portuguese_dataframe.groupby(
        list(COURSE_MATCH_COLUMNS), dropna=False
    ).size()
    common = math_counts.to_frame("math_count").join(
        portuguese_counts.to_frame("portuguese_count"), how="inner"
    )
    merged_rows = math_dataframe.merge(
        portuguese_dataframe,
        on=list(COURSE_MATCH_COLUMNS),
        how="inner",
        suffixes=("_math", "_portuguese"),
    )
    one_to_one = common["math_count"].eq(1) & common["portuguese_count"].eq(1)
    return CourseOverlapReport(
        match_columns=COURSE_MATCH_COLUMNS,
        common_key_groups=len(common),
        joined_rows=len(merged_rows),
        math_rows_with_common_key=int(common["math_count"].sum()),
        portuguese_rows_with_common_key=int(common["portuguese_count"].sum()),
        one_to_one_key_groups=int(one_to_one.sum()),
        non_unique_key_groups=int((~one_to_one).sum()),
    )
