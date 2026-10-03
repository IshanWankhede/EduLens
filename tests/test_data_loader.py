from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from src.data_loader import (
    COURSE_MATCH_COLUMNS,
    inspect_course_overlap,
    load_uci_dataset,
    load_uploaded_csv,
)
from src.preprocessing import derive_performance_category
from src.validation import InsufficientDataError


def test_primary_portuguese_dataset_loads_with_verified_facts() -> None:
    bundle = load_uci_dataset()
    categories = derive_performance_category(bundle.raw)

    assert bundle.metadata.file_name == "student-por.csv"
    assert bundle.metadata.course == "Portuguese"
    assert bundle.metadata.encoding == "utf-8"
    assert bundle.metadata.delimiter == ";"
    assert bundle.raw.shape == (649, 33)
    assert bundle.clean.shape == (649, 34)
    assert bundle.clean["G3_zero_flag"].sum() == 15
    assert bundle.raw.isna().sum().sum() == 0
    assert bundle.raw.duplicated().sum() == 0
    assert (bundle.raw["G3"].min(), bundle.raw["G3"].max()) == (0, 19)
    assert (bundle.raw["absences"].min(), bundle.raw["absences"].max()) == (0, 32)
    assert categories.class_counts == {"Low": 100, "Medium": 355, "High": 194}


def test_mathematics_dataset_loads_separately_with_verified_facts() -> None:
    bundle = load_uci_dataset("mat")
    categories = derive_performance_category(bundle.raw)

    assert bundle.metadata.file_name == "student-mat.csv"
    assert bundle.metadata.course == "Mathematics"
    assert bundle.raw.shape == (395, 33)
    assert bundle.clean.shape == (395, 34)
    assert bundle.clean["G3_zero_flag"].sum() == 38
    assert bundle.raw.isna().sum().sum() == 0
    assert bundle.raw.duplicated().sum() == 0
    assert (bundle.raw["G3"].min(), bundle.raw["G3"].max()) == (0, 20)
    assert (bundle.raw["absences"].min(), bundle.raw["absences"].max()) == (0, 75)
    assert categories.class_counts == {"Low": 130, "Medium": 165, "High": 100}


def test_overlap_uses_r_script_key_and_reports_key_ambiguity() -> None:
    math = load_uci_dataset("mat").raw
    portuguese = load_uci_dataset("por").raw

    report = inspect_course_overlap(math, portuguese)

    assert report.match_columns == COURSE_MATCH_COLUMNS
    assert report.common_key_groups == 366
    assert report.joined_rows == 382
    assert report.math_rows_with_common_key == 370
    assert report.portuguese_rows_with_common_key == 374
    assert report.one_to_one_key_groups == 358
    assert report.non_unique_key_groups == 8


def test_uploaded_csv_is_parsed_in_memory_and_has_no_uci_schema_requirement() -> None:
    content = ("name,score\n" + "\n".join(f"student-{n},{n}" for n in range(20))).encode()

    bundle = load_uploaded_csv(content, "../local-name.csv")

    assert bundle.metadata.source == "user upload"
    assert bundle.metadata.file_name == "local-name.csv"
    assert bundle.metadata.row_count == 20
    assert bundle.metadata.column_count == 2
    assert bundle.raw.equals(pd.read_csv(BytesIO(content)))
    assert bundle.clean.columns.tolist() == ["name", "score"]


def test_uploaded_csv_enforces_default_minimum_rows() -> None:
    content = ("score\n" + "\n".join(str(value) for value in range(19))).encode()

    with pytest.raises(InsufficientDataError, match="at least 20"):
        load_uploaded_csv(content)
