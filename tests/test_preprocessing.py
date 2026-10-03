from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import (
    build_cleaning_report,
    clean_dataset,
    derive_performance_category,
    encode_binary,
    encode_nominal,
)
from src.validation import DataValidationError


def test_cleaning_report_detects_quality_issues_without_changing_rows() -> None:
    frame = pd.DataFrame(
        {
            "score": [1, 2, 2, 2, 100],
            "note": ["a", "b", "b", "b", "z"],
        }
    )

    cleaned, report = clean_dataset(frame)

    assert len(cleaned) == len(frame)
    assert report.rows_before == report.rows_after == 5
    assert report.columns_before == 2
    assert report.columns_after == 2
    assert report.missing_cells == 0
    assert report.exact_duplicate_rows == 2
    assert report.outliers["score"].count == 2
    assert report.outliers["score"].lower_fence == 2
    assert report.outliers["score"].upper_fence == 2
    assert report.numeric_summary_before["score"]["mean"] == 21.4
    pd.testing.assert_frame_equal(
        frame,
        pd.DataFrame(
            {
                "score": [1, 2, 2, 2, 100],
                "note": ["a", "b", "b", "b", "z"],
            }
        ),
    )


def test_cleaning_adds_g3_zero_flag_and_preserves_grade_values() -> None:
    frame = pd.DataFrame({"G3": [0, 10, None]})

    cleaned, report = clean_dataset(frame)

    assert cleaned["G3"].tolist()[:2] == [0.0, 10.0]
    assert cleaned["G3_zero_flag"].tolist()[:2] == [True, False]
    assert pd.isna(cleaned["G3_zero_flag"].iloc[2])
    assert report.columns_before == 1
    assert report.columns_after == 2
    assert "G3_zero_flag" not in frame.columns


def test_cleaning_report_counts_missing_values_and_numeric_summaries() -> None:
    frame = pd.DataFrame({"value": [1.0, None, 3.0], "label": ["x", None, "z"]})

    report = build_cleaning_report(frame)

    assert report.missing_cells == 2
    assert report.missing_by_column == {"value": 1, "label": 1}
    assert report.numeric_summary_before["value"]["count"] == 2
    assert report.numeric_summary_before["value"]["mean"] == 2.0


def test_encode_binary_maps_positive_category_to_one_and_preserves_missing() -> None:
    result = encode_binary(pd.Series(["yes", "no", None]), positive_label="YES")

    assert result.iloc[0] == 1
    assert result.iloc[1] == 0
    assert pd.isna(result.iloc[2])


def test_encode_binary_rejects_non_binary_columns_and_unknown_positive_label() -> None:
    with pytest.raises(DataValidationError, match="exactly two"):
        encode_binary(pd.Series(["yes", "no", "maybe"]), positive_label="yes")
    with pytest.raises(DataValidationError, match="not one"):
        encode_binary(pd.Series(["yes", "no"]), positive_label="true")


def test_encode_nominal_uses_sorted_reference_and_does_not_mutate_input() -> None:
    frame = pd.DataFrame({"job": ["teacher", "other", "health", None], "value": [1, 2, 3, 4]})

    encoded, references = encode_nominal(frame, ["job"])

    assert references == {"job": "health"}
    assert "job" not in encoded.columns
    assert list(encoded.columns) == ["value", "job_other", "job_teacher"]
    assert encoded.loc[1, ["job_other", "job_teacher"]].tolist() == [1, 0]
    assert encoded.loc[3, ["job_other", "job_teacher"]].tolist() == [0, 0]
    assert frame.columns.tolist() == ["job", "value"]


def test_derive_fixed_grade_bands_assigns_boundaries_and_counts_missing() -> None:
    result = derive_performance_category(pd.DataFrame({"G3": [0, 9, 10, 13, 14, 20, None]}))

    assert result.categories.tolist() == [
        "Low",
        "Low",
        "Medium",
        "Medium",
        "High",
        "High",
        pd.NA,
    ]
    assert result.method == "fixed grade bands"
    assert result.class_counts == {"Low": 2, "Medium": 2, "High": 2}
    assert result.unclassified_count == 1


def test_derive_performance_category_reports_invalid_grade_values() -> None:
    with pytest.raises(DataValidationError, match="non-numeric"):
        derive_performance_category(pd.DataFrame({"G3": ["unknown"]}))


def test_derive_performance_category_accepts_valid_explicit_thresholds() -> None:
    thresholds = {
        "LOW_UPPER_EXCLUSIVE": 8,
        "MEDIUM_LOWER_INCLUSIVE": 8,
        "MEDIUM_UPPER_INCLUSIVE": 15,
        "HIGH_LOWER_INCLUSIVE": 16,
    }

    result = derive_performance_category(
        pd.DataFrame({"G3": [7, 8, 15, 16]}),
        thresholds=thresholds,
    )

    assert result.categories.tolist() == ["Low", "Medium", "Medium", "High"]
    assert result.class_counts == {"Low": 1, "Medium": 2, "High": 1}


def test_derive_performance_category_rejects_non_contiguous_thresholds() -> None:
    thresholds = {
        "LOW_UPPER_EXCLUSIVE": 8,
        "MEDIUM_LOWER_INCLUSIVE": 9,
        "MEDIUM_UPPER_INCLUSIVE": 15,
        "HIGH_LOWER_INCLUSIVE": 16,
    }

    with pytest.raises(DataValidationError, match="contiguous"):
        derive_performance_category(pd.DataFrame({"G3": [10]}), thresholds=thresholds)
