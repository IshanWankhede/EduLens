from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.data_loader import load_uci_dataset
from src.descriptive_stats import (
    AllNaNColumnError,
    ConstantColumnError,
    InsufficientGroupSizeError,
    NonNumericColumnError,
    SingleObservationError,
    frequency_table,
    group_summary,
    summary_table,
)


def test_summary_table_matches_hand_calculated_fixture() -> None:
    # Observed values are [1, 2, 2, 5]: mean=2.5, sample variance=3,
    # Q1=1.75, median=2, Q3=2.75, and IQR=1.
    data = pd.DataFrame({"score": [1.0, 2.0, 2.0, 5.0, np.nan]})

    result = summary_table(data, ["score"])
    row = result.table.loc["score"]

    assert row["count"] == 4
    assert row["missing_count"] == 1
    assert row["mean"] == 2.5
    assert row["median"] == 2.0
    assert row["mode"] == (2.0,)
    assert row["minimum"] == 1.0
    assert row["maximum"] == 5.0
    assert row["range"] == 4.0
    assert row["variance"] == 3.0
    assert row["standard_deviation"] == pytest.approx(math.sqrt(3))
    assert row["Q1"] == 1.75
    assert row["Q2"] == 2.0
    assert row["Q3"] == 2.75
    assert row["IQR"] == 1.0
    assert row["skewness"] == pytest.approx(stats.skew([1, 2, 2, 5], bias=False))
    assert result.measurement_levels == {"score": "numeric"}
    assert "linear" in result.quartile_method
    assert "Fisher-Pearson" in result.skewness_method


def test_summary_table_reports_all_tied_modes() -> None:
    result = summary_table(pd.DataFrame({"studytime": [1, 1, 2, 2]}), ["studytime"])

    assert result.table.loc["studytime", "mode"] == (1, 2)
    assert result.measurement_levels["studytime"] == "ordinal"


def test_summary_table_cross_checks_numpy_and_scipy() -> None:
    values = np.array([3.0, 5.0, 8.0, 10.0, 12.0, 15.0])
    result = summary_table(pd.DataFrame({"value": values}), ["value"])
    row = result.table.loc["value"]

    assert row["mean"] == pytest.approx(np.mean(values))
    assert row["variance"] == pytest.approx(np.var(values, ddof=1))
    assert row["standard_deviation"] == pytest.approx(np.std(values, ddof=1))
    assert row["skewness"] == pytest.approx(stats.skew(values, bias=False))


def test_summary_table_raises_friendly_custom_edge_case_errors() -> None:
    with pytest.raises(ConstantColumnError, match="constant"):
        summary_table(pd.DataFrame({"value": [4, 4, 4]}), ["value"])
    with pytest.raises(SingleObservationError, match="only one"):
        summary_table(pd.DataFrame({"value": [4.0, np.nan]}), ["value"])
    with pytest.raises(AllNaNColumnError, match="no valid"):
        summary_table(pd.DataFrame({"value": [np.nan, np.nan]}), ["value"])
    with pytest.raises(NonNumericColumnError, match="not numeric"):
        summary_table(pd.DataFrame({"value": ["a", "b"]}), ["value"])


def test_group_summary_has_hand_calculated_statistics_and_t_confidence_intervals() -> None:
    # Group A values [1, 2, 3] have mean 2 and SD 1.
    # Group B values [2, 3, 4] have mean 3 and SD 1.
    data = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B"],
            "studytime": [1, 2, 3, 2, 3, 4],
        }
    )
    result = group_summary(data, "studytime", "group", confidence_level=0.95)
    table = result.table.set_index("group")
    critical = stats.t.ppf(0.975, df=2)

    assert table.loc["A", "group_size"] == 3
    assert table.loc["A", "count"] == 3
    assert table.loc["A", "mean"] == 2
    assert table.loc["A", "standard_deviation"] == 1
    assert table.loc["A", "mean_ci_lower"] == pytest.approx(2 - critical / math.sqrt(3))
    assert table.loc["A", "mean_ci_upper"] == pytest.approx(2 + critical / math.sqrt(3))
    assert table.loc["B", "mean"] == 3
    assert table.loc["B", "variance"] == 1
    assert result.confidence_level == 0.95
    assert result.measurement_level == "ordinal"
    assert result.missing_group_rows == 0


def test_group_summary_cross_checks_confidence_interval_with_scipy() -> None:
    data = pd.DataFrame(
        {
            "group": ["one"] * 5,
            "value": [2.0, 3.0, 5.0, 8.0, 12.0],
        }
    )
    result = group_summary(data, "value", "group", confidence_level=0.90)
    row = result.table.iloc[0]
    expected = stats.t.interval(
        0.90,
        df=4,
        loc=np.mean(data["value"]),
        scale=stats.sem(data["value"]),
    )

    assert row["mean_ci_lower"] == pytest.approx(expected[0])
    assert row["mean_ci_upper"] == pytest.approx(expected[1])


def test_group_summary_rejects_small_groups_and_invalid_confidence_levels() -> None:
    data = pd.DataFrame({"group": ["large", "large", "small"], "value": [1.0, 2.0, 4.0]})

    with pytest.raises(InsufficientGroupSizeError, match="small"):
        group_summary(data, "value", "group")
    with pytest.raises(ValueError, match="confidence_level"):
        group_summary(data.iloc[:2], "value", "group", confidence_level=1.0)


def test_group_summary_counts_missing_values_and_missing_group_labels() -> None:
    data = pd.DataFrame(
        {
            "group": ["a", "a", "a", "b", "b", None],
            "value": [1.0, 2.0, np.nan, 4.0, 6.0, 9.0],
        }
    )
    result = group_summary(data, "value", "group")
    table = result.table.set_index("group")

    assert table.loc["a", "group_size"] == 3
    assert table.loc["a", "count"] == 2
    assert table.loc["a", "missing_count"] == 1
    assert result.missing_group_rows == 1


def test_frequency_table_has_hand_calculated_counts_percentages_and_ordinal_order() -> None:
    # The three non-missing observations are 1, 2, 2, giving 1/3 and 2/3.
    data = pd.DataFrame({"studytime": [2, 1, 2, None]})

    result = frequency_table(data, "studytime")

    assert result.table["category"].tolist() == [1.0, 2.0]
    assert result.table["count"].tolist() == [1, 2]
    assert result.table["percentage"].tolist() == pytest.approx([100 / 3, 200 / 3])
    assert result.total_count == 4
    assert result.valid_count == 3
    assert result.missing_count == 1
    assert result.measurement_level == "ordinal"


def test_frequency_table_preserves_declared_ordered_categorical_order() -> None:
    series = pd.Series(
        pd.Categorical(
            ["medium", "low", "medium"],
            categories=["low", "medium", "high"],
            ordered=True,
        )
    )

    result = frequency_table(pd.DataFrame({"rating": series}), "rating")

    assert result.table["category"].tolist() == ["low", "medium", "high"]
    assert result.table["count"].tolist() == [1, 2, 0]
    assert result.table["percentage"].tolist() == pytest.approx([100 / 3, 200 / 3, 0])
    assert result.measurement_level == "ordinal"


def test_real_portuguese_data_stats_use_phase_two_cleaned_data_pipeline() -> None:
    bundle = load_uci_dataset("por")
    result = summary_table(bundle.clean, ["G3", "absences"])
    frequencies = frequency_table(bundle.clean, "studytime")

    assert bundle.metadata.row_count == 649
    assert result.table["count"].to_dict() == {"G3": 649, "absences": 649}
    assert result.table["missing_count"].to_dict() == {"G3": 0, "absences": 0}
    assert frequencies.valid_count == 649
    assert frequencies.missing_count == 0
