from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.assumptions import (
    AssumptionCheckError,
    expected_count_check,
    levene_check,
    minimum_group_size_check,
    shapiro_by_group,
    shapiro_check,
    vif_check,
)


def test_shapiro_and_levene_checks_cross_check_scipy() -> None:
    first = np.array([1.0, 2.0, 3.0, 5.0, 7.0])
    second = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    shapiro = shapiro_check(first)
    by_group = shapiro_by_group({"first": first, "second": second})
    levene = levene_check([first, second])

    expected_shapiro = stats.shapiro(first)
    expected_levene = stats.levene(first, second, center="median")
    assert shapiro.statistic == pytest.approx(expected_shapiro.statistic)
    assert shapiro.p_value == pytest.approx(expected_shapiro.pvalue)
    assert set(by_group) == {"first", "second"}
    assert levene.statistic == pytest.approx(expected_levene.statistic)
    assert levene.p_value == pytest.approx(expected_levene.pvalue)
    assert shapiro.status in {"ok", "warning"}


def test_expected_counts_and_minimum_group_size_return_explanations() -> None:
    expected = expected_count_check(np.array([[10, 12], [11, 9]]))
    sparse = expected_count_check(np.array([[0.5, 9.5], [4.5, 5.5]]))
    groups = minimum_group_size_check({"a": 4, "b": 1}, minimum=2)

    assert expected.status == "ok"
    assert "guideline is met" in expected.message
    assert sparse.status == "warning"
    assert "guideline is not met" in sparse.message
    assert groups.status == "warning"
    assert "b=1" in groups.message


def test_vif_matches_independent_predictors_and_reports_collinearity() -> None:
    data = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 4.0, 7.0, 8.0, 11.0],
            "x2": [3.0, 9.0, 2.0, 8.0, 5.0, 10.0],
        }
    )
    result = vif_check(data)
    assert set(result.table["predictor"]) == {"x1", "x2"}
    assert result.status == "ok"
    correlation = np.corrcoef(data["x1"], data["x2"])[0, 1]
    expected_vif = 1 / (1 - correlation**2)
    assert result.table["vif"].tolist() == pytest.approx([expected_vif, expected_vif])

    collinear = pd.DataFrame({"x1": [1, 2, 3, 4], "x2": [2, 4, 6, 8]})
    with pytest.raises(AssumptionCheckError, match="linearly dependent"):
        vif_check(collinear)


def test_assumption_checks_handle_tiny_samples_and_invalid_inputs() -> None:
    tiny = shapiro_check([1.0, 2.0])
    small_group = levene_check([[1.0], [2.0, 3.0]])

    assert tiny.status == "warning"
    assert "fewer than three" in tiny.message
    assert small_group.status == "warning"
    with pytest.raises(AssumptionCheckError, match="finite"):
        shapiro_check([np.nan])
    with pytest.raises(AssumptionCheckError, match="two-dimensional"):
        expected_count_check([1, 2, 3])
    with pytest.raises(AssumptionCheckError, match="two predictors"):
        vif_check(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
