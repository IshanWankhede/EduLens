from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.correlation import (
    CorrelationAnalysisError,
    correlation_matrices,
    pearson_matrix,
    ranked_associations,
    recommended_method,
    spearman_matrix,
    strength_label,
)


def test_correlation_matrices_match_scipy_and_fisher_z_interval() -> None:
    # x increases linearly; y is a monotone nonlinear transform, so both coefficients equal 1.
    data = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 4.0, 9.0, 16.0, 25.0],
        }
    )
    result = correlation_matrices(data, ["x", "y"], confidence_level=0.95)
    pearson = stats.pearsonr(data["x"], data["y"])
    spearman = stats.spearmanr(data["x"], data["y"])

    assert result.pearson.coefficients.loc["x", "y"] == pytest.approx(pearson.statistic)
    assert result.pearson.p_values.loc["x", "y"] == pytest.approx(pearson.pvalue)
    assert result.spearman.coefficients.loc["x", "y"] == pytest.approx(spearman.statistic)
    assert result.spearman.p_values.loc["x", "y"] == pytest.approx(spearman.pvalue)
    assert result.pearson.sample_sizes.loc["x", "y"] == 5

    z_critical = stats.norm.ppf(0.975)
    expected_lower = np.tanh(np.arctanh(pearson.statistic) - z_critical / np.sqrt(2))
    expected_upper = np.tanh(np.arctanh(pearson.statistic) + z_critical / np.sqrt(2))
    assert result.pearson.ci_lower is not None
    assert result.pearson.ci_upper is not None
    assert result.pearson.ci_lower.loc["x", "y"] == pytest.approx(expected_lower)
    assert result.pearson.ci_upper.loc["x", "y"] == pytest.approx(expected_upper)
    assert result.spearman.ci_lower is None


def test_spearman_recommendation_and_ranked_association_table() -> None:
    data = pd.DataFrame(
        {
            "studytime": [1, 2, 3, 4, 1, 2],
            "G3": [5, 7, 10, 15, 6, 9],
            "age": [15, 16, 17, 18, 16, 17],
        }
    )
    result = ranked_associations(data, "studytime", ["studytime", "G3", "age"])

    assert recommended_method("studytime", "G3") == "spearman"
    assert recommended_method("age", "G3") == "pearson"
    assert result.table["variable"].tolist()[0] == "G3"
    assert result.table.set_index("variable").loc["G3", "method"] == "spearman"
    assert result.table.set_index("variable").loc["age", "method"] == "spearman"
    assert "0.3" in result.strength_thresholds


def test_strength_thresholds_and_matrix_pairwise_missing_counts() -> None:
    assert strength_label(0.29) == "weak"
    assert strength_label(-0.3) == "moderate"
    assert strength_label(0.5) == "moderate"
    assert strength_label(-0.51) == "strong"
    with pytest.raises(CorrelationAnalysisError, match="finite"):
        strength_label(float("nan"))

    data = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, np.nan],
            "y": [2.0, 4.0, 1.0, 3.0, 5.0],
        }
    )
    result = pearson_matrix(data, ["x", "y"])
    assert result.sample_sizes.loc["x", "y"] == 4

    ranked = ranked_associations(data, "x", ["x", "y"], method="pearson")
    assert ranked.table.iloc[0]["method"] == "pearson"


def test_correlation_rejects_constant_nonnumeric_and_too_small_pairs() -> None:
    with pytest.raises(CorrelationAnalysisError, match="constant"):
        pearson_matrix(pd.DataFrame({"x": [1, 1, 1], "y": [1, 2, 3]}), ["x", "y"])
    with pytest.raises(CorrelationAnalysisError, match="numeric"):
        spearman_matrix(pd.DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]}), ["x", "y"])
    with pytest.raises(CorrelationAnalysisError, match="at least three"):
        pearson_matrix(
            pd.DataFrame({"x": [1.0, 2.0, np.nan], "y": [2.0, 3.0, 4.0]}),
            ["x", "y"],
        )
