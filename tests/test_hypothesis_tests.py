from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway as statsmodels_anova_oneway

from src.hypothesis_tests import (
    HypothesisTestError,
    InsufficientGroupError,
    UndefinedTestError,
    chi_square_independence,
    create_binary_groups,
    independent_t_test,
    mann_whitney_u_test,
    one_way_anova,
)


def test_welch_t_test_matches_scipy_and_has_confidence_interval() -> None:
    first = np.array([2.0, 4.0, 5.0, 7.0, 8.0])
    second = np.array([1.0, 2.0, 2.0, 3.0, 5.0, 6.0])
    result = independent_t_test(first, second, alpha=0.05, confidence_level=0.95)
    expected = stats.ttest_ind(first, second, equal_var=False)

    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.degrees_of_freedom == pytest.approx(expected.df)
    assert result.mean_difference == pytest.approx(np.mean(first) - np.mean(second))
    assert result.confidence_interval[0] < result.mean_difference < result.confidence_interval[1]
    assert result.test.startswith("Independent-samples Welch")
    assert result.null_hypothesis.startswith("H0")
    assert result.alternative_hypothesis.startswith("H1")
    assert result.decision in {
        "Sufficient evidence against H0",
        "Insufficient evidence against H0",
    }
    assert "causation" in result.interpretation
    assert (
        result.suggested_alternative is None
        or result.suggested_alternative == "Mann–Whitney U test"
    )
    assert len(result.assumptions) == 4


def test_pooled_t_test_and_hedges_correction_match_hand_calculation() -> None:
    first = np.array([1.0, 2.0, 3.0, 4.0])
    second = np.array([2.0, 3.0, 4.0, 5.0])
    result = independent_t_test(first, second, equal_var=True)
    pooled_sd = np.sqrt(
        ((len(first) - 1) * np.var(first, ddof=1) + (len(second) - 1) * np.var(second, ddof=1))
        / (len(first) + len(second) - 2)
    )
    expected_d = (np.mean(first) - np.mean(second)) / pooled_sd
    correction = 1 - 3 / (4 * (len(first) + len(second) - 2) - 1)

    assert result.equal_variance_assumed
    assert result.cohens_d == pytest.approx(expected_d)
    assert result.hedges_g == pytest.approx(expected_d * correction)
    assert result.degrees_of_freedom == 6


def test_mann_whitney_callable_matches_scipy_and_tiny_group_errors() -> None:
    first = [1, 2, 3, 4]
    second = [5, 6, 7, 8]
    result = mann_whitney_u_test(first, second)
    expected = stats.mannwhitneyu(first, second, alternative="two-sided", method="auto")

    assert result.statistic == expected.statistic
    assert result.p_value == pytest.approx(expected.pvalue)
    with pytest.raises(InsufficientGroupError, match="at least two"):
        independent_t_test([1], [2, 3])
    with pytest.raises(InsufficientGroupError, match="non-empty"):
        mann_whitney_u_test([], [1])
    with pytest.raises(UndefinedTestError, match="zero within-group variance"):
        independent_t_test([1, 1, 1], [2, 2, 2])


def test_create_binary_groups_returns_cutoff_and_counts() -> None:
    data = pd.DataFrame({"studytime": [1, 2, 3, 4, np.nan]})
    result = create_binary_groups(data, "studytime", 3)

    assert result.cutoff == 3
    assert result.operator == ">="
    assert result.groups.tolist()[:4] == [
        "Below cutoff",
        "Below cutoff",
        "At or above cutoff",
        "At or above cutoff",
    ]
    assert pd.isna(result.groups.iloc[4])
    assert result.group_sizes == {"Below cutoff": 2, "At or above cutoff": 2}
    with pytest.raises(HypothesisTestError, match="not present"):
        create_binary_groups(data, "missing", 3)


def test_one_way_anova_matches_scipy_and_statsmodels_and_tukey_when_significant() -> None:
    data = pd.DataFrame(
        {
            "score": [1, 2, 2, 3, 8, 9, 10, 11, 15, 16, 17, 18],
            "group": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
        }
    )
    result = one_way_anova(data, "score", "group")
    groups = [group["score"].to_numpy() for _, group in data.groupby("group")]
    scipy_result = stats.f_oneway(*groups)
    welch_result = statsmodels_anova_oneway(
        data["score"].to_numpy(),
        data["group"].to_numpy(),
        use_var="unequal",
    )
    tukey = pairwise_tukeyhsd(data["score"], data["group"], alpha=0.05)

    assert result.statistic == pytest.approx(scipy_result.statistic)
    assert result.p_value == pytest.approx(scipy_result.pvalue)
    assert result.welch_statistic == pytest.approx(welch_result.statistic)
    assert result.welch_p_value == pytest.approx(welch_result.pvalue)
    assert result.kruskal_statistic == pytest.approx(stats.kruskal(*groups).statistic)
    assert result.eta_squared == pytest.approx(
        sum(len(group) * (np.mean(group) - np.mean(data["score"])) ** 2 for group in groups)
        / np.sum((data["score"] - np.mean(data["score"])) ** 2)
    )
    assert result.group_means["group"].tolist() == ["A", "B", "C"]
    assert result.tukey_comparisons is not None
    assert len(result.tukey_comparisons) == len(tukey._results_table.data) - 1
    assert all(check.status in {"ok", "warning"} for check in result.assumptions)
    assert "Dunn" in result.posthoc_note
    assert "causation" in result.interpretation


def test_anova_raises_for_empty_tiny_or_zero_variance_groups() -> None:
    with pytest.raises(InsufficientGroupError, match="No complete"):
        one_way_anova(
            pd.DataFrame({"score": [np.nan, np.nan], "group": ["A", "B"]}),
            "score",
            "group",
        )
    with pytest.raises(InsufficientGroupError, match="at least three"):
        one_way_anova(
            pd.DataFrame({"score": [1, 2, 3, 4], "group": ["A", "A", "B", "B"]}),
            "score",
            "group",
        )
    with pytest.raises(InsufficientGroupError, match="at least two"):
        one_way_anova(
            pd.DataFrame({"score": [1, 2, 3, 4, 5], "group": ["A", "A", "B", "B", "C"]}),
            "score",
            "group",
        )
    with pytest.raises(UndefinedTestError, match="zero within-group variance"):
        one_way_anova(
            pd.DataFrame({"score": [1, 1, 2, 2, 3, 3], "group": ["A", "A", "B", "B", "C", "C"]}),
            "score",
            "group",
        )


def test_chi_square_matches_scipy_and_reports_yates_setting() -> None:
    data = pd.DataFrame(
        {
            "study_group": ["low"] * 10 + ["high"] * 10,
            "outcome": ["no"] * 8 + ["yes"] * 2 + ["no"] * 3 + ["yes"] * 7,
        }
    )
    result = chi_square_independence(data, "study_group", "outcome", yates_correction=True)
    expected = stats.chi2_contingency(result.observed, correction=True)

    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.degrees_of_freedom == expected.dof
    assert result.expected.to_numpy() == pytest.approx(expected.expected_freq)
    assert result.standardized_residuals.shape == result.observed.shape
    assert result.cramers_v == pytest.approx(
        np.sqrt(result.statistic / (result.observed.to_numpy().sum() * 1))
    )
    assert result.yates_correction
    assert "Yates' correction was applied" in result.interpretation
    assert "causation" in result.interpretation


def test_chi_square_sparse_2x2_includes_fisher_and_empty_group_errors() -> None:
    sparse = pd.DataFrame(
        {
            "a": ["x"] * 10 + ["y"] * 10,
            "b": ["yes"] + ["no"] * 9 + ["yes"] * 8 + ["no"] * 2,
        }
    )
    result = chi_square_independence(sparse, "a", "b")
    assert result.expected_count_check.status == "warning"
    assert result.fisher_exact_p_value == pytest.approx(
        stats.fisher_exact(result.observed.to_numpy()).pvalue
    )
    assert result.fisher_exact_odds_ratio is not None
    with pytest.raises(InsufficientGroupError, match="at least two levels"):
        chi_square_independence(pd.DataFrame({"a": ["x", "x"], "b": ["yes", "no"]}), "a", "b")
