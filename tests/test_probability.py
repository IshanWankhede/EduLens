from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from src import config
from src.correlation import correlation_matrices
from src.data_loader import load_uci_dataset
from src.probability import (
    EmptySampleError,
    EventCondition,
    InsufficientSampleError,
    InvalidEventError,
    ProbabilityAnalysisError,
    bayes,
    conditional_probability,
    empirical_cdf,
    fit_binomial,
    fit_distributions,
    high_performance_event,
    independence_check,
)


@pytest.fixture
def probability_fixture() -> pd.DataFrame:
    # A occurs in rows 1, 2, and 5; B occurs in rows 1, 2, and 3.
    # Thus n=6, n(A)=3, n(B)=3, n(A∩B)=2, P(A)=1/2, and P(A|B)=2/3.
    return pd.DataFrame(
        {
            "high": [True, True, False, False, True, False],
            "study_group": ["yes", "yes", "yes", "no", "no", "no"],
        }
    )


def test_conditional_probability_matches_hand_calculation_and_wilson(
    probability_fixture: pd.DataFrame,
) -> None:
    result = conditional_probability(
        probability_fixture,
        EventCondition("high", "==", True),
        EventCondition("study_group", "==", "yes"),
    )

    lower, upper = proportion_confint(2, 3, alpha=0.05, method="wilson")
    assert (result.n_a, result.n_b, result.n_a_and_b, result.n) == (3, 3, 2, 6)
    assert result.p_a == pytest.approx(0.5)
    assert result.p_b == pytest.approx(0.5)
    assert result.p_a_and_b == pytest.approx(1 / 3)
    assert result.p_a_given_b == pytest.approx(2 / 3)
    assert result.wilson_ci == pytest.approx((lower, upper))
    assert "Very small" in result.message


def test_bayes_matches_direct_frequency_and_total_probability(
    probability_fixture: pd.DataFrame,
) -> None:
    # P(B)=3/6, P(A|B)=2/3, P(A|not B)=1/3; hence P(A)=1/2 and
    # P(B|A)=(2/3)(1/2)/(1/2)=2/3, equal to n(A∩B)/n(A)=2/3.
    result = bayes(
        probability_fixture,
        EventCondition("high", "==", True),
        EventCondition("study_group", "==", "yes"),
    )

    assert (result.n_a, result.n_b, result.n_not_b) == (3, 3, 3)
    assert result.n_a_and_b == 2
    assert result.n_a_and_not_b == 1
    assert result.prior_p_b == pytest.approx(0.5)
    assert result.likelihood_p_a_given_b == pytest.approx(2 / 3)
    assert result.p_a_given_not_b == pytest.approx(1 / 3)
    assert result.evidence_p_a == pytest.approx(0.5)
    assert result.posterior_p_b_given_a == pytest.approx(2 / 3)
    assert result.posterior_p_b_given_a == pytest.approx(result.n_a_and_b / result.n_a)
    assert result.posterior_p_b_given_a == pytest.approx(result.direct_p_b_given_a)


def test_zero_probability_and_empty_joint_cohort_are_reported_clearly(
    probability_fixture: pd.DataFrame,
) -> None:
    result = conditional_probability(
        probability_fixture,
        EventCondition("high", "==", True),
        EventCondition("study_group", "==", "missing"),
    )
    independence = independence_check(
        probability_fixture,
        EventCondition("high", "==", True),
        EventCondition("study_group", "==", "missing"),
    )
    assert result.p_b == 0
    assert result.p_a_given_b is None
    assert result.wilson_ci is None
    assert "P(B) = 0" in result.message
    assert independence.difference is None
    assert "not causal" in independence.interpretation

    with pytest.raises(EmptySampleError, match="non-empty"):
        conditional_probability(
            probability_fixture.iloc[:0],
            EventCondition("high", "==", True),
            EventCondition("study_group", "==", "yes"),
        )


def test_event_conditions_validate_columns_and_operators(
    probability_fixture: pd.DataFrame,
) -> None:
    with pytest.raises(InvalidEventError, match="not present"):
        conditional_probability(
            probability_fixture,
            EventCondition("unknown", "==", True),
            EventCondition("study_group", "==", "yes"),
        )
    with pytest.raises(InvalidEventError, match="Unsupported"):
        conditional_probability(
            probability_fixture,
            EventCondition("high", "contains", True),  # type: ignore[arg-type]
            EventCondition("study_group", "==", "yes"),
        )


def test_high_performance_event_uses_configured_band(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(config.PERFORMANCE_BANDS, "HIGH_LOWER_INCLUSIVE", 18)
    event = high_performance_event()
    data = pd.DataFrame({"G3": [17, 18, 19]})
    mask = data[event.column].ge(event.value)

    assert event.operator == ">="
    assert event.value == 18
    assert mask.tolist() == [False, True, True]


def test_probability_and_correlation_use_cleaned_portuguese_data() -> None:
    clean = load_uci_dataset("por").clean
    high = high_performance_event()
    study = EventCondition("studytime", ">=", 3)

    probability = conditional_probability(clean, high, study)
    correlations = correlation_matrices(clean, ["G1", "G3", "studytime"])

    expected_high = int((clean["G3"] >= config.PERFORMANCE_BANDS["HIGH_LOWER_INCLUSIVE"]).sum())
    expected_study = int((clean["studytime"] >= 3).sum())
    assert probability.n == len(clean)
    assert probability.n_a == expected_high
    assert probability.n_b == expected_study
    assert probability.p_a_given_b == pytest.approx(probability.n_a_and_b / probability.n_b)
    assert correlations.pearson.sample_sizes.to_numpy().min() == len(clean)
    assert correlations.spearman.sample_sizes.to_numpy().min() == len(clean)


def test_binomial_and_empirical_cdf_hand_calculation() -> None:
    # Four valid binary trials contain two successes, so p-hat=1/2.
    # The ECDF of [1, 1, 2, 4] is 1/2, 3/4, and 1 at its support points.
    fitted = fit_binomial(pd.Series([1, 0, 1, 0, np.nan]))
    ecdf = empirical_cdf([1, 1, 2, 4, np.nan])

    assert fitted.n == 4
    assert fitted.successes == 2
    assert fitted.p_hat == pytest.approx(0.5)
    assert ecdf.values == (1.0, 2.0, 4.0)
    assert ecdf.cumulative_probabilities == pytest.approx((0.5, 0.75, 1.0))
    assert ecdf.n == 4


def test_normal_fit_and_goodness_of_fit_cross_check_scipy() -> None:
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 4.0])
    result = fit_distributions(
        values,
        binary_event=[False, False, True, False, True, True],
    )
    expected_shapiro = stats.shapiro(values)
    expected_ks = stats.kstest(
        values,
        "norm",
        args=(np.mean(values), np.std(values, ddof=1)),
    )

    assert result.normal is not None
    assert result.normal.mu_hat == pytest.approx(np.mean(values))
    assert result.normal.sigma_hat == pytest.approx(np.std(values, ddof=1))
    assert result.binomial is not None
    assert result.binomial.p_hat == pytest.approx(0.5)
    assert result.normality.shapiro_statistic == pytest.approx(expected_shapiro.statistic)
    assert result.normality.shapiro_p_value == pytest.approx(expected_shapiro.pvalue)
    assert result.normality.ks_statistic == pytest.approx(expected_ks.statistic)
    assert result.normality.ks_p_value == pytest.approx(expected_ks.pvalue)
    assert result.normality.ks_p_value_approximate
    assert len(result.normality.qq_theoretical_quantiles) == len(values)
    assert result.normality.verdict in {"reasonable fit", "poor fit"}
    assert any("approximate" in item for item in result.normality.evidence)


def test_g3_fit_reports_bounded_integer_caveat_and_constant_sample_is_not_forced() -> None:
    g3_result = fit_distributions([7, 8, 9, 10, 11], variable_name="G3")
    constant_result = fit_distributions([4, 4, 4, 4])

    assert g3_result.normal is not None
    assert "0 to 20" in (g3_result.normal.bounded_integer_score_note or "")
    assert constant_result.normal is None
    assert constant_result.normality.verdict == "poor fit"
    assert "constant" in constant_result.normality.evidence[0]


def test_distribution_functions_raise_friendly_errors_for_invalid_samples() -> None:
    with pytest.raises(EmptySampleError, match="finite"):
        fit_distributions([np.nan, np.inf])
    with pytest.raises(InsufficientSampleError, match="at least three"):
        fit_distributions([1, 2])
    with pytest.raises(ProbabilityAnalysisError, match="binomial"):
        fit_binomial(["yes", "no"])
    with pytest.raises(ProbabilityAnalysisError, match="same number"):
        fit_distributions([1, 2, 3], binary_event=[True, False])
    with pytest.raises(ProbabilityAnalysisError, match="0–20"):
        fit_distributions([1, 2, 21], variable_name="G3")
