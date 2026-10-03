"""Empirical probability, Bayes, and distribution-fitting helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from src import config

ConditionOperator = Literal["==", "!=", ">", ">=", "<", "<=", "in", "not in"]


class ProbabilityAnalysisError(ValueError):
    """Base exception for invalid probability-analysis inputs."""


class EmptySampleError(ProbabilityAnalysisError):
    """Raised when there are no usable observations for an analysis."""


class InvalidEventError(ProbabilityAnalysisError):
    """Raised when an event condition cannot be evaluated."""


class InsufficientSampleError(ProbabilityAnalysisError):
    """Raised when a distribution diagnostic needs more observations."""


@dataclass(frozen=True)
class EventCondition:
    """A simple dataframe event defined by a column comparison.

    `value` is a category or scalar threshold for comparisons, and a collection of categories
    for `in` and `not in`. Missing rows in either event column are excluded from joint analyses.
    """

    column: str
    operator: ConditionOperator
    value: object


@dataclass(frozen=True)
class ConditionalProbabilityResult:
    """Empirical event counts, probabilities, Wilson interval, and stability message."""

    n_a: int
    n_b: int
    n_a_and_b: int
    n: int
    p_a: float
    p_b: float
    p_a_and_b: float
    p_a_given_b: float | None
    wilson_ci: tuple[float, float] | None
    message: str


@dataclass(frozen=True)
class IndependenceCheckResult:
    """Descriptive comparison of marginal and conditional probability."""

    probabilities: ConditionalProbabilityResult
    difference: float | None
    interpretation: str


@dataclass(frozen=True)
class BayesResult:
    """Bayes calculation with both conditional-probability paths and their counts."""

    n: int
    n_a: int
    n_b: int
    n_not_b: int
    n_a_and_b: int
    n_a_and_not_b: int
    prior_p_b: float
    likelihood_p_a_given_b: float | None
    p_a_given_not_b: float | None
    evidence_p_a: float
    posterior_p_b_given_a: float | None
    direct_p_b_given_a: float | None
    message: str


@dataclass(frozen=True)
class BinomialFitResult:
    """Estimated event probability and trial count for a binary sample."""

    p_hat: float
    n: int
    successes: int


@dataclass(frozen=True)
class EmpiricalCDFResult:
    """Step-function empirical CDF represented by its sorted support and cumulative shares."""

    values: tuple[float, ...]
    cumulative_probabilities: tuple[float, ...]
    n: int


@dataclass(frozen=True)
class NormalFitResult:
    """Sample mean and sample standard deviation for an approximate normal fit."""

    mu_hat: float
    sigma_hat: float
    n: int
    bounded_integer_score_note: str | None


@dataclass(frozen=True)
class NormalityResult:
    """Normal-fit diagnostics, Q–Q coordinates, evidence, and an explicit fit verdict."""

    shapiro_statistic: float | None
    shapiro_p_value: float | None
    ks_statistic: float | None
    ks_p_value: float | None
    ks_p_value_approximate: bool
    qq_theoretical_quantiles: tuple[float, ...]
    qq_ordered_values: tuple[float, ...]
    verdict: Literal["reasonable fit", "poor fit"]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class DistributionFitResult:
    """Normal diagnostics, optional binomial estimates, and empirical CDF for a sample."""

    normal: NormalFitResult | None
    binomial: BinomialFitResult | None
    empirical_cdf: EmpiricalCDFResult
    normality: NormalityResult


def high_performance_event(grade_column: str = "G3") -> EventCondition:
    """Return the high-performance condition using the configured fixed grade-band boundary."""
    return EventCondition(
        column=grade_column,
        operator=">=",
        value=config.PERFORMANCE_BANDS["HIGH_LOWER_INCLUSIVE"],
    )


def _validate_event(data: pd.DataFrame, event: EventCondition) -> None:
    """Validate event columns/operators before evaluating any row-wise comparisons."""
    if not event.column or event.column not in data.columns:
        raise InvalidEventError(f"Event column '{event.column}' is not present in the dataset.")
    if event.operator not in {"==", "!=", ">", ">=", "<", "<=", "in", "not in"}:
        raise InvalidEventError(f"Unsupported event operator '{event.operator}'.")
    if event.operator in {"in", "not in"} and (
        isinstance(event.value, (str, bytes)) or not isinstance(event.value, Sequence)
    ):
        raise InvalidEventError(
            f"Event operator '{event.operator}' requires a collection of categories."
        )


def _event_mask(data: pd.DataFrame, event: EventCondition) -> pd.Series:
    """Evaluate one validated event as a boolean Series."""
    _validate_event(data, event)
    values = data[event.column]
    try:
        if event.operator == "==":
            result = values.eq(event.value)
        elif event.operator == "!=":
            result = values.ne(event.value)
        elif event.operator == ">":
            result = values.gt(event.value)
        elif event.operator == ">=":
            result = values.ge(event.value)
        elif event.operator == "<":
            result = values.lt(event.value)
        elif event.operator == "<=":
            result = values.le(event.value)
        elif event.operator == "in":
            result = values.isin(event.value)
        else:
            result = ~values.isin(event.value)
    except (TypeError, ValueError) as exc:
        raise InvalidEventError(
            f"Condition {event.column} {event.operator} {event.value!r} cannot be evaluated."
        ) from exc
    return result.fillna(False).astype(bool)


def _joint_event_counts(
    data: pd.DataFrame,
    event_a: EventCondition,
    event_b: EventCondition,
) -> tuple[int, int, int, int, int, int]:
    """Return n, n(A), n(B), n(A∩B), n(not B), and n(A∩not B) on a common cohort."""
    if data.empty:
        raise EmptySampleError("Probability analysis requires a non-empty dataset.")
    _validate_event(data, event_a)
    _validate_event(data, event_b)
    cohort = data.loc[:, list(dict.fromkeys((event_a.column, event_b.column)))].dropna()
    if cohort.empty:
        raise EmptySampleError("No complete observations are available for these event conditions.")
    a = _event_mask(cohort, event_a)
    b = _event_mask(cohort, event_b)
    n = len(cohort)
    n_a = int(a.sum())
    n_b = int(b.sum())
    n_a_and_b = int((a & b).sum())
    n_not_b = n - n_b
    n_a_and_not_b = n_a - n_a_and_b
    return n, n_a, n_b, n_a_and_b, n_not_b, n_a_and_not_b


def conditional_probability(
    data: pd.DataFrame,
    event_a: EventCondition,
    event_b: EventCondition,
    *,
    confidence_level: float = config.DEFAULT_CONFIDENCE,
) -> ConditionalProbabilityResult:
    """Estimate P(A), P(B), P(A∩B), and P(A|B) with a Wilson interval.

    The analysis uses complete rows for the columns defining both events. If B has no observed
    cases, P(A|B) and its interval are undefined and the returned message explains why. A
    denominator below `config.SMALL_CONDITIONAL_SAMPLE_SIZE` is flagged as unstable. Empirical
    probabilities describe the observed cohort; generalization assumes it is representative of
    the population of interest.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    n, n_a, n_b, n_a_and_b, _, _ = _joint_event_counts(data, event_a, event_b)
    p_a = n_a / n
    p_b = n_b / n
    p_a_and_b = n_a_and_b / n
    if n_b == 0:
        return ConditionalProbabilityResult(
            n_a=n_a,
            n_b=n_b,
            n_a_and_b=n_a_and_b,
            n=n,
            p_a=p_a,
            p_b=p_b,
            p_a_and_b=p_a_and_b,
            p_a_given_b=None,
            wilson_ci=None,
            message="P(B) = 0 in the observed sample, so P(A|B) and its Wilson interval are undefined.",
        )
    lower, upper = proportion_confint(
        count=n_a_and_b,
        nobs=n_b,
        alpha=1 - confidence_level,
        method="wilson",
    )
    message = ""
    if n_b < config.SMALL_CONDITIONAL_SAMPLE_SIZE:
        message = (
            f"Very small conditioning group (n(B)={n_b} < "
            f"{config.SMALL_CONDITIONAL_SAMPLE_SIZE}); interpret the estimate cautiously."
        )
    return ConditionalProbabilityResult(
        n_a=n_a,
        n_b=n_b,
        n_a_and_b=n_a_and_b,
        n=n,
        p_a=p_a,
        p_b=p_b,
        p_a_and_b=p_a_and_b,
        p_a_given_b=n_a_and_b / n_b,
        wilson_ci=(float(lower), float(upper)),
        message=message,
    )


def independence_check(
    data: pd.DataFrame,
    event_a: EventCondition,
    event_b: EventCondition,
) -> IndependenceCheckResult:
    """Compare P(A|B) with P(A) as a descriptive association, not a causal claim."""
    probabilities = conditional_probability(data, event_a, event_b)
    difference = (
        None if probabilities.p_a_given_b is None else probabilities.p_a_given_b - probabilities.p_a
    )
    if difference is None:
        interpretation = (
            "Independence cannot be assessed from the observed sample because P(B) is zero; "
            "this comparison is descriptive, not causal."
        )
    else:
        interpretation = (
            f"P(A|B) - P(A) = {difference:.4f}. This describes a difference in observed "
            "probabilities and does not establish causation or independence in a population."
        )
    return IndependenceCheckResult(
        probabilities=probabilities,
        difference=difference,
        interpretation=interpretation,
    )


def bayes(
    data: pd.DataFrame,
    event_a: EventCondition,
    event_b: EventCondition,
) -> BayesResult:
    """Apply Bayes' theorem and total probability to a common complete-case cohort.

    The event B and its complement partition the observed rows. Interpretation as population
    probabilities assumes the complete-case cohort is representative of the target population.
    """
    n, n_a, n_b, n_a_and_b, n_not_b, n_a_and_not_b = _joint_event_counts(data, event_a, event_b)
    prior = n_b / n
    likelihood = n_a_and_b / n_b if n_b else None
    complement_likelihood = n_a_and_not_b / n_not_b if n_not_b else None
    evidence = (likelihood or 0.0) * prior + (complement_likelihood or 0.0) * (n_not_b / n)
    posterior = n_a_and_b / n_a if n_a else None
    direct = posterior
    message = ""
    if n_a == 0:
        message = "P(A) = 0 in the observed sample, so P(B|A) is undefined."
    elif n_b == 0:
        message = "P(B) = 0 in the observed sample; posterior is zero by direct frequency."
    elif n_b < config.SMALL_CONDITIONAL_SAMPLE_SIZE:
        message = (
            f"Very small conditioning group (n(B)={n_b} < "
            f"{config.SMALL_CONDITIONAL_SAMPLE_SIZE}); interpret the likelihood cautiously."
        )
    return BayesResult(
        n=n,
        n_a=n_a,
        n_b=n_b,
        n_not_b=n_not_b,
        n_a_and_b=n_a_and_b,
        n_a_and_not_b=n_a_and_not_b,
        prior_p_b=prior,
        likelihood_p_a_given_b=likelihood,
        p_a_given_not_b=complement_likelihood,
        evidence_p_a=evidence,
        posterior_p_b_given_a=posterior,
        direct_p_b_given_a=direct,
        message=message,
    )


def fit_binomial(event_values: pd.Series | Sequence[bool | int]) -> BinomialFitResult:
    """Estimate p for a binary event; `n` is its number of non-missing trials.

    A binomial model additionally assumes a fixed number of independent trials with a common
    event probability; this function estimates the sample proportion but does not verify those
    assumptions.
    """
    values = pd.Series(event_values, dtype="object").dropna()
    if values.empty:
        raise EmptySampleError("A binomial fit requires at least one observed event value.")
    if not values.isin([True, False, 0, 1]).all():
        raise ProbabilityAnalysisError("A binomial event must contain only Boolean or 0/1 values.")
    successes = int(values.astype(bool).sum())
    return BinomialFitResult(p_hat=successes / len(values), n=len(values), successes=successes)


def empirical_cdf(values: pd.Series | Sequence[float]) -> EmpiricalCDFResult:
    """Compute the right-continuous ECDF on unique finite values.

    The ECDF describes the supplied sample; population interpretation depends on representative
    sampling.
    """
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    sample = series[np.isfinite(series)].to_numpy(dtype=float)
    if sample.size == 0:
        raise EmptySampleError("An empirical CDF requires at least one finite observation.")
    support, counts = np.unique(sample, return_counts=True)
    cumulative = np.cumsum(counts) / sample.size
    return EmpiricalCDFResult(
        values=tuple(float(value) for value in support),
        cumulative_probabilities=tuple(float(value) for value in cumulative),
        n=int(sample.size),
    )


def fit_distributions(
    values: pd.Series | Sequence[float],
    *,
    binary_event: pd.Series | Sequence[bool | int] | None = None,
    variable_name: str | None = None,
    alpha: float = config.DEFAULT_ALPHA,
) -> DistributionFitResult:
    """Fit an approximate normal model, optional binary-event binomial, and empirical CDF.

    The normal estimate uses the sample mean and sample SD (ddof=1) for independent numeric
    observations. Q–Q, Shapiro–Wilk, and
    one-sample KS diagnostics assess normality; KS p-values are explicitly approximate because
    the normal parameters are estimated from the same data. G3 is a bounded integer in 0–20, so
    a normal fit for G3 is only an approximation. A constant sample is not forced into a normal
    model and receives a poor-fit verdict.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    raw_values = pd.Series(values)
    numeric = pd.to_numeric(raw_values, errors="coerce")
    sample_mask = np.isfinite(numeric.to_numpy(dtype=float))
    sample = numeric.to_numpy(dtype=float)[sample_mask]
    if sample.size == 0:
        raise EmptySampleError("Distribution fitting requires at least one finite observation.")
    if sample.size < 3:
        raise InsufficientSampleError(
            "Normality diagnostics require at least three finite observations."
        )

    binomial: BinomialFitResult | None = None
    if binary_event is not None:
        events = pd.Series(binary_event, dtype="object")
        if len(events) != len(raw_values):
            raise ProbabilityAnalysisError(
                "binary_event must have the same number of rows as the input values."
            )
        events = events.iloc[np.flatnonzero(sample_mask)]
        binomial = fit_binomial(events)

    ecdf = empirical_cdf(sample)
    unique_count = np.unique(sample).size
    if unique_count < 2:
        normal = None
        evidence = ("The sample is constant; a non-degenerate normal fit is not appropriate.",)
        normality = NormalityResult(
            shapiro_statistic=None,
            shapiro_p_value=None,
            ks_statistic=None,
            ks_p_value=None,
            ks_p_value_approximate=True,
            qq_theoretical_quantiles=(),
            qq_ordered_values=(),
            verdict="poor fit",
            evidence=evidence,
        )
        return DistributionFitResult(
            normal=normal,
            binomial=binomial,
            empirical_cdf=ecdf,
            normality=normality,
        )

    mu_hat = float(np.mean(sample))
    sigma_hat = float(np.std(sample, ddof=1))
    score_note: str | None = None
    if variable_name == "G3":
        score_note = "G3 is a bounded integer score from 0 to 20; a normal fit is approximate."
        if (
            not np.equal(sample, np.floor(sample)).all()
            or (sample < 0).any()
            or (sample > 20).any()
        ):
            raise ProbabilityAnalysisError(
                "G3 values must be integers in the documented 0–20 range."
            )
    normal = NormalFitResult(
        mu_hat=mu_hat,
        sigma_hat=sigma_hat,
        n=int(sample.size),
        bounded_integer_score_note=score_note,
    )

    shapiro_sample = sample
    shapiro_note = ""
    if sample.size > 5000:
        indices = np.random.default_rng(config.RANDOM_SEED).choice(sample.size, 5000, replace=False)
        shapiro_sample = sample[indices]
        shapiro_note = "Shapiro–Wilk used a reproducible 5,000-observation subsample."
    shapiro_result = stats.shapiro(shapiro_sample)
    ks_result = stats.kstest(sample, "norm", args=(mu_hat, sigma_hat))
    (theoretical, ordered), _ = stats.probplot(sample, dist="norm")
    evidence_items = [
        f"Shapiro–Wilk p-value={float(shapiro_result.pvalue):.6g}.",
        (
            f"KS p-value={float(ks_result.pvalue):.6g}; this p-value is approximate "
            "because normal parameters were estimated from this sample."
        ),
        "Use the Q–Q coordinates to inspect systematic departures from a straight reference pattern.",
    ]
    if shapiro_note:
        evidence_items.append(shapiro_note)
    if score_note:
        evidence_items.append(score_note)
    verdict: Literal["reasonable fit", "poor fit"] = (
        "poor fit"
        if shapiro_result.pvalue < alpha or ks_result.pvalue < alpha
        else "reasonable fit"
    )
    normality = NormalityResult(
        shapiro_statistic=float(shapiro_result.statistic),
        shapiro_p_value=float(shapiro_result.pvalue),
        ks_statistic=float(ks_result.statistic),
        ks_p_value=float(ks_result.pvalue),
        ks_p_value_approximate=True,
        qq_theoretical_quantiles=tuple(float(value) for value in theoretical),
        qq_ordered_values=tuple(float(value) for value in ordered),
        verdict=verdict,
        evidence=tuple(evidence_items),
    )
    return DistributionFitResult(
        normal=normal,
        binomial=binomial,
        empirical_cdf=ecdf,
        normality=normality,
    )
