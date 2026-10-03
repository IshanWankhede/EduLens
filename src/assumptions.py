"""Reusable statistical assumption checks with explicit status and plain-English findings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


class AssumptionCheckError(ValueError):
    """Raised when an assumption check cannot be meaningfully calculated."""


CheckStatus = Literal["ok", "warning"]


@dataclass(frozen=True)
class CheckResult:
    """Status and explanatory text for one statistical assumption check."""

    name: str
    status: CheckStatus
    message: str
    statistic: float | None = None
    p_value: float | None = None


@dataclass(frozen=True)
class VIFResult:
    """Per-predictor variance inflation factors and an explanatory status."""

    status: CheckStatus
    table: pd.DataFrame
    message: str


def _finite_sample(values: Sequence[float] | pd.Series, name: str) -> np.ndarray:
    """Return a non-empty finite numeric sample or raise a friendly check error."""
    try:
        sample = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise AssumptionCheckError(f"{name} requires numeric values.") from exc
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        raise AssumptionCheckError(f"{name} requires at least one finite observation.")
    return sample


def shapiro_check(
    values: Sequence[float] | pd.Series,
    *,
    name: str = "Shapiro–Wilk normality",
    alpha: float = 0.05,
) -> CheckResult:
    """Check a sample's compatibility with normality using Shapiro–Wilk.

    Samples must contain 3–5,000 finite values. A p-value below alpha is a warning of evidence
    against normality, not proof that the population distribution is non-normal.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    sample = _finite_sample(values, name)
    if sample.size < 3:
        return CheckResult(
            name=name,
            status="warning",
            message=f"{name}: fewer than three finite observations; normality was not assessed.",
        )
    if sample.size > 5000:
        return CheckResult(
            name=name,
            status="warning",
            message=(
                f"{name}: {sample.size} observations exceed Shapiro–Wilk's 5,000-value limit; "
                "use a documented subsample or another diagnostic."
            ),
        )
    if np.unique(sample).size < 2:
        return CheckResult(
            name=name,
            status="warning",
            message=f"{name}: sample is constant; normality is not assessable.",
        )
    result = stats.shapiro(sample)
    status = "warning" if result.pvalue < alpha else "ok"
    message = f"{name}: Shapiro–Wilk p={result.pvalue:.4g}; " + (
        "the result flags possible non-normality."
        if status == "warning"
        else "the check did not flag departure from normality."
    )
    return CheckResult(
        name=name,
        status=status,
        message=message,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
    )


def shapiro_by_group(
    groups: dict[str, Sequence[float] | pd.Series],
    *,
    alpha: float = 0.05,
) -> dict[str, CheckResult]:
    """Run Shapiro–Wilk checks separately for each named group."""
    if not groups:
        raise AssumptionCheckError("At least one group is required for Shapiro–Wilk checks.")
    return {
        name: shapiro_check(values, name=f"Shapiro–Wilk for group '{name}'", alpha=alpha)
        for name, values in groups.items()
    }


def levene_check(
    groups: Sequence[Sequence[float] | pd.Series],
    *,
    alpha: float = 0.05,
    center: str = "median",
) -> CheckResult:
    """Check equality of group variances using Levene's test (median-centered by default)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if len(groups) < 2:
        raise AssumptionCheckError("Levene's test requires at least two groups.")
    samples = [_finite_sample(group, "Levene's test") for group in groups]
    if any(sample.size < 2 for sample in samples):
        return CheckResult(
            name="Levene equal-variance check",
            status="warning",
            message="At least one group has fewer than two observations; variance equality was not assessed.",
        )
    if center not in {"mean", "median", "trimmed"}:
        raise ValueError("center must be 'mean', 'median', or 'trimmed'.")
    result = stats.levene(*samples, center=center)
    status = "warning" if result.pvalue < alpha else "ok"
    message = f"Levene's test (center={center}) p={result.pvalue:.4g}; " + (
        "the result flags possible unequal variances."
        if status == "warning"
        else "the check did not flag unequal variances."
    )
    return CheckResult(
        name="Levene equal-variance check",
        status=status,
        message=message,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
    )


def expected_count_check(expected: pd.DataFrame | np.ndarray) -> CheckResult:
    """Apply the common chi-square expected-count guideline to an expected-frequency table.

    A warning is returned if any expected value is below 1 or if fewer than 80% are at least 5.
    """
    values = np.asarray(expected, dtype=float)
    if values.size == 0 or values.ndim != 2:
        raise AssumptionCheckError("Expected counts must be a non-empty two-dimensional table.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise AssumptionCheckError("Expected counts must be finite and non-negative.")
    below_one = int((values < 1).sum())
    proportion_at_least_five = float((values >= 5).mean())
    status = "warning" if below_one > 0 or proportion_at_least_five < 0.8 else "ok"
    message = (
        f"Expected counts: {below_one} cell(s) below 1; "
        f"{proportion_at_least_five:.1%} of cells are at least 5. "
        + (
            "The usual chi-square expected-count guideline is not met."
            if status == "warning"
            else "The usual chi-square expected-count guideline is met."
        )
    )
    return CheckResult(
        name="Chi-square expected-count check",
        status=status,
        message=message,
    )


def minimum_group_size_check(
    group_sizes: dict[str, int],
    *,
    minimum: int = 2,
) -> CheckResult:
    """Warn when any named group has fewer than the requested number of observations."""
    if minimum < 1:
        raise ValueError("minimum must be at least 1.")
    if not group_sizes:
        raise AssumptionCheckError("At least one group size is required.")
    if any(size < 0 for size in group_sizes.values()):
        raise AssumptionCheckError("Group sizes cannot be negative.")
    small = {name: size for name, size in group_sizes.items() if size < minimum}
    if small:
        details = ", ".join(f"{name}={size}" for name, size in small.items())
        return CheckResult(
            name="Minimum group-size check",
            status="warning",
            message=f"Group(s) below the minimum size of {minimum}: {details}.",
        )
    return CheckResult(
        name="Minimum group-size check",
        status="ok",
        message=f"All groups meet the minimum size of {minimum}.",
    )


def vif_check(
    predictors: pd.DataFrame,
    *,
    warning_threshold: float = 5.0,
) -> VIFResult:
    """Calculate VIF per predictor and warn for values above a documented screening threshold.

    The input must contain finite, non-constant numeric predictors and at least two rows.
    VIF is a diagnostic, not a formal hypothesis test; the default threshold of 5 is a
    conventional screening rule.
    """
    if warning_threshold <= 0:
        raise ValueError("warning_threshold must be positive.")
    if predictors.empty or predictors.shape[1] < 2 or len(predictors) < 2:
        raise AssumptionCheckError("VIF requires at least two rows and two predictors.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in predictors.dtypes):
        raise AssumptionCheckError("VIF predictors must be numeric.")
    matrix = predictors.astype(float)
    if not np.isfinite(matrix.to_numpy()).all():
        raise AssumptionCheckError("VIF predictors must contain only finite values.")
    if any(matrix[column].nunique() < 2 for column in matrix.columns):
        raise AssumptionCheckError("VIF cannot be calculated for a constant predictor.")
    design = np.column_stack([np.ones(len(matrix)), matrix.to_numpy()])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise AssumptionCheckError(
            "VIF cannot be calculated because predictors are linearly dependent."
        )
    values = [
        float(variance_inflation_factor(design, index + 1)) for index in range(matrix.shape[1])
    ]
    table = pd.DataFrame({"predictor": matrix.columns, "vif": values})
    high = table.loc[table["vif"] > warning_threshold, "predictor"].tolist()
    status = "warning" if high else "ok"
    message = (
        f"VIF exceeds {warning_threshold:g} for: {', '.join(map(str, high))}."
        if high
        else f"All VIF values are at or below the screening threshold of {warning_threshold:g}."
    )
    return VIFResult(status=status, table=table, message=message)
