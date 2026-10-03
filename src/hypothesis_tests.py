"""Classical hypothesis tests with effect sizes, diagnostics, and cautious interpretation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway as statsmodels_anova_oneway

from src import config
from src.assumptions import (
    CheckResult,
    expected_count_check,
    levene_check,
    minimum_group_size_check,
    shapiro_by_group,
    shapiro_check,
)

TestStatus = Literal["ok", "warning"]


class HypothesisTestError(ValueError):
    """Base exception for invalid or insufficient hypothesis-test input."""


class InsufficientGroupError(HypothesisTestError):
    """Raised when a test requires more observations or non-empty groups."""


class UndefinedTestError(HypothesisTestError):
    """Raised when a statistic is undefined for the supplied observations."""


@dataclass(frozen=True)
class GroupCreationResult:
    """Two-group labels and explicit split rule for UI display."""

    groups: pd.Series
    variable: str
    cutoff: float
    operator: Literal[">=", ">"]
    lower_label: str
    upper_label: str
    group_sizes: dict[str, int]


@dataclass(frozen=True)
class IndependentTTestResult:
    """Independent-samples test result ordered for TEST → STATISTIC → P-VALUE display."""

    test: str
    statistic: float
    degrees_of_freedom: float
    p_value: float
    alpha: float
    confidence_level: float
    confidence_interval: tuple[float, float]
    mean_difference: float
    cohens_d: float
    hedges_g: float
    null_hypothesis: str
    alternative_hypothesis: str
    decision: str
    interpretation: str
    assumptions: tuple[CheckResult, ...]
    suggested_alternative: str | None
    group_sizes: tuple[int, int]
    equal_variance_assumed: bool


@dataclass(frozen=True)
class MannWhitneyResult:
    """Mann–Whitney U test result, interpreted as a distributional association."""

    test: str
    statistic: float
    p_value: float
    alpha: float
    null_hypothesis: str
    alternative_hypothesis: str
    decision: str
    interpretation: str
    group_sizes: tuple[int, int]


@dataclass(frozen=True)
class OneWayAnovaResult:
    """One-way ANOVA, alternatives, assumption checks, and conditional Tukey comparisons."""

    test: str
    statistic: float
    degrees_of_freedom: tuple[float, float]
    p_value: float
    alpha: float
    decision: str
    interpretation: str
    group_means: pd.DataFrame
    eta_squared: float
    assumptions: tuple[CheckResult, ...]
    welch_statistic: float
    welch_degrees_of_freedom: tuple[float, float]
    welch_p_value: float
    kruskal_statistic: float
    kruskal_degrees_of_freedom: int
    kruskal_p_value: float
    tukey_comparisons: pd.DataFrame | None
    posthoc_note: str


@dataclass(frozen=True)
class ChiSquareResult:
    """Chi-square independence test with expected counts and optional Fisher alternative."""

    test: str
    statistic: float
    degrees_of_freedom: int
    p_value: float
    alpha: float
    decision: str
    interpretation: str
    observed: pd.DataFrame
    expected: pd.DataFrame
    standardized_residuals: pd.DataFrame
    cramers_v: float
    expected_count_check: CheckResult
    fisher_exact_odds_ratio: float | None
    fisher_exact_p_value: float | None
    yates_correction: bool


@dataclass(frozen=True)
class _TestDecision:
    """Common null/alternative decision wording."""

    decision: str
    verb: str


def _decision(p_value: float, alpha: float) -> _TestDecision:
    """Return the project-standard evidence decision without accepting a null hypothesis."""
    sufficient = p_value < alpha
    return _TestDecision(
        decision=(
            "Sufficient evidence against H0" if sufficient else "Insufficient evidence against H0"
        ),
        verb="is" if sufficient else "is not",
    )


def _sample(values: Sequence[float] | pd.Series, group_name: str) -> np.ndarray:
    """Convert a group to its finite numeric observations."""
    try:
        result = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise HypothesisTestError(f"Group '{group_name}' must contain numeric values.") from exc
    return result[np.isfinite(result)]


def _validate_test_levels(alpha: float, confidence_level: float) -> None:
    """Validate test alpha and confidence level."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")


def create_binary_groups(
    data: pd.DataFrame,
    column: str,
    cutoff: float,
    *,
    operator: Literal[">=", ">"] = ">=",
    lower_label: str = "Below cutoff",
    upper_label: str = "At or above cutoff",
) -> GroupCreationResult:
    """Split a numeric or ordinal-coded column and return its labels and displayed cutoff rule.

    The default split is `value >= cutoff` versus `value < cutoff`; missing and non-finite
    observations remain unassigned and are excluded from the returned group-size counts.
    """
    if column not in data.columns:
        raise HypothesisTestError(f"Grouping column '{column}' is not present in the dataset.")
    if not pd.api.types.is_numeric_dtype(data[column].dtype):
        raise HypothesisTestError(f"Grouping column '{column}' must be numeric or ordinal-coded.")
    if not np.isfinite(cutoff):
        raise ValueError("cutoff must be finite.")
    if operator not in {">=", ">"}:
        raise ValueError("operator must be '>=' or '>'.")
    if not lower_label or not upper_label or lower_label == upper_label:
        raise ValueError("Group labels must be non-empty and distinct.")

    values = pd.to_numeric(data[column], errors="coerce")
    valid = values.notna() & np.isfinite(values)
    high = values >= cutoff if operator == ">=" else values > cutoff
    labels = pd.Series(pd.NA, index=data.index, dtype="string", name=f"{column}_group")
    labels.loc[valid & ~high] = lower_label
    labels.loc[valid & high] = upper_label
    counts = labels.value_counts().reindex([lower_label, upper_label], fill_value=0)
    return GroupCreationResult(
        groups=labels,
        variable=column,
        cutoff=float(cutoff),
        operator=operator,
        lower_label=lower_label,
        upper_label=upper_label,
        group_sizes={name: int(counts.loc[name]) for name in (lower_label, upper_label)},
    )


def independent_t_test(
    group1: Sequence[float] | pd.Series,
    group2: Sequence[float] | pd.Series,
    *,
    equal_var: bool = False,
    alpha: float = config.DEFAULT_ALPHA,
    confidence_level: float = config.DEFAULT_CONFIDENCE,
) -> IndependentTTestResult:
    """Compare two independent means with Welch's test by default.

    Finite observations are used. The confidence interval uses Welch–Satterthwaite degrees of
    freedom for the default unequal-variance test and pooled variance for `equal_var=True`.
    Shapiro and Levene results are diagnostics; independence and suitable sampling remain design
    assumptions. Hedges' g applies the small-sample correction to pooled-standardized Cohen's d.
    """
    _validate_test_levels(alpha, confidence_level)
    first = _sample(group1, "group1")
    second = _sample(group2, "group2")
    sizes = (len(first), len(second))
    if min(sizes) < 2:
        raise InsufficientGroupError(
            "Each independent t-test group requires at least two finite observations."
        )
    variances = (float(np.var(first, ddof=1)), float(np.var(second, ddof=1)))
    if variances[0] == 0 and variances[1] == 0:
        raise UndefinedTestError(
            "Both groups have zero within-group variance; the t statistic and standardized effect are undefined."
        )
    result = stats.ttest_ind(first, second, equal_var=equal_var)
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if not np.isfinite(statistic) or not np.isfinite(p_value):
        raise UndefinedTestError("The t statistic is undefined for these group values.")

    mean_difference = float(np.mean(first) - np.mean(second))
    degrees_freedom = (
        float(len(first) + len(second) - 2)
        if equal_var
        else float(
            (variances[0] / len(first) + variances[1] / len(second)) ** 2
            / (
                (variances[0] / len(first)) ** 2 / (len(first) - 1)
                + (variances[1] / len(second)) ** 2 / (len(second) - 1)
            )
        )
    )
    if equal_var:
        pooled_variance = (
            (len(first) - 1) * variances[0] + (len(second) - 1) * variances[1]
        ) / degrees_freedom
        standard_error = np.sqrt(pooled_variance * (1 / len(first) + 1 / len(second)))
        pooled_sd = float(np.sqrt(pooled_variance))
    else:
        standard_error = np.sqrt(variances[0] / len(first) + variances[1] / len(second))
        pooled_sd = float(
            np.sqrt(
                ((len(first) - 1) * variances[0] + (len(second) - 1) * variances[1])
                / (len(first) + len(second) - 2)
            )
        )
    critical = stats.t.ppf(1 - (1 - confidence_level) / 2, degrees_freedom)
    interval = (
        float(mean_difference - critical * standard_error),
        float(mean_difference + critical * standard_error),
    )
    if pooled_sd == 0:
        raise UndefinedTestError("The pooled standard deviation is zero; Cohen's d is undefined.")
    cohens_d = mean_difference / pooled_sd
    degrees_pooled = len(first) + len(second) - 2
    correction = 1 - 3 / (4 * degrees_pooled - 1)
    hedges_g = float(correction * cohens_d)

    normality_checks = (
        shapiro_check(first, name="Shapiro–Wilk for group 1", alpha=alpha),
        shapiro_check(second, name="Shapiro–Wilk for group 2", alpha=alpha),
    )
    levene = levene_check([first, second], alpha=alpha)
    group_size_check = minimum_group_size_check(
        {"group 1": sizes[0], "group 2": sizes[1]}, minimum=2
    )
    checks = (*normality_checks, levene, group_size_check)
    assumption_warning = any(check.status == "warning" for check in checks)
    alternative = "Mann–Whitney U test" if assumption_warning else None
    evidence = _decision(p_value, alpha)
    variance_method = "pooled-variance" if equal_var else "Welch"
    interpretation = (
        f"{evidence.decision} for a difference in group means "
        f"({variance_method} t={statistic:.4g}, df={degrees_freedom:.4g}, "
        f"p={p_value:.4g}, {confidence_level:.0%} CI for mean difference "
        f"[{interval[0]:.4g}, {interval[1]:.4g}]). The standardized difference is "
        f"Cohen's d={cohens_d:.4g} (Hedges' g={hedges_g:.4g}); this is an association "
        "comparison and does not show causation."
    )
    if alternative:
        interpretation += " One or more assumption checks raised a warning; consider the suggested Mann–Whitney U alternative."
    return IndependentTTestResult(
        test=f"Independent-samples {variance_method} t-test",
        statistic=statistic,
        degrees_of_freedom=degrees_freedom,
        p_value=p_value,
        alpha=alpha,
        confidence_level=confidence_level,
        confidence_interval=interval,
        mean_difference=mean_difference,
        cohens_d=float(cohens_d),
        hedges_g=hedges_g,
        null_hypothesis="H0: the population means are equal.",
        alternative_hypothesis="H1: the population means differ.",
        decision=evidence.decision,
        interpretation=interpretation,
        assumptions=checks,
        suggested_alternative=alternative,
        group_sizes=sizes,
        equal_variance_assumed=equal_var,
    )


def mann_whitney_u_test(
    group1: Sequence[float] | pd.Series,
    group2: Sequence[float] | pd.Series,
    *,
    alpha: float = config.DEFAULT_ALPHA,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> MannWhitneyResult:
    """Run a two-independent-sample Mann–Whitney U test as a nonparametric alternative.

    This tests a distributional/rank difference; it is not automatically a test of medians
    unless group distribution shapes are comparable.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    first = _sample(group1, "group1")
    second = _sample(group2, "group2")
    if not len(first) or not len(second):
        raise InsufficientGroupError("Mann–Whitney U requires two non-empty groups.")
    result = stats.mannwhitneyu(first, second, alternative=alternative, method="auto")
    evidence = _decision(float(result.pvalue), alpha)
    return MannWhitneyResult(
        test="Mann–Whitney U test",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alpha=alpha,
        null_hypothesis="H0: the two independent groups have the same distribution.",
        alternative_hypothesis=(
            "H1: the group distributions differ."
            if alternative == "two-sided"
            else f"H1: the first group tends to be {alternative} than the second."
        ),
        decision=evidence.decision,
        interpretation=(
            f"{evidence.decision} for a distributional difference "
            f"(U={result.statistic:.4g}, p={result.pvalue:.4g}). This describes group "
            "association and does not show causation."
        ),
        group_sizes=(len(first), len(second)),
    )


def _group_arrays(
    data: pd.DataFrame,
    value_column: str,
    group_column: str,
) -> tuple[dict[str, np.ndarray], int]:
    """Extract finite outcomes for observed non-missing groups and count excluded rows."""
    missing = [column for column in (value_column, group_column) if column not in data.columns]
    if missing:
        raise HypothesisTestError(f"Column(s) missing from data: {', '.join(missing)}.")
    numeric = pd.to_numeric(data[value_column], errors="coerce")
    valid = data[group_column].notna() & numeric.notna() & np.isfinite(numeric)
    filtered = data.loc[valid, [group_column]].copy()
    filtered[value_column] = numeric.loc[valid]
    if filtered.empty:
        raise InsufficientGroupError("No complete finite observations are available for the test.")
    grouped = {
        str(label): group[value_column].to_numpy(dtype=float)
        for label, group in filtered.groupby(group_column, observed=True, sort=False)
    }
    return grouped, int(len(data) - len(filtered))


def one_way_anova(
    data: pd.DataFrame,
    value_column: str,
    group_column: str,
    *,
    alpha: float = config.DEFAULT_ALPHA,
    confidence_level: float = config.DEFAULT_CONFIDENCE,
) -> OneWayAnovaResult:
    """Compare at least three independent group means with ANOVA and robust alternatives.

    Groups require at least two finite outcomes. Classical ANOVA CIs use each group's t interval;
    assumption checks include per-group Shapiro–Wilk and median-centered Levene. Welch ANOVA and
    Kruskal–Wallis are always supplied as alternatives. Tukey HSD is run only after significant
    classical ANOVA. Dunn pairwise post-hoc comparisons are not included because this phase has no
    selected Dunn implementation; Kruskal–Wallis is reported as an omnibus alternative only.
    """
    _validate_test_levels(alpha, confidence_level)
    groups, excluded_rows = _group_arrays(data, value_column, group_column)
    if len(groups) < 3:
        raise InsufficientGroupError("One-way ANOVA requires at least three non-empty groups.")
    sizes = {name: len(values) for name, values in groups.items()}
    if any(size < 2 for size in sizes.values()):
        small = ", ".join(f"{name}={size}" for name, size in sizes.items() if size < 2)
        raise InsufficientGroupError(
            f"Each ANOVA group requires at least two observations; insufficient group(s): {small}."
        )
    samples = list(groups.values())
    if all(np.var(sample, ddof=1) == 0 for sample in samples):
        raise UndefinedTestError(
            "All groups have zero within-group variance; ANOVA's F statistic is undefined."
        )

    classic = stats.f_oneway(*samples)
    try:
        welch = statsmodels_anova_oneway(
            np.concatenate(samples),
            np.concatenate([np.repeat(index, len(sample)) for index, sample in enumerate(samples)]),
            use_var="unequal",
        )
    except (ValueError, ZeroDivisionError) as exc:
        raise UndefinedTestError(f"Welch ANOVA could not be calculated: {exc}") from exc
    kruskal = stats.kruskal(*samples)

    values = np.concatenate(samples)
    grand_mean = float(np.mean(values))
    between_ss = sum(len(sample) * (float(np.mean(sample)) - grand_mean) ** 2 for sample in samples)
    total_ss = float(np.sum((values - grand_mean) ** 2))
    eta_squared = between_ss / total_ss if total_ss > 0 else float("nan")

    mean_rows: list[dict[str, float | int | str]] = []
    shapiro_checks = shapiro_by_group(groups, alpha=alpha)
    for name, sample in groups.items():
        count = len(sample)
        mean = float(np.mean(sample))
        standard_error = float(stats.sem(sample))
        critical = float(stats.t.ppf(1 - (1 - confidence_level) / 2, count - 1))
        margin = critical * standard_error
        mean_rows.append(
            {
                "group": name,
                "count": count,
                "mean": mean,
                "mean_ci_lower": mean - margin,
                "mean_ci_upper": mean + margin,
            }
        )
    mean_table = pd.DataFrame(mean_rows)
    levene = levene_check(samples, alpha=alpha)
    min_sizes = minimum_group_size_check(sizes, minimum=2)
    checks = (*shapiro_checks.values(), levene, min_sizes)
    evidence = _decision(float(classic.pvalue), alpha)
    tukey_frame: pd.DataFrame | None = None
    posthoc_note = "Tukey HSD was not run because the classical ANOVA was not significant."
    if classic.pvalue < alpha:
        tukey = pairwise_tukeyhsd(
            values,
            np.concatenate([np.repeat(name, len(sample)) for name, sample in groups.items()]),
            alpha=alpha,
        )
        tukey_frame = pd.DataFrame(
            tukey._results_table.data[1:], columns=tukey._results_table.data[0]
        )
        posthoc_note = (
            "Tukey HSD pairwise comparisons are adjusted for multiple comparisons. Dunn's "
            "post-hoc test was not selected; Kruskal–Wallis is provided as an omnibus "
            "nonparametric alternative only."
        )
    interpretation = (
        f"{evidence.decision} that all group means are equal "
        f"(F({len(groups) - 1}, {len(values) - len(groups)})={classic.statistic:.4g}, "
        f"p={classic.pvalue:.4g}, eta-squared={eta_squared:.4g}). The result describes "
        "group association and does not establish causation."
    )
    if excluded_rows:
        interpretation += (
            f" {excluded_rows} row(s) with missing or non-finite values were excluded."
        )
    return OneWayAnovaResult(
        test="One-way ANOVA",
        statistic=float(classic.statistic),
        degrees_of_freedom=(float(len(groups) - 1), float(len(values) - len(groups))),
        p_value=float(classic.pvalue),
        alpha=alpha,
        decision=evidence.decision,
        interpretation=interpretation,
        group_means=mean_table,
        eta_squared=float(eta_squared),
        assumptions=checks,
        welch_statistic=float(welch.statistic),
        welch_degrees_of_freedom=(
            float(welch.df_num),
            float(welch.df_denom),
        ),
        welch_p_value=float(welch.pvalue),
        kruskal_statistic=float(kruskal.statistic),
        kruskal_degrees_of_freedom=len(groups) - 1,
        kruskal_p_value=float(kruskal.pvalue),
        tukey_comparisons=tukey_frame,
        posthoc_note=posthoc_note,
    )


def chi_square_independence(
    data: pd.DataFrame,
    row_column: str,
    column: str,
    *,
    alpha: float = config.DEFAULT_ALPHA,
    yates_correction: bool = True,
) -> ChiSquareResult:
    """Test categorical independence and return observed/expected/residual tables.

    Uses complete cases and SciPy's Pearson chi-square test. Yates' continuity correction is
    applied only to 2×2 tables when `yates_correction=True`. Fisher's exact test is additionally
    reported for 2×2 tables when the expected-count guideline is not met.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    missing_columns = [name for name in (row_column, column) if name not in data.columns]
    if missing_columns:
        raise HypothesisTestError(f"Column(s) missing from data: {', '.join(missing_columns)}.")
    complete = data[[row_column, column]].dropna()
    if complete.empty:
        raise InsufficientGroupError("Chi-square test requires at least one complete observation.")
    observed = pd.crosstab(complete[row_column], complete[column], dropna=True)
    observed = observed.loc[observed.sum(axis=1) > 0, observed.sum(axis=0) > 0]
    if observed.shape[0] < 2 or observed.shape[1] < 2:
        raise InsufficientGroupError(
            "Chi-square independence requires at least two levels per variable."
        )
    result = stats.chi2_contingency(observed.to_numpy(), correction=yates_correction)
    expected_values = np.asarray(result.expected_freq, dtype=float)
    expected = pd.DataFrame(expected_values, index=observed.index, columns=observed.columns)
    residuals = pd.DataFrame(
        (observed.to_numpy(dtype=float) - expected_values) / np.sqrt(expected_values),
        index=observed.index,
        columns=observed.columns,
    )
    count_check = expected_count_check(expected_values)
    total_n = int(observed.to_numpy().sum())
    minimum_dimension = min(observed.shape[0] - 1, observed.shape[1] - 1)
    cramers_v = float(np.sqrt(result.statistic / (total_n * minimum_dimension)))
    fisher_odds_ratio: float | None = None
    fisher_p_value: float | None = None
    if observed.shape == (2, 2) and count_check.status == "warning":
        fisher = stats.fisher_exact(observed.to_numpy(), alternative="two-sided")
        fisher_odds_ratio = float(fisher.statistic)
        fisher_p_value = float(fisher.pvalue)
    evidence = _decision(float(result.pvalue), alpha)
    yates_text = (
        "Yates' correction was applied for this 2×2 table."
        if yates_correction and observed.shape == (2, 2)
        else (
            "Yates' correction was requested but is not applied by convention outside a 2×2 table."
            if yates_correction
            else "Yates' correction was disabled."
        )
    )
    interpretation = (
        f"{evidence.decision} that '{row_column}' and '{column}' are independent "
        f"(chi-square({result.dof})={result.statistic:.4g}, p={result.pvalue:.4g}, "
        f"Cramer's V={cramers_v:.4g}). The result indicates association, not its direction "
        f"or causation. {yates_text}"
    )
    if fisher_p_value is not None:
        interpretation += (
            f" Expected counts are sparse, so Fisher's exact alternative is also reported "
            f"(odds ratio={fisher_odds_ratio:.4g}, p={fisher_p_value:.4g})."
        )
    return ChiSquareResult(
        test="Chi-square test of independence",
        statistic=float(result.statistic),
        degrees_of_freedom=int(result.dof),
        p_value=float(result.pvalue),
        alpha=alpha,
        decision=evidence.decision,
        interpretation=interpretation,
        observed=observed,
        expected=expected,
        standardized_residuals=residuals,
        cramers_v=cramers_v,
        expected_count_check=count_check,
        fisher_exact_odds_ratio=fisher_odds_ratio,
        fisher_exact_p_value=fisher_p_value,
        yates_correction=yates_correction,
    )
