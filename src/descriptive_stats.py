"""Descriptive summaries for numeric, ordinal, and categorical data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from scipy import stats

from src import config


class DescriptiveStatsError(ValueError):
    """Base exception for invalid descriptive-statistics inputs."""


class ConstantColumnError(DescriptiveStatsError):
    """Raised when a constant numeric column makes skewness undefined."""


class SingleObservationError(DescriptiveStatsError):
    """Raised when a numeric summary has fewer than two valid observations."""


class AllNaNColumnError(DescriptiveStatsError):
    """Raised when a requested column contains no valid observations."""


class NonNumericColumnError(DescriptiveStatsError):
    """Raised when a numeric statistic is requested for a non-numeric column."""


class InsufficientGroupSizeError(DescriptiveStatsError):
    """Raised when a group has fewer than two valid rows for a mean confidence interval."""


@dataclass(frozen=True)
class SummaryTableResult:
    """Numeric summaries plus measurement-level and method metadata."""

    table: pd.DataFrame
    measurement_levels: dict[str, str]
    quartile_method: str
    skewness_method: str


@dataclass(frozen=True)
class GroupSummaryResult:
    """Per-group summaries and confidence intervals for group means."""

    table: pd.DataFrame
    confidence_level: float
    measurement_level: str
    quartile_method: str
    skewness_method: str
    missing_group_rows: int


@dataclass(frozen=True)
class FrequencyTableResult:
    """Observed category counts and percentages with missingness metadata."""

    table: pd.DataFrame
    total_count: int
    valid_count: int
    missing_count: int
    measurement_level: str


QUARTILE_METHOD = "pandas Series.quantile default interpolation='linear'"
SKEWNESS_METHOD = (
    "Adjusted Fisher-Pearson standardized moment coefficient (Pandas Series.skew, bias-corrected)"
)


def _get_numeric_series(data: pd.DataFrame, column: str) -> pd.Series:
    """Validate a requested numeric column and return it with missing values preserved."""
    if column not in data.columns:
        raise DescriptiveStatsError(f"Column '{column}' is not present in the dataset.")
    series = data[column]
    if series.isna().all():
        raise AllNaNColumnError(f"Column '{column}' contains no valid observations.")
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise NonNumericColumnError(
            f"Column '{column}' is not numeric; numeric statistics cannot be calculated."
        )
    values = series.dropna()
    if len(values) < 2:
        raise SingleObservationError(
            f"Column '{column}' has only one valid observation; sample spread and skewness "
            "cannot be summarized."
        )
    if values.nunique(dropna=True) == 1:
        raise ConstantColumnError(f"Column '{column}' is constant; its skewness is undefined.")
    return series


def _summary_values(series: pd.Series) -> dict[str, Any]:
    """Compute one validated numeric series' descriptive statistics."""
    values = series.dropna()
    quartiles = values.quantile([0.25, 0.5, 0.75], interpolation="linear")
    modes = tuple(values.mode(dropna=True).tolist())
    return {
        "count": int(values.count()),
        "missing_count": int(series.isna().sum()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "mode": modes,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "range": float(values.max() - values.min()),
        "variance": float(values.var(ddof=1)),
        "standard_deviation": float(values.std(ddof=1)),
        "Q1": float(quartiles.loc[0.25]),
        "Q2": float(quartiles.loc[0.5]),
        "Q3": float(quartiles.loc[0.75]),
        "IQR": float(quartiles.loc[0.75] - quartiles.loc[0.25]),
        "skewness": float(values.skew()),
    }


def _measurement_level(column: str, series: pd.Series | None = None) -> str:
    """Classify known project columns as ordinal or numeric for interpretation."""
    if column in config.ORDINAL_COLUMNS:
        return "ordinal"
    if series is not None and isinstance(series.dtype, pd.CategoricalDtype):
        return "ordinal" if series.dtype.ordered else "categorical"
    if column in config.NOMINAL_COLUMNS or column in config.BINARY_COLUMNS:
        return "categorical"
    if series is not None and not pd.api.types.is_numeric_dtype(series.dtype):
        return "categorical"
    return "numeric"


def summary_table(data: pd.DataFrame, columns: Sequence[str]) -> SummaryTableResult:
    """Summarize requested numeric columns, preserving tied modes and missing counts.

    Assumes each selected column is numeric. Ordinal-coded project variables may be summarized
    numerically for convenience but are identified as ordinal in the result metadata. Quartiles
    use pandas' linear interpolation (`Series.quantile(..., interpolation="linear")`). Skewness is
    pandas' adjusted Fisher-Pearson standardized moment coefficient (the bias-corrected G1
    estimator, equivalent to `scipy.stats.skew(..., bias=False)`). Its formula is
    `G1 = sqrt(n*(n-1))/(n-2) * m3/(m2**1.5)`, where `m2` and `m3` are the second and third
    central moments computed with denominator `n`. Constant columns, one valid
    observation, all-missing columns, and non-numeric columns raise descriptive custom exceptions.
    """
    if data.empty:
        raise DescriptiveStatsError("Cannot summarize an empty dataset.")
    if not columns:
        raise DescriptiveStatsError("Select at least one numeric column to summarize.")
    if len(set(columns)) != len(columns):
        raise DescriptiveStatsError("The selected column list contains duplicates.")

    records: dict[str, dict[str, Any]] = {}
    for column in columns:
        values = _get_numeric_series(data, column)
        records[column] = _summary_values(values)
    table = pd.DataFrame.from_dict(records, orient="index")
    table.index.name = "variable"
    return SummaryTableResult(
        table=table,
        measurement_levels={column: _measurement_level(column) for column in columns},
        quartile_method=QUARTILE_METHOD,
        skewness_method=SKEWNESS_METHOD,
    )


def _mean_confidence_interval(values: pd.Series, confidence_level: float) -> tuple[float, float]:
    """Compute a two-sided Student-t confidence interval for a sample mean."""
    count = int(values.count())
    if count < 2:
        raise InsufficientGroupSizeError(
            "A group needs at least two valid observations to calculate a mean confidence interval."
        )
    mean = float(values.mean())
    standard_error = float(stats.sem(values))
    critical_value = float(stats.t.ppf((1 + confidence_level) / 2, df=count - 1))
    margin = critical_value * standard_error
    return mean - margin, mean + margin


def group_summary(
    data: pd.DataFrame,
    value_col: str,
    group_col: str,
    confidence_level: float = config.DEFAULT_CONFIDENCE,
) -> GroupSummaryResult:
    """Summarize a numeric/ordinal variable by group and attach a Student-t mean confidence interval.

    Each group must have at least two non-missing numeric observations. Missing values in the value
    column are excluded from statistics and counted; rows with missing group labels are excluded
    from groups and reported separately. Ordinal-coded project variables are marked ordinal.
    Quartiles use pandas linear interpolation; skewness uses the bias-corrected Fisher-Pearson
    G1 estimator from `Series.skew`.
    """
    if not 0 < confidence_level < 1:
        raise DescriptiveStatsError("confidence_level must be strictly between 0 and 1.")
    if data.empty:
        raise DescriptiveStatsError("Cannot group an empty dataset.")
    missing_columns = [column for column in (value_col, group_col) if column not in data.columns]
    if missing_columns:
        raise DescriptiveStatsError(f"Column(s) missing: {', '.join(missing_columns)}.")

    series = data[value_col]
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise NonNumericColumnError(
            f"Column '{value_col}' is not numeric; numeric statistics cannot be calculated."
        )

    missing_group_rows = int(data[group_col].isna().sum())
    group_records: list[dict[str, Any]] = []
    grouped = data.loc[data[group_col].notna()].groupby(group_col, sort=False, observed=True)
    for group_value, frame in grouped:
        non_missing = frame[value_col].dropna()
        if len(non_missing) < 2:
            raise InsufficientGroupSizeError(
                f"Group '{group_value}' has {len(non_missing)} valid row(s); at least 2 are "
                "required to calculate a mean confidence interval."
            )
        if non_missing.nunique(dropna=True) == 1:
            raise ConstantColumnError(
                f"Column '{value_col}' is constant in group '{group_value}'; its skewness is undefined."
            )
        record: dict[str, Any] = {"group": group_value, "group_size": len(frame)}
        record.update(_summary_values(frame[value_col]))
        ci_lower, ci_upper = _mean_confidence_interval(non_missing, confidence_level)
        record["mean_ci_lower"] = ci_lower
        record["mean_ci_upper"] = ci_upper
        group_records.append(record)

    if not group_records:
        raise DescriptiveStatsError("No non-missing group labels are available to summarize.")
    return GroupSummaryResult(
        table=pd.DataFrame(group_records),
        confidence_level=confidence_level,
        measurement_level=_measurement_level(value_col, series),
        quartile_method=QUARTILE_METHOD,
        skewness_method=SKEWNESS_METHOD,
        missing_group_rows=missing_group_rows,
    )


def frequency_table(data: pd.DataFrame, column: str) -> FrequencyTableResult:
    """Return observed-category counts and percentages, excluding missing values from the denominator.

    Ordered pandas categoricals retain their declared order. Numeric-coded project ordinal columns
    are sorted by their numeric code; nominal and other categorical values retain first-seen order.
    Percentages are computed over non-missing observations, and missing rows are reported in result
    metadata rather than represented as an invented category.
    """
    if column not in data.columns:
        raise DescriptiveStatsError(f"Column '{column}' is not present in the dataset.")
    series = data[column]
    valid = series.dropna()
    if valid.empty:
        raise AllNaNColumnError(f"Column '{column}' contains no valid observations.")

    counts = valid.value_counts(sort=False, dropna=True)
    if isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered:
        counts = counts.reindex(series.cat.categories, fill_value=0)
    elif column in config.ORDINAL_COLUMNS and pd.api.types.is_numeric_dtype(series.dtype):
        counts = counts.sort_index()

    table = pd.DataFrame(
        {
            "category": counts.index.to_list(),
            "count": counts.astype(int).to_numpy(),
            "percentage": (counts / len(valid) * 100).to_numpy(),
        }
    )
    return FrequencyTableResult(
        table=table,
        total_count=len(series),
        valid_count=len(valid),
        missing_count=int(series.isna().sum()),
        measurement_level=_measurement_level(column, series),
    )
