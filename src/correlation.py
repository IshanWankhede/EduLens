"""Pearson and Spearman correlation analyses for numeric and ordinal variables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from src import config

CorrelationMethod = Literal["pearson", "spearman"]
AssociationMethod = Literal["auto", "pearson", "spearman"]


class CorrelationAnalysisError(ValueError):
    """Raised when a correlation cannot be calculated from the requested columns."""


@dataclass(frozen=True)
class CorrelationMatrixResult:
    """Correlation coefficients, p-values, pair counts, and optional Pearson CIs."""

    method: CorrelationMethod
    coefficients: pd.DataFrame
    p_values: pd.DataFrame
    sample_sizes: pd.DataFrame
    ci_lower: pd.DataFrame | None
    ci_upper: pd.DataFrame | None
    confidence_level: float | None
    recommended_spearman: dict[str, bool]


@dataclass(frozen=True)
class CorrelationMatricesResult:
    """Both correlation matrices, computed on pairwise-complete observations."""

    pearson: CorrelationMatrixResult
    spearman: CorrelationMatrixResult


@dataclass(frozen=True)
class RankedAssociationsResult:
    """A selectable target's strongest pairwise associations, ranked by absolute coefficient."""

    target: str
    table: pd.DataFrame
    method_requested: AssociationMethod
    strength_thresholds: str


STRENGTH_THRESHOLDS = "weak: |r| < 0.3; moderate: 0.3 <= |r| <= 0.5; strong: |r| > 0.5"


def recommended_method(column_a: str, column_b: str) -> CorrelationMethod:
    """Recommend Spearman if either project column has an ordinal measurement role."""
    return (
        "spearman"
        if column_a in config.ORDINAL_COLUMNS or column_b in config.ORDINAL_COLUMNS
        else "pearson"
    )


def strength_label(coefficient: float) -> str:
    """Label association magnitude using the documented absolute-correlation thresholds."""
    if not np.isfinite(coefficient):
        raise CorrelationAnalysisError("A finite correlation coefficient is required for labeling.")
    magnitude = abs(coefficient)
    if magnitude < 0.3:
        return "weak"
    if magnitude <= 0.5:
        return "moderate"
    return "strong"


def _select_columns(data: pd.DataFrame, columns: list[str] | None) -> list[str]:
    """Choose and validate the requested numeric and ordinal columns."""
    if data.empty:
        raise CorrelationAnalysisError("Correlation analysis requires a non-empty dataset.")
    selected = (
        list(columns)
        if columns is not None
        else [
            column
            for column in config.NUMERIC_COLUMNS + config.ORDINAL_COLUMNS
            if column in data.columns
        ]
    )
    if len(selected) < 2:
        raise CorrelationAnalysisError("At least two numeric or ordinal columns are required.")
    if len(selected) != len(set(selected)):
        raise CorrelationAnalysisError("Correlation column names must be unique.")
    missing = [column for column in selected if column not in data.columns]
    if missing:
        raise CorrelationAnalysisError(
            f"Correlation column(s) missing from dataset: {', '.join(missing)}."
        )
    non_numeric = [
        column
        for column in selected
        if not pd.api.types.is_numeric_dtype(data[column].dtype)
        or pd.api.types.is_bool_dtype(data[column].dtype)
    ]
    if non_numeric:
        raise CorrelationAnalysisError(
            f"Correlation columns must be numeric or ordinal codes: {', '.join(non_numeric)}."
        )
    return selected


def _fisher_z_interval(
    coefficient: float,
    n: int,
    confidence_level: float,
) -> tuple[float, float]:
    """Return a two-sided Fisher-z confidence interval for a Pearson coefficient."""
    if n <= 3:
        return (float("nan"), float("nan"))
    if abs(coefficient) == 1:
        return (coefficient, coefficient)
    z = np.arctanh(coefficient)
    critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    margin = critical / np.sqrt(n - 3)
    return float(np.tanh(z - margin)), float(np.tanh(z + margin))


def _correlation_matrix(
    data: pd.DataFrame,
    columns: list[str] | None,
    method: CorrelationMethod,
    confidence_level: float,
) -> CorrelationMatrixResult:
    """Calculate pairwise coefficients and p-values, plus Pearson Fisher-z intervals."""
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    selected = _select_columns(data, columns)
    coefficients = pd.DataFrame(np.nan, index=selected, columns=selected, dtype=float)
    p_values = coefficients.copy()
    sample_sizes = pd.DataFrame(0, index=selected, columns=selected, dtype=int)
    ci_lower = coefficients.copy() if method == "pearson" else None
    ci_upper = coefficients.copy() if method == "pearson" else None

    for row_index, column_a in enumerate(selected):
        for column_b in selected[row_index:]:
            pair_columns = list(dict.fromkeys((column_a, column_b)))
            complete = data[pair_columns].dropna()
            complete = complete.loc[
                np.isfinite(complete[column_a]) & np.isfinite(complete[column_b])
            ]
            n = len(complete)
            if n < 3:
                raise CorrelationAnalysisError(
                    f"Columns '{column_a}' and '{column_b}' have only {n} complete rows; "
                    "at least three are required."
                )
            values_a = complete[column_a].to_numpy(dtype=float)
            values_b = complete[column_b].to_numpy(dtype=float)
            if np.unique(values_a).size < 2 or np.unique(values_b).size < 2:
                raise CorrelationAnalysisError(
                    f"Correlation is undefined because '{column_a}' or '{column_b}' is constant."
                )
            if column_a == column_b:
                coefficient, p_value = 1.0, 0.0
            elif method == "pearson":
                result = stats.pearsonr(values_a, values_b)
                coefficient, p_value = float(result.statistic), float(result.pvalue)
            else:
                result = stats.spearmanr(values_a, values_b)
                coefficient, p_value = float(result.statistic), float(result.pvalue)
                if not np.isfinite(coefficient) or not np.isfinite(p_value):
                    raise CorrelationAnalysisError(
                        f"Spearman correlation is undefined for '{column_a}' and '{column_b}'."
                    )
            coefficients.loc[column_a, column_b] = coefficient
            coefficients.loc[column_b, column_a] = coefficient
            p_values.loc[column_a, column_b] = p_value
            p_values.loc[column_b, column_a] = p_value
            sample_sizes.loc[column_a, column_b] = n
            sample_sizes.loc[column_b, column_a] = n
            if method == "pearson":
                lower, upper = _fisher_z_interval(coefficient, n, confidence_level)
                assert ci_lower is not None and ci_upper is not None
                ci_lower.loc[column_a, column_b] = lower
                ci_lower.loc[column_b, column_a] = lower
                ci_upper.loc[column_a, column_b] = upper
                ci_upper.loc[column_b, column_a] = upper

    return CorrelationMatrixResult(
        method=method,
        coefficients=coefficients,
        p_values=p_values,
        sample_sizes=sample_sizes,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level if method == "pearson" else None,
        recommended_spearman={column: column in config.ORDINAL_COLUMNS for column in selected},
    )


def pearson_matrix(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    confidence_level: float = config.DEFAULT_CONFIDENCE,
) -> CorrelationMatrixResult:
    """Return Pearson coefficients, two-sided p-values, pair counts, and Fisher-z intervals.

    Pairwise-complete observations are used. Inputs should be numeric and reasonably suited to
    linear-association analysis; confidence intervals use Fisher's z transform.
    """
    return _correlation_matrix(data, columns, "pearson", confidence_level)


def spearman_matrix(
    data: pd.DataFrame,
    columns: list[str] | None = None,
) -> CorrelationMatrixResult:
    """Return Spearman rank coefficients, p-values, and pairwise sample counts.

    Pairwise-complete numeric or ordinal-coded observations are used. Spearman is recommended
    automatically in ranked associations whenever either variable is ordinal.
    """
    return _correlation_matrix(data, columns, "spearman", config.DEFAULT_CONFIDENCE)


def correlation_matrices(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    confidence_level: float = config.DEFAULT_CONFIDENCE,
) -> CorrelationMatricesResult:
    """Compute Pearson and Spearman matrices for the same requested columns."""
    return CorrelationMatricesResult(
        pearson=pearson_matrix(data, columns, confidence_level=confidence_level),
        spearman=spearman_matrix(data, columns),
    )


def ranked_associations(
    data: pd.DataFrame,
    target: str,
    columns: list[str] | None = None,
    *,
    method: AssociationMethod = "auto",
) -> RankedAssociationsResult:
    """Rank pairwise associations with a target, recommending Spearman for ordinal pairs.

    `auto` selects Spearman whenever the target or comparison column is in the configured
    ordinal roles; otherwise it selects Pearson. Strength labels follow `STRENGTH_THRESHOLDS`.
    """
    selected = _select_columns(data, columns)
    if target not in selected:
        raise CorrelationAnalysisError(
            f"Target '{target}' must be included in the selected correlation columns."
        )
    if method not in {"auto", "pearson", "spearman"}:
        raise ValueError("method must be 'auto', 'pearson', or 'spearman'.")
    records: list[dict[str, object]] = []
    for column in selected:
        if column == target:
            continue
        pair_method: CorrelationMethod = (
            recommended_method(target, column) if method == "auto" else method
        )
        result = _correlation_matrix(
            data,
            [target, column],
            pair_method,
            config.DEFAULT_CONFIDENCE,
        )
        coefficient = float(result.coefficients.loc[target, column])
        records.append(
            {
                "variable": column,
                "method": pair_method,
                "coefficient": coefficient,
                "p_value": float(result.p_values.loc[target, column]),
                "n": int(result.sample_sizes.loc[target, column]),
                "strength": strength_label(coefficient),
            }
        )
    table = pd.DataFrame(
        records,
        columns=["variable", "method", "coefficient", "p_value", "n", "strength"],
    )
    if not table.empty:
        table = (
            table.assign(_absolute=table["coefficient"].abs())
            .sort_values("_absolute", ascending=False, kind="stable")
            .drop(columns="_absolute")
            .reset_index(drop=True)
        )
    return RankedAssociationsResult(
        target=target,
        table=table,
        method_requested=method,
        strength_thresholds=STRENGTH_THRESHOLDS,
    )
