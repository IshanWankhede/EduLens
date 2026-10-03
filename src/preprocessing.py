"""Non-destructive cleaning reports, encoders, and target-category derivation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from src import config
from src.validation import DataValidationError


@dataclass(frozen=True)
class OutlierSummary:
    """Potential IQR outliers for one numeric column; values are reported, not removed."""

    lower_fence: float | None
    upper_fence: float | None
    count: int


@dataclass(frozen=True)
class CleaningReport:
    """Dataset quality counts and numeric summaries before and after non-destructive cleaning."""

    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    missing_cells: int
    missing_by_column: Mapping[str, int]
    exact_duplicate_rows: int
    dtypes: Mapping[str, str]
    outliers: Mapping[str, OutlierSummary]
    numeric_summary_before: Mapping[str, Mapping[str, float | int | None]]
    numeric_summary_after: Mapping[str, Mapping[str, float | int | None]]


@dataclass(frozen=True)
class PerformanceCategoryResult:
    """Performance labels, fixed-band metadata, class counts, and unclassified count."""

    categories: pd.Series
    method: str
    thresholds: Mapping[str, int]
    class_counts: Mapping[str, int]
    unclassified_count: int


def _numeric_summary(dataframe: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, float | int | None]] = {}
    for column in dataframe.select_dtypes(include="number").columns:
        values = dataframe[column].dropna()
        summary[column] = {
            "count": int(values.count()),
            "mean": float(values.mean()) if not values.empty else None,
            "median": float(values.median()) if not values.empty else None,
            "minimum": float(values.min()) if not values.empty else None,
            "maximum": float(values.max()) if not values.empty else None,
        }
    return summary


def _outlier_summary(dataframe: pd.DataFrame) -> dict[str, OutlierSummary]:
    outliers: dict[str, OutlierSummary] = {}
    for column in dataframe.select_dtypes(include="number").columns:
        values = dataframe[column].dropna()
        if values.empty:
            outliers[column] = OutlierSummary(None, None, 0)
            continue
        first_quartile = float(values.quantile(0.25))
        third_quartile = float(values.quantile(0.75))
        spread = third_quartile - first_quartile
        lower_fence = first_quartile - 1.5 * spread
        upper_fence = third_quartile + 1.5 * spread
        count = int(((values < lower_fence) | (values > upper_fence)).sum())
        outliers[column] = OutlierSummary(lower_fence, upper_fence, count)
    return outliers


def build_cleaning_report(
    before: pd.DataFrame, after: pd.DataFrame | None = None
) -> CleaningReport:
    """Report missingness, duplicates, IQR outliers, and numeric summaries without mutating data.

    Missing values, duplicate rows, and outliers are measured, not automatically dropped or
    imputed. Quartile fences use pandas' default linear interpolation.
    """
    cleaned = before if after is None else after
    missing = before.isna().sum()
    return CleaningReport(
        rows_before=len(before),
        rows_after=len(cleaned),
        columns_before=len(before.columns),
        columns_after=len(cleaned.columns),
        missing_cells=int(missing.sum()),
        missing_by_column={str(k): int(v) for k, v in missing[missing > 0].items()},
        exact_duplicate_rows=int(before.duplicated().sum()),
        dtypes={str(column): str(dtype) for column, dtype in before.dtypes.items()},
        outliers=_outlier_summary(before),
        numeric_summary_before=_numeric_summary(before),
        numeric_summary_after=_numeric_summary(cleaned),
    )


def clean_dataset(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Return a copy with a G3-zero flag and a report; do not drop, impute, or alter source rows.

    The G3-zero flag is added only when G3 exists. The observed grade itself remains unchanged.
    """
    cleaned = dataframe.copy()
    if "G3" in cleaned.columns:
        grades = pd.to_numeric(cleaned["G3"], errors="coerce")
        cleaned["G3_zero_flag"] = grades.eq(0).astype("boolean")
        cleaned.loc[grades.isna(), "G3_zero_flag"] = pd.NA
    return cleaned, build_cleaning_report(dataframe, cleaned)


def encode_binary(series: pd.Series, *, positive_label: str) -> pd.Series:
    """Encode a two-category series as nullable integers, with the requested label mapped to 1.

    The input must have exactly two distinct non-missing values. Missing values remain missing.
    """
    labels = series.dropna().astype(str).str.strip()
    categories = labels.unique().tolist()
    positive = positive_label.strip().casefold()
    normalized = {category.casefold() for category in categories}
    if len(categories) != 2 or len(normalized) != 2:
        raise DataValidationError(
            f"Binary encoding requires exactly two non-missing categories; found {len(categories)}."
        )
    if positive not in normalized:
        raise DataValidationError(
            f"Positive label '{positive_label}' is not one of the column's categories."
        )

    mapping = {category: int(category.casefold() == positive) for category in categories}
    result = series.map(lambda value: pd.NA if pd.isna(value) else mapping[str(value).strip()])
    return result.astype("Int64")


def encode_nominal(
    dataframe: pd.DataFrame, columns: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """One-hot encode nominal columns using the lexicographically first category as reference.

    The input is not mutated. Missing categories produce all-zero dummy values and are not imputed.
    """
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise DataValidationError(f"Nominal column(s) missing: {', '.join(missing)}.")

    encoded = dataframe.drop(columns=list(columns)).copy()
    references: dict[str, str] = {}
    for column in columns:
        values = dataframe[column].dropna().astype(str)
        categories = sorted(values.unique().tolist())
        if len(categories) < 2:
            raise DataValidationError(
                f"Nominal encoding for '{column}' requires at least two categories."
            )
        references[column] = categories[0]
        category_series = pd.Categorical(
            dataframe[column].astype("string"), categories=categories, ordered=False
        )
        dummies = pd.get_dummies(
            category_series,
            prefix=column,
            drop_first=True,
            dtype=int,
        )
        dummies.index = dataframe.index
        encoded = encoded.join(dummies)
    return encoded, references


def derive_performance_category(
    dataframe: pd.DataFrame, grade_column: str = "G3"
) -> PerformanceCategoryResult:
    """Derive Low/Medium/High labels using the fixed configured grade bands.

    Scores below 10 are Low, 10 through 13 inclusive are Medium, and scores of at least 14 are
    High. Missing or non-numeric scores remain unclassified; no rows are removed.
    """
    if grade_column not in dataframe.columns:
        raise DataValidationError(f"Grade column '{grade_column}' is missing.")
    scores = pd.to_numeric(dataframe[grade_column], errors="coerce")
    invalid = dataframe[grade_column].notna() & scores.isna()
    if invalid.any():
        raise DataValidationError(
            f"Grade column '{grade_column}' contains {int(invalid.sum())} non-numeric value(s)."
        )

    bands = config.PERFORMANCE_BANDS
    labels = pd.Series(pd.NA, index=dataframe.index, dtype="string", name="PerformanceCategory")
    labels.loc[scores < bands["LOW_UPPER_EXCLUSIVE"]] = "Low"
    labels.loc[
        scores.between(
            bands["MEDIUM_LOWER_INCLUSIVE"],
            bands["MEDIUM_UPPER_INCLUSIVE"],
            inclusive="both",
        )
    ] = "Medium"
    labels.loc[scores >= bands["HIGH_LOWER_INCLUSIVE"]] = "High"
    counts = {
        "Low": int(labels.eq("Low").sum()),
        "Medium": int(labels.eq("Medium").sum()),
        "High": int(labels.eq("High").sum()),
    }
    thresholds = {
        "low_upper_exclusive": bands["LOW_UPPER_EXCLUSIVE"],
        "medium_lower_inclusive": bands["MEDIUM_LOWER_INCLUSIVE"],
        "medium_upper_inclusive": bands["MEDIUM_UPPER_INCLUSIVE"],
        "high_lower_inclusive": bands["HIGH_LOWER_INCLUSIVE"],
    }
    return PerformanceCategoryResult(
        categories=labels,
        method="fixed grade bands",
        thresholds=thresholds,
        class_counts=counts,
        unclassified_count=int(labels.isna().sum()),
    )
