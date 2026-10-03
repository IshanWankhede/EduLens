"""Shared accessors for analysis-page data and current statistical settings."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import DatasetBundle
from src.prediction import (
    FeatureSet,
    ModelComparison,
    PredictionRun,
    compare_models,
    evaluate_model,
)
from src.regression import RegressionResult, fit_ols


def selected_bundle() -> DatasetBundle | None:
    """Return the selected dataset bundle when one is available."""
    bundle = st.session_state.get("edulens_dataset_bundle")
    return bundle if isinstance(bundle, DatasetBundle) else None


def selected_data() -> pd.DataFrame | None:
    """Return the cleaned frame for statistical analysis."""
    bundle = selected_bundle()
    return None if bundle is None else bundle.clean


def analysis_settings() -> dict[str, Any]:
    """Return confidence, alpha, and seed settings, with project defaults as fallback."""
    settings = st.session_state.get("edulens_settings")
    result = settings if isinstance(settings, dict) else {}
    return {
        "confidence_level": float(result.get("confidence_level", config.DEFAULT_CONFIDENCE)),
        "alpha": float(result.get("alpha", config.DEFAULT_ALPHA)),
        "thresholds": result.get("thresholds", config.PERFORMANCE_BANDS),
    }


def numeric_columns(data: pd.DataFrame) -> list[str]:
    """List numeric analysis columns, excluding Boolean helper flags."""
    return [
        str(column)
        for column in data.select_dtypes(include="number").columns
        if not pd.api.types.is_bool_dtype(data[column].dtype)
    ]


def ordinal_columns(data: pd.DataFrame) -> list[str]:
    """List available columns documented as ordinal."""
    return [column for column in config.ORDINAL_COLUMNS if column in data.columns]


def categorical_columns(data: pd.DataFrame) -> list[str]:
    """List available category/ordinal columns suitable for group selection."""
    ordered = config.ORDINAL_COLUMNS + config.NOMINAL_COLUMNS + config.BINARY_COLUMNS
    return list(dict.fromkeys(column for column in ordered if column in data.columns))


@st.cache_resource(show_spinner="Fitting regression model and diagnostics…")
def cached_ols(
    dataset_key: str,
    data: pd.DataFrame,
    target: str,
    predictors: tuple[str, ...],
    confidence_level: float,
    robust_se: bool,
    ordinal_mode: str,
    allow_model_b: bool,
    settings_key: tuple[object, ...],
) -> RegressionResult:
    """Cache OLS results by dataset, model settings, and selected terms."""
    del dataset_key, settings_key
    return fit_ols(
        data,
        target,
        predictors,
        confidence_level=confidence_level,
        robust_se=robust_se,
        ordinal_mode=ordinal_mode,
        allow_model_b=allow_model_b,
    )


@st.cache_resource(show_spinner="Training and evaluating the selected classifier…")
def cached_prediction_run(
    dataset_key: str,
    data: pd.DataFrame,
    feature_set: FeatureSet,
    seed: int,
    test_size: float,
    cv_folds: int,
    settings_key: tuple[object, ...],
) -> PredictionRun:
    """Cache one trained prediction pipeline by dataset and evaluation settings."""
    del dataset_key, settings_key
    return evaluate_model(
        data,
        feature_set=feature_set,
        seed=seed,
        test_size=test_size,
        cv_folds=cv_folds,
    )


@st.cache_resource(show_spinner="Comparing Model A and Model B on identical splits…")
def cached_model_comparison(
    dataset_key: str,
    data: pd.DataFrame,
    seed: int,
    test_size: float,
    cv_folds: int,
    settings_key: tuple[object, ...],
) -> ModelComparison:
    """Cache paired model evaluation by dataset and all active evaluation settings."""
    del dataset_key, settings_key
    return compare_models(data, seed=seed, test_size=test_size, cv_folds=cv_folds)
