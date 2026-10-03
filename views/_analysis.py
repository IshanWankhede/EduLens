"""Shared accessors for analysis-page data and current statistical settings."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import DatasetBundle


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
