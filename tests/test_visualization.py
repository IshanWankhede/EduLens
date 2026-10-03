from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.data_loader import load_uci_dataset
from src.visualization import (
    VisualizationInputError,
    absences_vs_g3,
    apply_plotly_theme,
    bar_count_plot,
    box_plot,
    correlation_heatmap,
    failures_vs_g3,
    famsup_vs_g3,
    freetime_vs_g3,
    g1_vs_g3,
    g2_vs_g3,
    grouped_comparison,
    health_vs_g3,
    histogram_plot,
    kde_plot,
    parental_education_vs_g3,
    qq_plot,
    scatter_plot,
    studytime_vs_g3,
    violin_plot,
)


@pytest.fixture
def cleaned_por() -> pd.DataFrame:
    return load_uci_dataset("por").clean


@pytest.fixture
def edge_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": [1.0, 2.0, 2.0, 4.0, 6.0, 8.0],
            "constant": [3.0] * 6,
            "category": ["a", "a", "b", "b", "c", "c"],
            "studytime": [1, 1, 2, 2, 3, 3],
            "outcome": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "G1": [4, 5, 6, 7, 8, 9],
            "G2": [5, 6, 7, 8, 9, 10],
            "G3": [5, 6, 7, 8, 9, 10],
            "absences": [0, 1, 2, 3, 4, 5],
            "failures": [0, 1, 0, 2, 1, 0],
            "famsup": ["yes", "no", "yes", "no", "yes", "no"],
            "Medu": [0, 1, 2, 3, 4, 1],
            "Fedu": [1, 2, 3, 4, 0, 2],
            "health": [1, 2, 3, 4, 5, 2],
            "freetime": [5, 4, 3, 2, 1, 4],
        }
    )


def _assert_plotly_chart_contract(figure: go.Figure) -> None:
    assert isinstance(figure, go.Figure)
    assert figure.layout.title.text
    assert figure.layout.xaxis.title.text
    assert figure.layout.yaxis.title.text
    assert figure.layout.meta["caption"]
    assert any("How to read this:" in annotation.text for annotation in figure.layout.annotations)


def test_all_generic_plotly_builders_create_figures_from_real_cleaned_dataset(
    cleaned_por: pd.DataFrame,
) -> None:
    correlation = cleaned_por[["G1", "G2", "G3", "absences"]].corr()
    builders = [
        histogram_plot(cleaned_por, "G3"),
        kde_plot(cleaned_por, "absences"),
        box_plot(cleaned_por, "G3", "studytime"),
        box_plot(cleaned_por, "G3"),
        violin_plot(cleaned_por, "G3", "studytime"),
        violin_plot(cleaned_por, "G3"),
        bar_count_plot(cleaned_por, "studytime"),
        bar_count_plot(cleaned_por, "famsup", probability=True),
        scatter_plot(cleaned_por, "absences", "G3", trend_line=True),
        scatter_plot(cleaned_por, "studytime", "G3"),
        correlation_heatmap(correlation),
        grouped_comparison(cleaned_por, "studytime", "G3"),
    ]

    for figure in builders:
        _assert_plotly_chart_contract(figure)


def test_presets_create_figures_from_real_cleaned_portuguese_data(
    cleaned_por: pd.DataFrame,
) -> None:
    figures = [
        studytime_vs_g3(cleaned_por),
        absences_vs_g3(cleaned_por),
        failures_vs_g3(cleaned_por),
        g1_vs_g3(cleaned_por),
        g2_vs_g3(cleaned_por),
        famsup_vs_g3(cleaned_por),
        parental_education_vs_g3(cleaned_por),
        health_vs_g3(cleaned_por),
        freetime_vs_g3(cleaned_por),
    ]

    for figure in figures:
        _assert_plotly_chart_contract(figure)
    parent_chart = figures[6]
    subplot_titles = [annotation.text for annotation in parent_chart.layout.annotations]
    assert "Mother's education (Medu)" in subplot_titles
    assert "Father's education (Fedu)" in subplot_titles


def test_presets_create_figures_from_small_hand_checkable_fixture(
    edge_data: pd.DataFrame,
) -> None:
    figures = [
        studytime_vs_g3(edge_data),
        absences_vs_g3(edge_data),
        failures_vs_g3(edge_data),
        g1_vs_g3(edge_data),
        g2_vs_g3(edge_data),
        famsup_vs_g3(edge_data),
        parental_education_vs_g3(edge_data),
        health_vs_g3(edge_data),
        freetime_vs_g3(edge_data),
    ]

    for figure in figures:
        _assert_plotly_chart_contract(figure)


def test_qq_builder_returns_labeled_matplotlib_figure_from_real_data(
    cleaned_por: pd.DataFrame,
) -> None:
    figure = qq_plot(cleaned_por["G3"])

    assert isinstance(figure, plt.Figure)
    axis = figure.axes[0]
    assert axis.get_title()
    assert axis.get_xlabel()
    assert axis.get_ylabel()
    assert any("How to read this:" in text.get_text() for text in figure.texts)
    plt.close(figure)


def test_generic_builders_create_figures_from_small_hand_checkable_fixture(
    edge_data: pd.DataFrame,
) -> None:
    figures = [
        histogram_plot(edge_data, "value", bins=4),
        kde_plot(edge_data, "value"),
        box_plot(edge_data, "outcome", "category"),
        violin_plot(edge_data, "outcome", "category"),
        bar_count_plot(edge_data, "category"),
        bar_count_plot(edge_data, "category", probability=True),
        scatter_plot(edge_data, "value", "outcome", trend_line=True),
        scatter_plot(edge_data, "studytime", "outcome"),
        correlation_heatmap(edge_data[["value", "outcome"]].corr()),
        grouped_comparison(edge_data, "category", "outcome"),
    ]

    for figure in figures:
        _assert_plotly_chart_contract(figure)


def test_empty_inputs_raise_friendly_visualization_exception() -> None:
    empty = pd.DataFrame(columns=["x", "y", "group"])
    builders = [
        lambda: histogram_plot(empty, "x"),
        lambda: kde_plot(empty, "x"),
        lambda: box_plot(empty, "y", "group"),
        lambda: violin_plot(empty, "y", "group"),
        lambda: bar_count_plot(empty, "group"),
        lambda: scatter_plot(empty, "x", "y"),
        lambda: grouped_comparison(empty, "group", "y"),
        lambda: correlation_heatmap(pd.DataFrame()),
        lambda: qq_plot([]),
    ]

    for build in builders:
        with pytest.raises(VisualizationInputError, match="empty|non-empty"):
            build()


def test_constant_columns_are_rejected_for_density_estimators(edge_data: pd.DataFrame) -> None:
    with pytest.raises(VisualizationInputError, match="constant"):
        kde_plot(edge_data, "constant")
    with pytest.raises(VisualizationInputError, match="constant"):
        violin_plot(edge_data, "constant")
    with pytest.raises(VisualizationInputError, match="variation"):
        qq_plot(edge_data["constant"])


def test_ordinal_kde_is_rejected_with_explanation(cleaned_por: pd.DataFrame) -> None:
    with pytest.raises(VisualizationInputError, match="ordinal"):
        kde_plot(cleaned_por, "studytime")


def test_probability_bar_axis_is_zero_to_one_hundred_percent(edge_data: pd.DataFrame) -> None:
    figure = bar_count_plot(edge_data, "category", probability=True)

    assert figure.layout.yaxis.range == (0, 100)
    assert figure.layout.yaxis.title.text == "Share of valid observations (%)"


def test_theme_requires_title_labels_and_caption() -> None:
    with pytest.raises(VisualizationInputError, match="title"):
        apply_plotly_theme(
            go.Figure(),
            title="",
            x_label="x",
            y_label="y",
            caption="Caption.",
        )
