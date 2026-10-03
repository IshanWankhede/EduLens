"""Dynamic exploratory visualizations backed by the shared figure builders."""

from __future__ import annotations

import streamlit as st

from src.visualization import (
    VisualizationInputError,
    bar_count_plot,
    box_plot,
    grouped_comparison,
    histogram_plot,
    kde_plot,
    scatter_plot,
    violin_plot,
)
from views._analysis import numeric_columns, selected_data

st.title("Exploratory Analysis")
st.markdown(
    '<p class="el-caption">Choose variables and a chart type to inspect observed distributions '
    "and associations. Charts are descriptive and do not establish causation.</p>",
    unsafe_allow_html=True,
)
data = selected_data()
if data is None:
    st.info("Select or upload a dataset from the sidebar to explore its distributions.")
else:
    numerics = numeric_columns(data)
    chart_type = st.selectbox(
        "Chart",
        options=[
            "Histogram",
            "Kernel density estimate",
            "Box plot",
            "Violin plot",
            "Category counts",
            "Scatter / grouped points",
            "Grouped comparison",
        ],
        key="eda_chart_type",
    )
    try:
        if chart_type in {"Histogram", "Kernel density estimate"}:
            if not numerics:
                st.warning("No numeric variables are available for this chart.")
            else:
                column = st.selectbox("Variable", numerics, key="eda_distribution_column")
                figure = (
                    histogram_plot(data, column)
                    if chart_type == "Histogram"
                    else kde_plot(data, column)
                )
                st.plotly_chart(figure, use_container_width=True)
        elif chart_type in {"Box plot", "Violin plot", "Grouped comparison"}:
            if not numerics:
                st.warning("No numeric outcome variables are available for this chart.")
            else:
                left, right = st.columns(2)
                with left:
                    value_column = st.selectbox("Numeric outcome", numerics, key="eda_value_column")
                with right:
                    group_column = st.selectbox(
                        "Group variable",
                        list(data.columns),
                        key="eda_group_column",
                    )
                if chart_type == "Box plot":
                    figure = box_plot(data, value_column, group_column)
                elif chart_type == "Violin plot":
                    figure = violin_plot(data, value_column, group_column)
                else:
                    statistic = st.radio(
                        "Comparison summary",
                        options=["median", "mean"],
                        horizontal=True,
                        key="eda_group_statistic",
                    )
                    figure = grouped_comparison(
                        data,
                        group_column,
                        value_column,
                        statistic=statistic,
                    )
                st.plotly_chart(figure, use_container_width=True)
        elif chart_type == "Category counts":
            column = st.selectbox("Category variable", list(data.columns), key="eda_count_column")
            probability = st.checkbox(
                "Show percentages instead of counts",
                key="eda_count_percentages",
            )
            figure = bar_count_plot(data, column, probability=probability)
            st.plotly_chart(figure, use_container_width=True)
        else:
            if not numerics:
                st.warning("A numeric response variable is required for this chart.")
            else:
                left, right = st.columns(2)
                with left:
                    x_column = st.selectbox("X variable", list(data.columns), key="eda_x_column")
                with right:
                    y_column = st.selectbox(
                        "Y variable",
                        numerics,
                        index=min(1, len(numerics) - 1),
                        key="eda_y_column",
                    )
                trend = st.checkbox(
                    "Add a linear trend line when the x variable supports it",
                    key="eda_trend_line",
                )
                figure = scatter_plot(data, x_column, y_column, trend_line=trend)
                st.plotly_chart(figure, use_container_width=True)
    except VisualizationInputError as exc:
        st.warning(str(exc))
