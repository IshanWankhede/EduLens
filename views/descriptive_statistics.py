"""Interactive descriptive summaries for numeric, ordinal, and categorical variables."""

from __future__ import annotations

import streamlit as st

from src.descriptive_stats import (
    DescriptiveStatsError,
    frequency_table,
    group_summary,
    summary_table,
)
from views._analysis import analysis_settings, numeric_columns, selected_data

st.title("Descriptive Statistics")
st.markdown(
    '<p class="el-caption">Select variables to calculate descriptive summaries from the '
    "currently selected, cleaned dataset. Ordinal-coded variables are labeled accordingly.</p>",
    unsafe_allow_html=True,
)
data = selected_data()
if data is None:
    st.info("Select or upload a dataset from the sidebar to calculate descriptive statistics.")
else:
    numerics = numeric_columns(data)
    if not numerics:
        st.warning(
            "The selected dataset has no numeric variables available for summary statistics."
        )
    else:
        summary_tab, frequency_tab, grouped_tab = st.tabs(
            ["Numeric summary", "Frequency table", "Group comparison"]
        )
        with summary_tab:
            columns = st.multiselect(
                "Numeric and ordinal variables",
                options=numerics,
                default=numerics[: min(3, len(numerics))],
                key="desc_summary_columns",
            )
            if columns:
                try:
                    result = summary_table(data, columns)
                    st.dataframe(result.table, use_container_width=True)
                    st.caption(
                        f"Measurement roles: {', '.join(f'{name}: {role}' for name, role in result.measurement_levels.items())}. "
                        f"Quartiles: {result.quartile_method}; skewness: {result.skewness_method}."
                    )
                except DescriptiveStatsError as exc:
                    st.warning(str(exc))
            else:
                st.info("Select at least one numeric or ordinal variable.")
        with frequency_tab:
            column = st.selectbox(
                "Variable for frequency table",
                options=list(data.columns),
                key="desc_frequency_column",
            )
            try:
                result = frequency_table(data, column)
                st.dataframe(result.table, use_container_width=True, hide_index=True)
                st.caption(
                    f"Valid observations: {result.valid_count:,}; missing: {result.missing_count:,}; "
                    f"measurement role: {result.measurement_level}. Percentages use valid observations."
                )
            except DescriptiveStatsError as exc:
                st.warning(str(exc))
        with grouped_tab:
            grouping_candidates = [column for column in data.columns if column not in numerics]
            grouping_candidates += [
                column for column in numerics if column not in grouping_candidates
            ]
            if len(numerics) < 1 or not grouping_candidates:
                st.info("A numeric outcome and a grouping variable are required.")
            else:
                left, right = st.columns(2)
                with left:
                    value_column = st.selectbox(
                        "Numeric outcome",
                        options=numerics,
                        key="desc_group_value",
                    )
                with right:
                    group_column = st.selectbox(
                        "Group variable",
                        options=grouping_candidates,
                        key="desc_group_column",
                    )
                confidence = analysis_settings()["confidence_level"]
                try:
                    result = group_summary(
                        data,
                        value_column,
                        group_column,
                        confidence_level=confidence,
                    )
                    st.dataframe(result.table, use_container_width=True, hide_index=True)
                    st.caption(
                        f"Each group mean interval uses confidence level {result.confidence_level:.0%}. "
                        f"Ordinal outcome role: {result.measurement_level}; quartiles: "
                        f"{result.quartile_method}; skewness: {result.skewness_method}."
                    )
                    if result.missing_group_rows:
                        st.info(
                            f"{result.missing_group_rows:,} rows with missing group labels were excluded."
                        )
                except DescriptiveStatsError as exc:
                    st.warning(str(exc))
