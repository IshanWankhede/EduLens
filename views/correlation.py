"""Pearson and Spearman association matrices and ranked relationships."""

from __future__ import annotations

import streamlit as st

from src.correlation import (
    CorrelationAnalysisError,
    correlation_matrices,
    ranked_associations,
)
from src.visualization import VisualizationInputError, correlation_heatmap
from views._analysis import analysis_settings, numeric_columns, selected_data

st.title("Correlation")
st.markdown(
    '<p class="el-caption">Compare linear (Pearson) and rank-based (Spearman) associations. '
    "Association is not evidence of causation; p-values and confidence intervals are "
    "conditional on the assumptions of these methods.</p>",
    unsafe_allow_html=True,
)
data = selected_data()
if data is None:
    st.info("Select or upload a dataset from the sidebar to calculate correlations.")
else:
    columns = numeric_columns(data)
    if len(columns) < 2:
        st.warning("At least two numeric or ordinal variables are required.")
    else:
        selected = st.multiselect(
            "Variables",
            options=columns,
            default=columns[: min(6, len(columns))],
            key="correlation_columns",
        )
        if len(selected) < 2:
            st.info("Select at least two variables.")
        else:
            matrix_tab, ranked_tab = st.tabs(["Correlation matrices", "Ranked associations"])
            with matrix_tab:
                try:
                    matrices = correlation_matrices(
                        data,
                        selected,
                        confidence_level=analysis_settings()["confidence_level"],
                    )
                    method = st.radio(
                        "Matrix to display",
                        options=["Pearson", "Spearman"],
                        horizontal=True,
                        key="correlation_matrix_method",
                    )
                    result = matrices.pearson if method == "Pearson" else matrices.spearman
                    figure = correlation_heatmap(
                        result.coefficients,
                        title=f"{method} correlation matrix",
                        caption=(
                            "Cells show pairwise coefficients; use the labeled colorbar for magnitude. "
                            "The coefficient, p-value, and pair count tables are also shown below."
                        ),
                    )
                    st.plotly_chart(figure, use_container_width=True)
                    st.subheader("Coefficients")
                    st.dataframe(result.coefficients, use_container_width=True)
                    st.subheader("P-values")
                    st.dataframe(result.p_values, use_container_width=True)
                    st.subheader("Pairwise sample sizes")
                    st.dataframe(result.sample_sizes, use_container_width=True)
                    if result.ci_lower is not None and result.ci_upper is not None:
                        st.caption(
                            "Pearson confidence intervals use Fisher's z transformation at "
                            f"{result.confidence_level:.0%} confidence."
                        )
                        with st.expander("Pearson Fisher-z interval bounds"):
                            st.markdown("**Lower bounds**")
                            st.dataframe(result.ci_lower, use_container_width=True)
                            st.markdown("**Upper bounds**")
                            st.dataframe(result.ci_upper, use_container_width=True)
                    recommended = [
                        column for column, ordinal in result.recommended_spearman.items() if ordinal
                    ]
                    if recommended:
                        st.info(
                            "Spearman is recommended for ordinal variables: "
                            + ", ".join(recommended)
                            + "."
                        )
                except (CorrelationAnalysisError, VisualizationInputError) as exc:
                    st.warning(str(exc))
            with ranked_tab:
                target = st.selectbox(
                    "Target variable",
                    selected,
                    key="correlation_target",
                )
                try:
                    result = ranked_associations(data, target, columns=selected, method="auto")
                    st.dataframe(result.table, use_container_width=True, hide_index=True)
                    st.caption(
                        f"Automatic method recommends Spearman whenever either variable is ordinal. "
                        f"Strength labels: {result.strength_thresholds}."
                    )
                except CorrelationAnalysisError as exc:
                    st.warning(str(exc))
