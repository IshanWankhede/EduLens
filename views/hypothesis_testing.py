"""Interactive classical hypothesis tests with explicit evidence and assumption reporting."""

from __future__ import annotations

import streamlit as st

from src.hypothesis_tests import (
    HypothesisTestError,
    chi_square_independence,
    create_binary_groups,
    independent_t_test,
    one_way_anova,
)
from src.ui.components import assumptions_panel, metric_card
from src.visualization import VisualizationInputError, grouped_comparison
from views._analysis import (
    analysis_settings,
    categorical_columns,
    numeric_columns,
    selected_data,
)


def _show_test_flow(
    *,
    test: str,
    statistic: str,
    p_value: float,
    decision: str,
    interpretation: str,
) -> None:
    """Show test output in the required TEST → STATISTIC → P-VALUE → DECISION → INTERPRETATION order."""
    fields = (
        ("TEST", test),
        ("STATISTIC", statistic),
        ("P-VALUE", f"{p_value:.6g}"),
        (
            "DECISION",
            (
                "Evidence against H0"
                if decision == "Sufficient evidence against H0"
                else "Insufficient evidence"
            ),
        ),
        ("INTERPRETATION", interpretation),
    )
    columns = st.columns(5)
    for column, (label, value) in zip(columns, fields, strict=True):
        with column:
            metric_card(label, value)


st.title("Hypothesis Testing")
st.markdown(
    '<p class="el-caption">Tests summarize evidence under stated assumptions. A result does not '
    "prove a claim, establish causation, or imply that a null hypothesis is true.</p>",
    unsafe_allow_html=True,
)
data = selected_data()
if data is None:
    st.info("Select or upload a dataset from the sidebar to run a hypothesis test.")
else:
    settings = analysis_settings()
    alpha = settings["alpha"]
    confidence = settings["confidence_level"]
    test_kind = st.selectbox(
        "Test",
        options=[
            "Independent t-test (Welch)",
            "One-way ANOVA",
            "Chi-square independence",
        ],
        key="hypothesis_test_kind",
    )
    numerics = numeric_columns(data)
    categories = categorical_columns(data)
    try:
        if test_kind == "Independent t-test (Welch)":
            if not numerics:
                st.warning("A numeric outcome variable is required for an independent t-test.")
            else:
                value_column = st.selectbox(
                    "Numeric outcome",
                    numerics,
                    key="hypothesis_t_value",
                )
                grouping_mode = st.radio(
                    "How should observations be grouped?",
                    options=["Use existing categories", "Split a numeric / ordinal variable"],
                    horizontal=True,
                    key="hypothesis_t_grouping_mode",
                )
                if grouping_mode == "Use existing categories":
                    if not categories:
                        st.warning("No categorical or ordinal grouping variables are available.")
                    else:
                        group_column = st.selectbox(
                            "Group variable",
                            categories,
                            key="hypothesis_t_group_column",
                        )
                        levels = data[group_column].dropna().drop_duplicates().tolist()
                        if len(levels) < 2:
                            st.warning(
                                f"'{group_column}' has fewer than two observed groups for comparison."
                            )
                        else:
                            selected_levels = st.multiselect(
                                "Choose exactly two groups",
                                levels,
                                default=levels[:2],
                                key="hypothesis_t_group_levels",
                            )
                            if len(selected_levels) != 2:
                                st.info("Select exactly two groups.")
                            else:
                                first, second = selected_levels
                                result = independent_t_test(
                                    data.loc[data[group_column].eq(first), value_column],
                                    data.loc[data[group_column].eq(second), value_column],
                                    alpha=alpha,
                                    confidence_level=confidence,
                                )
                                _show_test_flow(
                                    test=result.test,
                                    statistic=(
                                        f"t = {result.statistic:.5g}; df = "
                                        f"{result.degrees_of_freedom:.5g}"
                                    ),
                                    p_value=result.p_value,
                                    decision=result.decision,
                                    interpretation=result.interpretation,
                                )
                                st.write(f"H0: {result.null_hypothesis}")
                                st.write(f"H1: {result.alternative_hypothesis}")
                                st.caption(
                                    f"Group sizes: {result.group_sizes[0]:,} and "
                                    f"{result.group_sizes[1]:,}; mean difference interval at "
                                    f"{result.confidence_level:.0%} confidence: "
                                    f"[{result.confidence_interval[0]:.5g}, "
                                    f"{result.confidence_interval[1]:.5g}]. "
                                    f"Cohen's d = {result.cohens_d:.5g}; Hedges' g = "
                                    f"{result.hedges_g:.5g}."
                                )
                                assumptions_panel(
                                    result.assumptions,
                                    alternatives=(
                                        [result.suggested_alternative]
                                        if result.suggested_alternative
                                        else []
                                    ),
                                )
                                try:
                                    figure = grouped_comparison(
                                        data.loc[data[group_column].isin(selected_levels)],
                                        group_column,
                                        value_column,
                                        statistic="mean",
                                    )
                                    st.plotly_chart(figure, use_container_width=True)
                                except VisualizationInputError as exc:
                                    st.info(f"Group comparison chart unavailable: {exc}")
                else:
                    split_column = st.selectbox(
                        "Numeric or ordinal split variable",
                        numerics,
                        key="hypothesis_t_split_column",
                    )
                    cutoff = st.number_input(
                        "Cutoff",
                        value=0.0,
                        key="hypothesis_t_cutoff",
                    )
                    operator = st.selectbox(
                        "Upper-group rule",
                        options=[">=", ">"],
                        key="hypothesis_t_split_operator",
                    )
                    groups = create_binary_groups(
                        data,
                        split_column,
                        cutoff,
                        operator=operator,
                    )
                    st.caption(
                        f"Grouping rule: {split_column} {groups.operator} {groups.cutoff:g} "
                        f"versus {split_column} below that rule. Group counts: "
                        f"{groups.group_sizes[groups.lower_label]:,} and "
                        f"{groups.group_sizes[groups.upper_label]:,}."
                    )
                    grouped_data = data.assign(__ui_group=groups.groups)
                    first_values = grouped_data.loc[
                        grouped_data["__ui_group"].eq(groups.lower_label), value_column
                    ]
                    second_values = grouped_data.loc[
                        grouped_data["__ui_group"].eq(groups.upper_label), value_column
                    ]
                    result = independent_t_test(
                        first_values,
                        second_values,
                        alpha=alpha,
                        confidence_level=confidence,
                    )
                    _show_test_flow(
                        test=result.test,
                        statistic=(
                            f"t = {result.statistic:.5g}; df = {result.degrees_of_freedom:.5g}"
                        ),
                        p_value=result.p_value,
                        decision=result.decision,
                        interpretation=result.interpretation,
                    )
                    assumptions_panel(
                        result.assumptions,
                        alternatives=(
                            [result.suggested_alternative] if result.suggested_alternative else []
                        ),
                    )
        elif test_kind == "One-way ANOVA":
            if not numerics or not categories:
                st.warning("ANOVA requires a numeric outcome and a categorical or ordinal group.")
            else:
                left, right = st.columns(2)
                with left:
                    value_column = st.selectbox(
                        "Numeric outcome",
                        numerics,
                        key="hypothesis_anova_value",
                    )
                with right:
                    group_column = st.selectbox(
                        "Group variable",
                        categories,
                        key="hypothesis_anova_group",
                    )
                levels = data[group_column].dropna().drop_duplicates().tolist()
                selected_levels = st.multiselect(
                    "Groups to include (select at least three)",
                    levels,
                    default=levels,
                    key="hypothesis_anova_levels",
                )
                if len(selected_levels) < 3:
                    st.info("Select at least three groups for one-way ANOVA.")
                else:
                    test_data = data.loc[data[group_column].isin(selected_levels)]
                    result = one_way_anova(
                        test_data,
                        value_column,
                        group_column,
                        alpha=alpha,
                        confidence_level=confidence,
                    )
                    degrees = result.degrees_of_freedom
                    _show_test_flow(
                        test=result.test,
                        statistic=f"F({degrees[0]:g}, {degrees[1]:g}) = {result.statistic:.5g}",
                        p_value=result.p_value,
                        decision=result.decision,
                        interpretation=result.interpretation,
                    )
                    st.subheader("Group means and confidence intervals")
                    st.dataframe(result.group_means, use_container_width=True, hide_index=True)
                    st.caption(
                        f"Eta-squared = {result.eta_squared:.5g}. Welch ANOVA alternative: "
                        f"F = {result.welch_statistic:.5g}, p = {result.welch_p_value:.6g}. "
                        f"Kruskal–Wallis alternative: H = {result.kruskal_statistic:.5g}, "
                        f"p = {result.kruskal_p_value:.6g}."
                    )
                    assumptions_panel(
                        result.assumptions,
                        alternatives=[
                            "Welch ANOVA for unequal group variances",
                            "Kruskal–Wallis as an omnibus rank-based alternative",
                        ],
                    )
                    st.info(result.posthoc_note)
                    if result.tukey_comparisons is not None:
                        st.subheader("Tukey HSD pairwise comparisons")
                        st.dataframe(
                            result.tukey_comparisons,
                            use_container_width=True,
                            hide_index=True,
                        )
                    try:
                        figure = grouped_comparison(
                            test_data,
                            group_column,
                            value_column,
                            statistic="mean",
                        )
                        st.plotly_chart(figure, use_container_width=True)
                    except VisualizationInputError as exc:
                        st.info(f"Group comparison chart unavailable: {exc}")
        else:
            if len(categories) < 2:
                st.warning("At least two categorical or ordinal variables are required.")
            else:
                left, right = st.columns(2)
                with left:
                    row_column = st.selectbox(
                        "Row variable",
                        categories,
                        key="hypothesis_chi_row",
                    )
                with right:
                    column_options = [column for column in categories if column != row_column]
                    column = st.selectbox(
                        "Column variable",
                        column_options,
                        key="hypothesis_chi_column",
                    )
                result = chi_square_independence(
                    data,
                    row_column,
                    column,
                    alpha=alpha,
                )
                _show_test_flow(
                    test=result.test,
                    statistic=f"χ²({result.degrees_of_freedom}) = {result.statistic:.5g}",
                    p_value=result.p_value,
                    decision=result.decision,
                    interpretation=result.interpretation,
                )
                st.caption(
                    f"Cramer's V = {result.cramers_v:.5g}. Yates' correction: "
                    f"{'applied for 2×2 tables' if result.yates_correction else 'disabled'}."
                )
                assumptions_panel((result.expected_count_check,))
                for label, table in (
                    ("Observed counts", result.observed),
                    ("Expected counts", result.expected),
                    ("Standardized residuals", result.standardized_residuals),
                ):
                    with st.expander(label):
                        st.dataframe(table, use_container_width=True)
                if result.fisher_exact_p_value is not None:
                    st.info(
                        f"Sparse 2×2 table: Fisher's exact alternative p-value = "
                        f"{result.fisher_exact_p_value:.6g}."
                    )
    except (HypothesisTestError, ValueError) as exc:
        st.warning(str(exc))
