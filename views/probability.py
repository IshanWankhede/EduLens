"""Conditional probability, Bayes walkthrough, and distribution diagnostics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.probability import (
    EventCondition,
    ProbabilityAnalysisError,
    bayes,
    conditional_probability,
    fit_distributions,
)
from src.ui.components import metric_card, probability_card
from src.visualization import VisualizationInputError, histogram_plot, qq_plot
from views._analysis import analysis_settings, numeric_columns, selected_data


def _event_input(data: pd.DataFrame, label: str, key: str) -> EventCondition | None:
    """Render a condition editor and return its current event definition."""
    st.markdown(f"**Event {label}**")
    column = st.selectbox(
        f"Column for event {label}",
        list(data.columns),
        key=f"{key}_column",
    )
    values = data[column].dropna()
    if values.empty:
        st.warning(f"Column '{column}' has no values to define an event.")
        return None
    if pd.api.types.is_numeric_dtype(data[column].dtype) and not pd.api.types.is_bool_dtype(
        data[column].dtype
    ):
        operator = st.selectbox(
            f"Condition for event {label}",
            options=[">=", ">", "<=", "<", "==", "!="],
            key=f"{key}_operator",
        )
        threshold = st.number_input(
            f"Threshold for event {label}",
            value=0.0,
            key=f"{key}_threshold",
        )
        return EventCondition(column=column, operator=operator, value=float(threshold))
    categories = values.drop_duplicates().tolist()
    operator = st.selectbox(
        f"Condition for event {label}",
        options=["==", "!=", "in", "not in"],
        key=f"{key}_operator",
    )
    if operator in {"in", "not in"}:
        selected = st.multiselect(
            f"Category value(s) for event {label}",
            options=categories,
            default=categories[:1],
            key=f"{key}_categories",
        )
        if not selected:
            st.warning(f"Select at least one category for event {label}.")
            return None
        return EventCondition(column=column, operator=operator, value=selected)
    category = st.selectbox(
        f"Category for event {label}",
        categories,
        key=f"{key}_category",
    )
    return EventCondition(column=column, operator=operator, value=category)


st.title("Probability")
st.markdown(
    '<p class="el-caption">Calculate observed conditional probabilities, walk through Bayes’ '
    "theorem, and inspect distribution fits. These are descriptions of the selected sample.</p>",
    unsafe_allow_html=True,
)
data = selected_data()
if data is None:
    st.info("Select or upload a dataset from the sidebar to calculate probabilities.")
else:
    calculator_tab, bayes_tab, distributions_tab = st.tabs(
        ["Conditional probability", "Bayes walkthrough", "Distributions"]
    )
    with calculator_tab:
        event_columns = st.columns(2)
        with event_columns[0]:
            event_a = _event_input(data, "A", "probability_a")
        with event_columns[1]:
            event_b = _event_input(data, "B", "probability_b")
        if event_a is not None and event_b is not None:
            try:
                result = conditional_probability(
                    data,
                    event_a,
                    event_b,
                    confidence_level=analysis_settings()["confidence_level"],
                )
                probability_card(
                    "P(A)",
                    result.p_a * 100,
                    f"n(A) / n = {result.n_a:,} / {result.n:,}",
                )
                probability_card(
                    "P(B)",
                    result.p_b * 100,
                    f"n(B) / n = {result.n_b:,} / {result.n:,}",
                )
                if result.p_a_given_b is None:
                    metric_card("P(A|B)", "Undefined", "n(A ∩ B) / n(B) has a zero denominator.")
                else:
                    probability_card(
                        "P(A|B)",
                        result.p_a_given_b * 100,
                        f"n(A ∩ B) / n(B) = {result.n_a_and_b:,} / {result.n_b:,}",
                    )
                    if result.wilson_ci is not None:
                        st.caption(
                            f"Wilson interval for P(A|B), at "
                            f"{analysis_settings()['confidence_level']:.0%} confidence: "
                            f"[{result.wilson_ci[0]:.4f}, {result.wilson_ci[1]:.4f}]."
                        )
                metric_card(
                    "P(A ∩ B)",
                    f"{result.p_a_and_b:.1%}",
                    f"n(A ∩ B) / n = {result.n_a_and_b:,} / {result.n:,}",
                )
                if result.message:
                    st.warning(result.message)
                else:
                    st.info(
                        "Probabilities use the common complete-case cohort for the selected event columns."
                    )
            except (ProbabilityAnalysisError, ValueError) as exc:
                st.warning(str(exc))
    with bayes_tab:
        event_columns = st.columns(2)
        with event_columns[0]:
            event_a = _event_input(data, "A", "bayes_a")
        with event_columns[1]:
            event_b = _event_input(data, "B", "bayes_b")
        if event_a is not None and event_b is not None:
            try:
                result = bayes(data, event_a, event_b)
                posterior_values = (
                    result.prior_p_b,
                    result.likelihood_p_a_given_b,
                    result.evidence_p_a,
                    result.posterior_p_b_given_a,
                )
                formulas = (
                    f"Prior: n(B) / n = {result.n_b:,} / {result.n:,}",
                    f"Likelihood: n(A ∩ B) / n(B) = {result.n_a_and_b:,} / {result.n_b:,}",
                    (
                        f"Evidence: n(A) / n = {result.n_a:,} / {result.n:,}; "
                        f"total probability partitions B and not B"
                    ),
                    (f"Posterior: n(A ∩ B) / n(A) = {result.n_a_and_b:,} / " f"{result.n_a:,}"),
                )
                labels = ("Prior P(B)", "Likelihood P(A|B)", "Evidence P(A)", "Posterior P(B|A)")
                cards = st.columns(4)
                for column, label, value, formula in zip(
                    cards, labels, posterior_values, formulas, strict=True
                ):
                    with column:
                        if value is None:
                            metric_card(label, "Undefined", formula)
                        else:
                            probability_card(label, value * 100, formula)
                st.markdown(
                    "### Prior → likelihood → evidence → posterior",
                    help="The posterior is compared with the direct observed frequency.",
                )
                st.caption(
                    f"Direct check: n(A ∩ B) / n(A) = {result.n_a_and_b:,} / "
                    f"{result.n_a:,} = "
                    + (
                        "undefined"
                        if result.direct_p_b_given_a is None
                        else f"{result.direct_p_b_given_a:.4f}"
                    )
                    + "."
                )
                if result.message:
                    st.warning(result.message)
            except (ProbabilityAnalysisError, ValueError) as exc:
                st.warning(str(exc))
    with distributions_tab:
        numerics = numeric_columns(data)
        if not numerics:
            st.warning("No numeric variables are available for distribution analysis.")
        else:
            column = st.selectbox(
                "Numeric variable",
                numerics,
                key="distribution_variable",
            )
            categories = [
                name
                for name in data.columns
                if not pd.api.types.is_numeric_dtype(data[name].dtype)
                or pd.api.types.is_bool_dtype(data[name].dtype)
            ]
            binary_event = None
            if categories:
                binary_column = st.selectbox(
                    "Optional binary event source",
                    options=["No event fit", *categories],
                    key="distribution_binary_column",
                )
                if binary_column != "No event fit":
                    observed_categories = data[binary_column].dropna().drop_duplicates().tolist()
                    if observed_categories:
                        event_value = st.selectbox(
                            "Event value",
                            observed_categories,
                            key="distribution_binary_value",
                        )
                        binary_event = data[binary_column].eq(event_value)
            try:
                fit = fit_distributions(
                    data[column],
                    binary_event=binary_event,
                    variable_name=column,
                    alpha=analysis_settings()["alpha"],
                )
                st.subheader("Normal fit assessment")
                metric_columns = st.columns(3)
                with metric_columns[0]:
                    metric_card(
                        "Normal-fit verdict",
                        fit.normality.verdict.title(),
                        "Verdict is based on the configured alpha and reported evidence.",
                    )
                with metric_columns[1]:
                    metric_card(
                        "Estimated mean (μ̂)",
                        "Unavailable" if fit.normal is None else f"{fit.normal.mu_hat:.4g}",
                        "Sample estimate; not a population guarantee.",
                    )
                with metric_columns[2]:
                    metric_card(
                        "Estimated SD (σ̂)",
                        "Unavailable" if fit.normal is None else f"{fit.normal.sigma_hat:.4g}",
                        "Sample standard deviation (ddof=1).",
                    )
                for item in fit.normality.evidence:
                    st.write(item)
                if fit.normal is not None and fit.normal.bounded_integer_score_note:
                    st.info(fit.normal.bounded_integer_score_note)
                if fit.binomial is not None:
                    metric_card(
                        "Binomial event estimate",
                        f"{fit.binomial.p_hat:.1%}",
                        f"successes / trials = {fit.binomial.successes:,} / {fit.binomial.n:,}",
                    )
                chart_columns = st.columns(2)
                with chart_columns[0]:
                    st.plotly_chart(
                        histogram_plot(data, column),
                        use_container_width=True,
                    )
                with chart_columns[1]:
                    st.caption("Empirical CDF is calculated from observed values.")
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Value": fit.empirical_cdf.values,
                                "Cumulative probability": fit.empirical_cdf.cumulative_probabilities,
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                if fit.normality.qq_theoretical_quantiles:
                    st.pyplot(
                        qq_plot(
                            data[column].dropna(),
                            title=f"Normal Q–Q plot: {column}",
                            caption=(
                                "Points near the reference line are more consistent with a normal "
                                "distribution; systematic departures indicate differences."
                            ),
                        ),
                        use_container_width=True,
                    )
            except (ProbabilityAnalysisError, VisualizationInputError, ValueError) as exc:
                st.warning(str(exc))
