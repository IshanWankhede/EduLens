"""Factor-only and explicitly labeled Model B OLS regression with diagnostics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.regression import RegressionError
from src.ui.components import assumptions_panel, metric_card
from src.visualization import VisualizationInputError, qq_plot, scatter_plot
from views._analysis import (
    analysis_settings,
    cached_ols,
    numeric_columns,
    selected_bundle,
)

st.title("Regression")
st.markdown(
    '<p class="el-caption">Estimate associations conditional on included predictors. '
    "Regression coefficients do not establish causation.</p>",
    unsafe_allow_html=True,
)
bundle = selected_bundle()
if bundle is None:
    st.info("Select or upload a dataset from the sidebar to fit an OLS model.")
else:
    data = bundle.clean
    numeric = numeric_columns(data)
    if len(numeric) < 2:
        st.warning(
            "Regression needs a numeric outcome and at least one numeric or coded predictor."
        )
    else:
        target = st.selectbox(
            "Outcome",
            numeric,
            index=numeric.index("G3") if "G3" in numeric else 0,
            key="regression_target",
        )
        model_b = False
        if target == "G3":
            grade_model = st.radio(
                "Grade model",
                options=["Model A (factor-only)", "Model B (includes G1 and/or G2)"],
                horizontal=True,
                key="regression_grade_model",
                help="Model A excludes G1 and G2. Model B is available only when explicitly selected.",
            )
            model_b = grade_model.startswith("Model B")
        predictor_options = [
            column
            for column in data.columns
            if column != target and column not in {"G3", "G3_zero_flag"}
        ]
        if target == "G3" and not model_b:
            predictor_options = [
                column for column in predictor_options if column not in {"G1", "G2"}
            ]
        defaults = [
            column
            for column in ("age", "studytime", "failures", "absences", "famsup")
            if column in predictor_options
        ]
        predictors = st.multiselect(
            "Predictors",
            options=predictor_options,
            default=defaults,
            key="regression_predictors",
            help=(
                "G3 is never available as a predictor. G1/G2 are excluded for Model A; "
                "they are available for G3 only in explicitly selected Model B."
            ),
        )
        options = st.columns(2)
        with options[0]:
            ordinal_mode = st.selectbox(
                "Ordinal encoding",
                options=["numeric", "categorical"],
                key="regression_ordinal_mode",
                help="Numeric treats codes as one-step increments; categorical uses reference-coded dummies.",
            )
        with options[1]:
            robust_se = st.checkbox(
                "Use HC3 robust standard errors",
                key="regression_hc3",
            )
        if not predictors:
            st.info("Select at least one predictor.")
        else:
            settings = analysis_settings()
            settings_key = (
                settings["confidence_level"],
                settings["alpha"],
                settings["thresholds"],
                bool(robust_se),
                ordinal_mode,
            )
            try:
                result = cached_ols(
                    f"{bundle.metadata.source}:{bundle.metadata.file_name}",
                    data,
                    target,
                    tuple(predictors),
                    settings["confidence_level"],
                    bool(robust_se),
                    ordinal_mode,
                    model_b,
                    settings_key,
                )
                if result.model_label.startswith("Model A"):
                    st.info("Model A is factor-only and excludes G1, G2, and G3 as predictors.")
                elif result.model_label.startswith("Model B"):
                    st.warning(
                        "Model B explicitly includes prior grades (G1/G2); interpret separately "
                        "from the factor-only Model A."
                    )
                st.subheader(result.model_label)
                metric_columns = st.columns(5)
                for column, label, value in zip(
                    metric_columns,
                    ("R²", "Adjusted R²", "F-statistic", "F-test p-value", "Residual SE"),
                    (
                        result.r_squared,
                        result.adjusted_r_squared,
                        result.f_statistic,
                        result.f_p_value,
                        result.residual_standard_error,
                    ),
                    strict=True,
                ):
                    with column:
                        metric_card(label, f"{value:.5g}")
                st.caption(
                    f"Complete-case sample size: {result.sample_size:,}; residual degrees of "
                    f"freedom: {result.residual_degrees_of_freedom:g}; inference uses "
                    f"{result.confidence_level:.0%} confidence and {result.robust_se} standard errors."
                )
                st.subheader("Coefficient estimates")
                st.dataframe(result.coefficients, width="stretch")
                st.caption(
                    "Intervals are confidence intervals at the selected level. Interpret "
                    "coefficients as associations holding the other included variables constant."
                )
                with st.expander("Encoding and reference levels"):
                    st.json(result.encoding)
                with st.expander("Coefficient interpretations"):
                    for term, text in result.interpretations.items():
                        st.markdown(f"**{term}** — {text}")

                diagnostics = result.diagnostics
                assumptions_panel(
                    diagnostics.assumptions,
                    alternatives=diagnostics.alternatives,
                )
                st.subheader("Diagnostic plots")
                fitted = diagnostics.fitted_values
                residual = diagnostics.residuals
                left, right = st.columns(2)
                try:
                    with left:
                        residual_figure = scatter_plot(
                            pd.DataFrame({"fitted": fitted, "residual": residual}),
                            "fitted",
                            "residual",
                            title="Residuals versus fitted values",
                            caption=(
                                "Look for an unstructured cloud around zero; curves or changing "
                                "spread may indicate model limitations."
                            ),
                        )
                        st.plotly_chart(residual_figure, width="stretch")
                    with right:
                        qq_figure = qq_plot(
                            residual,
                            title="Residual normal Q–Q plot",
                            caption=(
                                "Points near the reference line are more consistent with normal "
                                "residuals; departures suggest checking the model assumptions."
                            ),
                        )
                        st.pyplot(qq_figure, width="stretch")
                    plot_frame = pd.DataFrame(
                        {
                            "observation": range(len(fitted)),
                            "leverage": diagnostics.leverage.to_numpy(),
                            "Cook's distance": diagnostics.cooks_distance.to_numpy(),
                            "Standardized residual": diagnostics.standardized_residuals.to_numpy(),
                        }
                    )
                    st.dataframe(
                        plot_frame.sort_values("Cook's distance", ascending=False).head(20),
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(
                        "The table shows the largest Cook's distances alongside leverage and "
                        "standardized residuals; observations are flagged for review, not automatic removal."
                    )
                except VisualizationInputError as exc:
                    st.warning(f"Diagnostic plot unavailable: {exc}")
                st.subheader("Variance inflation factors")
                st.dataframe(diagnostics.vif_table, width="stretch", hide_index=True)
                st.caption(result.interpretation)
            except (RegressionError, ValueError, KeyError) as exc:
                st.warning(str(exc))
