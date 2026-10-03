"""Holdout and cross-validation metrics with a paired Model A/B comparison."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.prediction import (
    ModelComparison,
    PredictionError,
    PredictionRun,
)
from views._analysis import analysis_settings, cached_model_comparison, selected_bundle


def _render_model_metrics(label: str, run: PredictionRun) -> None:
    """Render a prediction run's holdout metrics, confusion matrix, and CV spread."""
    metrics = run.metrics
    cv = run.cross_validation
    st.subheader(label)
    columns = st.columns(5)
    values = (
        (
            "Accuracy",
            metrics.accuracy,
            f"Majority baseline: {metrics.majority_baseline_accuracy:.1%}",
        ),
        (
            "Macro-F1",
            metrics.macro_f1,
            f"Majority baseline: {metrics.majority_baseline_macro_f1:.4f}",
        ),
        ("CV accuracy", cv.accuracy_mean, f"Fold SD: {cv.accuracy_sd:.4f}"),
        ("CV macro-F1", cv.macro_f1_mean, f"Fold SD: {cv.macro_f1_sd:.4f}"),
        ("Test observations", run.test_size, f"Training observations: {run.train_size:,}"),
    )
    for column, (metric_name, value, note) in zip(columns, values, strict=True):
        with column:
            display = f"{value:,}" if metric_name == "Test observations" else f"{value:.4f}"
            st.metric(metric_name, display, help=note)
            st.caption(note)

    metrics_tab, cv_tab, model_tab = st.tabs(
        [f"{label} metrics", f"{label} CV folds", f"{label} model details"]
    )
    with metrics_tab:
        st.markdown("**Per-class precision, recall, and F1**")
        st.dataframe(metrics.per_class, width="stretch")
        st.markdown("**Confusion matrix (actual rows × predicted columns)**")
        st.dataframe(metrics.confusion_matrix, width="stretch")
        st.caption(
            "Confusion-matrix entries are holdout counts. Metrics and the majority-class baseline "
            "are computed from the same holdout labels."
        )
        st.caption(
            f"Majority baseline predicts {metrics.majority_class!r} for every test observation. "
            f"Accuracy: {metrics.majority_baseline_accuracy:.4f}; macro-F1: "
            f"{metrics.majority_baseline_macro_f1:.4f}. Log-loss: {metrics.log_loss:.5g}; "
            f"multiclass Brier score: {metrics.multiclass_brier_score:.5g}."
        )
    with cv_tab:
        fold_data = pd.DataFrame(
            {
                "Fold": range(1, cv.folds + 1),
                "Accuracy": cv.fold_accuracy,
                "Macro-F1": cv.fold_macro_f1,
            }
        )
        st.dataframe(fold_data, width="stretch", hide_index=True)
        st.line_chart(fold_data.set_index("Fold"), y_label="Fold score")
        st.caption(
            f"Each fold refits preprocessing inside its training partition. Means and sample SDs "
            f"are shown across {cv.folds} stratified folds."
        )
    with model_tab:
        st.write(f"Encoded feature count: {len(run.feature_names_after_encoding):,}")
        forbidden = {"G1", "G2", "G3"}
        if run.feature_set == "A":
            leakage = [
                name
                for name in run.feature_names_after_encoding
                if any(name.endswith(f"__{column}") or name == column for column in forbidden)
            ]
            if leakage:
                st.error("A prohibited grade column appears in the encoded Model A feature names.")
            else:
                st.success("Verified: Model A encoded features contain no G1, G2, or G3.")
        st.json(dict(run.regularization_settings))
        st.caption(run.mnlogit_status)
        if run.mnlogit_p_values is not None:
            st.dataframe(run.mnlogit_p_values, width="stretch")
        st.dataframe(
            run.coefficients.head(30),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Coefficient/odds-ratio rows describe model contribution, not causation. "
            "The table is limited to the first rows for display."
        )


st.title("Model Evaluation")
st.markdown(
    '<p class="el-caption">Compare holdout metrics with stratified cross-validation and a '
    "majority-class baseline. Model differences describe this evaluation and do not establish "
    "that one model is generally better.</p>",
    unsafe_allow_html=True,
)
bundle = selected_bundle()
if bundle is None:
    st.info("Select or upload a dataset from the sidebar to evaluate the classifiers.")
elif "G3" not in bundle.clean.columns:
    st.warning("Model evaluation requires G3 to define the configured performance classes.")
else:
    data = bundle.clean
    settings = analysis_settings()
    seed = int(st.session_state.get("el_seed", config.RANDOM_SEED))
    settings_key = (
        settings["confidence_level"],
        settings["alpha"],
        settings["thresholds"],
        seed,
        0.2,
        5,
    )
    dataset_key = f"{bundle.metadata.source}:{bundle.metadata.file_name}"
    try:
        comparison: ModelComparison = cached_model_comparison(
            dataset_key,
            data,
            seed,
            0.2,
            5,
            settings_key,
        )
        st.caption(
            "Model A is factor-only. Model B explicitly adds G1/G2. Both use the same configured "
            "stratified holdout and paired folds."
        )
        _render_model_metrics("Model A (factor-only; no G1/G2/G3 features)", comparison.model_a)
        _render_model_metrics("Model B (explicitly includes G1/G2)", comparison.model_b)

        st.subheader("Model A vs Model B")
        delta_table = pd.DataFrame(
            [
                {
                    "Comparison": "Holdout accuracy (B − A)",
                    "Difference": comparison.holdout_accuracy_difference_b_minus_a,
                },
                {
                    "Comparison": "Holdout macro-F1 (B − A)",
                    "Difference": comparison.holdout_macro_f1_difference_b_minus_a,
                },
                {
                    "Comparison": "CV accuracy difference: mean",
                    "Difference": comparison.fold_accuracy_difference_mean,
                },
                {
                    "Comparison": "CV accuracy difference: fold SD",
                    "Difference": comparison.fold_accuracy_difference_sd,
                },
                {
                    "Comparison": "CV macro-F1 difference: mean",
                    "Difference": comparison.fold_macro_f1_difference_mean,
                },
                {
                    "Comparison": "CV macro-F1 difference: fold SD",
                    "Difference": comparison.fold_macro_f1_difference_sd,
                },
            ]
        )
        st.dataframe(delta_table, width="stretch", hide_index=True)
        fold_comparison = pd.DataFrame(
            {
                "Fold": range(1, comparison.model_a.cross_validation.folds + 1),
                "Accuracy difference (B − A)": (
                    pd.Series(comparison.model_b.cross_validation.fold_accuracy)
                    - pd.Series(comparison.model_a.cross_validation.fold_accuracy)
                ),
                "Macro-F1 difference (B − A)": (
                    pd.Series(comparison.model_b.cross_validation.fold_macro_f1)
                    - pd.Series(comparison.model_a.cross_validation.fold_macro_f1)
                ),
            }
        )
        st.line_chart(fold_comparison.set_index("Fold"), y_label="Score difference")
        st.caption(
            "Fold-by-fold differences use identical validation folds. Positive and negative "
            "values report observed differences only; they do not imply general superiority."
        )
        st.info(comparison.interpretation)
    except (PredictionError, ValueError) as exc:
        st.warning(str(exc))
