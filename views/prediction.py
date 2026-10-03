"""Profile-based Low/Medium/High probabilities from an explicitly selected model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.prediction import (
    CLASS_LABELS,
    MODEL_A_FEATURES,
    MODEL_B_FEATURES,
    PredictionError,
    predict_proba_for_profile,
)
from src.visualization import probability_bar_plot
from views._analysis import analysis_settings, cached_prediction_run, selected_bundle

DISCLAIMER = (
    "This is a statistical estimate based on the selected dataset and model. It should not be "
    "interpreted as a guaranteed prediction of an individual's academic outcome."
)


def _profile_input(data: pd.DataFrame, column: str, key: str) -> object:
    """Render a profile control using only values and roles present in the selected data."""
    observed = data[column].dropna()
    if observed.empty:
        st.caption(f"{column}: no observed values; the model's fitted imputer will supply a value.")
        return np.nan
    default = observed.iloc[0]
    if column in config.BINARY_COLUMNS or column in config.NOMINAL_COLUMNS:
        choices = observed.drop_duplicates().tolist()
        return st.selectbox(
            column,
            choices,
            index=choices.index(default),
            key=key,
        )
    number = float(pd.to_numeric(observed, errors="coerce").dropna().iloc[0])
    return st.number_input(
        column,
        value=number,
        key=key,
        help="Initial value comes from the first observed row; adjust it to explore a profile.",
    )


st.title("Prediction")
st.markdown(
    f'<p class="el-caption">{DISCLAIMER}</p>',
    unsafe_allow_html=True,
)
bundle = selected_bundle()
if bundle is None:
    st.info("Select or upload a dataset from the sidebar to use profile prediction.")
elif "G3" not in bundle.clean.columns:
    st.warning("Prediction requires G3 in the selected dataset to define performance categories.")
else:
    data = bundle.clean
    selection = st.radio(
        "Model",
        options=["Model A (factor-only)", "Model B (includes G1 and G2)"],
        horizontal=True,
        key="prediction_model_choice",
        help="Model B is an explicit alternative that uses prior grades; Model A never uses G1/G2.",
    )
    feature_set = "B" if selection.startswith("Model B") else "A"
    features = MODEL_B_FEATURES if feature_set == "B" else MODEL_A_FEATURES
    if feature_set == "A" and set(features).intersection({"G1", "G2", "G3"}):
        st.error("Model A feature configuration contains a prohibited grade column.")
    else:
        missing_columns = [column for column in features if column not in data.columns]
        if missing_columns:
            st.warning(
                "The selected dataset cannot train this configured feature set; missing columns: "
                + ", ".join(missing_columns)
                + "."
            )
        else:
            settings = analysis_settings()
            seed = int(st.session_state.get("el_seed", config.RANDOM_SEED))
            settings_key = (
                settings["confidence_level"],
                settings["alpha"],
                settings["thresholds"],
                seed,
                feature_set,
            )
            dataset_key = f"{bundle.metadata.source}:{bundle.metadata.file_name}"
            try:
                model = cached_prediction_run(
                    dataset_key,
                    data,
                    feature_set,
                    seed,
                    config.DEFAULT_TEST_SIZE,
                    config.DEFAULT_CV_FOLDS,
                    settings_key,
                )
                st.caption(
                    "Model B is selected and includes G1/G2."
                    if feature_set == "B"
                    else "Model A is factor-only; its available profile fields exclude G1, G2, and G3."
                )
                selected_features = st.multiselect(
                    "Profile fields to specify",
                    options=list(features),
                    default=[
                        column
                        for column in ("age", "studytime", "failures", "absences", "famsup")
                        if column in features
                    ],
                    key=f"prediction_fields_{feature_set}",
                    help=(
                        "Unselected fields are omitted and filled by imputers learned from the "
                        "training data. Model A's choices do not contain G1 or G2."
                    ),
                )
                profile: dict[str, object] = {}
                with st.form(f"profile_form_{feature_set}"):
                    input_columns = st.columns(3)
                    for index, column in enumerate(selected_features):
                        with input_columns[index % len(input_columns)]:
                            profile[column] = _profile_input(
                                data,
                                column,
                                f"prediction_{feature_set}_{column}",
                            )
                    submitted = st.form_submit_button("Estimate class probabilities")
                if submitted:
                    prediction = predict_proba_for_profile(profile, model)
                    probabilities = prediction.probabilities
                    figure = probability_bar_plot(
                        {label: probabilities[label] for label in CLASS_LABELS},
                        title="Estimated performance-category probabilities",
                        caption=(
                            "Each bar is the model-estimated class probability on a fixed "
                            "0–100% scale. The three probabilities sum to 100%; they are "
                            "model outputs, not guarantees."
                        ),
                    )
                    st.subheader("Estimated Low / Medium / High probabilities")
                    st.plotly_chart(figure, width="stretch")
                    result_columns = st.columns(3)
                    for column, label in zip(result_columns, CLASS_LABELS, strict=True):
                        with column:
                            st.metric(label, f"{probabilities[label]:.1%}")
                    st.info(prediction.message)
            except (PredictionError, ValueError) as exc:
                st.warning(str(exc))
