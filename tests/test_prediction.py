from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src import config
from src.data_loader import load_uci_dataset
from src.prediction import (
    CLASS_LABELS,
    MODEL_A_FEATURES,
    MODEL_B_FEATURES,
    InsufficientClassDataError,
    PredictionError,
    ProfilePredictionError,
    build_pipeline,
    compare_models,
    evaluate_model,
    performance_categories,
    predict_proba_for_profile,
    save_pipeline,
)


def _synthetic_course_data(rows_per_class: int = 30) -> pd.DataFrame:
    """Build a varied categorical/numeric fixture with all configured predictor columns."""
    rng = np.random.default_rng(42)
    size = rows_per_class * len(CLASS_LABELS)
    categories = np.repeat([5, 11, 16], rows_per_class)
    data: dict[str, object] = {"G3": categories}
    for column in config.NUMERIC_COLUMNS:
        if column == "G3":
            continue
        data[column] = rng.normal(15, 3, size)
    for column in config.ORDINAL_COLUMNS:
        data[column] = rng.integers(1, 5, size)
    for column in config.NOMINAL_COLUMNS:
        data[column] = rng.choice(["a", "b", "c"], size)
    for column in config.BINARY_COLUMNS:
        data[column] = rng.choice(["no", "yes"], size)
    return pd.DataFrame(data)


def test_feature_sets_are_derived_from_roles_and_exclude_grade_leakage() -> None:
    assert MODEL_A_FEATURES == config.MODEL_A_FEATURES
    assert MODEL_B_FEATURES == config.MODEL_B_FEATURES
    assert not set(MODEL_A_FEATURES).intersection({"G1", "G2", "G3"})
    assert MODEL_B_FEATURES == MODEL_A_FEATURES + ("G1", "G2")


def test_fixed_grade_bands_match_config() -> None:
    grades = pd.Series([0, 9, 10, 13, 14, 20, np.nan])
    categories = performance_categories(grades)

    assert categories.iloc[:6].tolist() == ["Low", "Low", "Medium", "Medium", "High", "High"]
    assert pd.isna(categories.iloc[6])
    assert config.PERFORMANCE_BANDS["HIGH_LOWER_INCLUSIVE"] == 14


def test_pipeline_contains_preprocessing_and_multinomial_regularization() -> None:
    pipeline = build_pipeline("A")

    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["preprocessor", "classifier"]
    assert pipeline.named_steps["classifier"].get_params()["l1_ratio"] == 0
    assert pipeline.named_steps["classifier"].get_params()["C"] == 1.0
    assert pipeline.named_steps["classifier"].get_params()["solver"] == "lbfgs"


def test_evaluation_is_deterministic_encodes_safe_features_and_fits_only_train_data() -> None:
    data = _synthetic_course_data()
    first = evaluate_model(data, "A", test_size=0.25, cv_folds=3, seed=42)
    second = evaluate_model(data, "A", test_size=0.25, cv_folds=3, seed=42)

    assert first.train_indices == second.train_indices
    assert first.test_indices == second.test_indices
    assert first.metrics.accuracy == second.metrics.accuracy
    assert first.metrics.macro_f1 == second.metrics.macro_f1
    assert first.cross_validation == second.cross_validation
    assert first.train_size + first.test_size == len(data)
    assert first.class_counts == {"Low": 30, "Medium": 30, "High": 30}
    assert first.metrics.confusion_matrix.index.tolist() == list(CLASS_LABELS)
    assert first.metrics.per_class.index.tolist() == list(CLASS_LABELS)
    assert first.metrics.majority_baseline_accuracy >= 0
    assert first.metrics.majority_baseline_macro_f1 >= 0
    assert first.cross_validation.folds == 3
    assert "G1" not in " ".join(first.feature_names_after_encoding)
    assert "G2" not in " ".join(first.feature_names_after_encoding)
    assert "G3" not in " ".join(first.feature_names_after_encoding)
    assert all(
        "model contribution, not causation" == value
        for value in first.coefficients["interpretation_caveat"]
    )
    assert first.mnlogit_status
    if first.mnlogit_p_values is not None:
        assert first.mnlogit_p_values.columns.tolist() == ["Medium", "High"]
        assert first.mnlogit_p_values.index[0] == "const"

    features = data.loc[list(first.train_indices), MODEL_A_FEATURES]
    expected_age_mean = features["age"].mean()
    scaler = (
        first.pipeline.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["scaler"]
    )
    numeric_columns = list(
        first.pipeline.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["imputer"]
        .get_feature_names_out()
    )
    age_position = numeric_columns.index("age")
    assert scaler.mean_[age_position] == pytest.approx(expected_age_mean)
    assert scaler.mean_[age_position] != pytest.approx(data["age"].mean())


def test_partial_profile_uses_fitted_imputer_and_probabilities_sum_to_one() -> None:
    model = evaluate_model(_synthetic_course_data(), "A", cv_folds=3)
    result = predict_proba_for_profile({"studytime": 3, "school": "a"}, model)

    assert list(result.probabilities) == list(CLASS_LABELS)
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert all(0 <= value <= 1 for value in result.probabilities.values())
    assert "imputers fitted on the training data" in result.message
    with pytest.raises(ProfilePredictionError, match="G3 is the outcome"):
        predict_proba_for_profile({"G3": 19}, model)
    with pytest.raises(ProfilePredictionError, match="mapping"):
        predict_proba_for_profile([], model)  # type: ignore[arg-type]


def test_model_comparison_uses_identical_split_and_reports_paired_spread() -> None:
    result = compare_models(_synthetic_course_data(), cv_folds=3, seed=42)

    assert result.model_a.train_indices == result.model_b.train_indices
    assert result.model_a.test_indices == result.model_b.test_indices
    assert len(result.model_a.cross_validation.fold_accuracy) == 3
    assert result.fold_accuracy_difference_mean == pytest.approx(
        np.mean(
            np.asarray(result.model_b.cross_validation.fold_accuracy)
            - np.asarray(result.model_a.cross_validation.fold_accuracy)
        )
    )
    assert "do not establish that either model is generally better" in result.interpretation


def test_insufficient_class_counts_and_invalid_thresholds_raise_friendly_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _synthetic_course_data(rows_per_class=4)
    with pytest.raises(InsufficientClassDataError, match="at least 5"):
        evaluate_model(data, "A", cv_folds=5)
    monkeypatch.setitem(config.PERFORMANCE_BANDS, "HIGH_LOWER_INCLUSIVE", 15)
    with pytest.raises(PredictionError, match="contiguous"):
        performance_categories(pd.Series([10, 11, 12]))


def test_save_pipeline_is_explicit_and_rejects_paths() -> None:
    model = evaluate_model(_synthetic_course_data(), "A", cv_folds=3)
    with pytest.raises(PredictionError, match="simple file name"):
        save_pipeline(model, "..\\outside.joblib")
    destination = save_pipeline(model, "test_phase8_model.pkl")
    try:
        assert destination.parent.name == "models"
        assert destination.exists()
        restored = joblib.load(destination)
        assert isinstance(restored, Pipeline)
        assert restored.predict_proba(
            pd.DataFrame([_synthetic_course_data().iloc[0].drop(labels=["G3"]).to_dict()])[
                list(MODEL_A_FEATURES)
            ]
        ).shape == (1, len(CLASS_LABELS))
    finally:
        destination.unlink(missing_ok=True)


def test_real_portuguese_class_counts_are_the_verified_phase_two_counts() -> None:
    clean = load_uci_dataset("por").clean
    labels = performance_categories(clean["G3"])

    assert labels.value_counts().to_dict() == {"Medium": 355, "High": 194, "Low": 100}
