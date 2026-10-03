"""Leakage-safe multinomial classification and estimated class probabilities."""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

FeatureSet = Literal["A", "B"]
CLASS_LABELS = ("Low", "Medium", "High")
MODEL_A_FEATURES = config.MODEL_A_FEATURES
MODEL_B_FEATURES = config.MODEL_B_FEATURES
REGULARIZATION_SETTINGS = {
    "penalty": "L2",
    "l1_ratio": 0.0,
    "C": 1.0,
    "solver": "lbfgs",
    "class_weight": None,
    "max_iter": 1000,
    "multiclass": "multinomial softmax (three-class target)",
}


class PredictionError(ValueError):
    """Base exception for invalid prediction/evaluation inputs."""


class InsufficientClassDataError(PredictionError):
    """Raised when stratified splitting or cross-validation cannot preserve classes."""


class ProfilePredictionError(PredictionError):
    """Raised when profile values cannot be scored by a fitted prediction pipeline."""


@dataclass(frozen=True)
class ClassificationMetrics:
    """Holdout classification metrics, per-class scores, and baseline comparison."""

    accuracy: float
    per_class: pd.DataFrame
    macro_f1: float
    confusion_matrix: pd.DataFrame
    majority_class: str
    majority_baseline_accuracy: float
    majority_baseline_macro_f1: float
    log_loss: float
    multiclass_brier_score: float


@dataclass(frozen=True)
class CrossValidationMetrics:
    """Mean and sample SD of stratified-fold accuracy and macro-F1."""

    accuracy_mean: float
    accuracy_sd: float
    macro_f1_mean: float
    macro_f1_sd: float
    fold_accuracy: tuple[float, ...]
    fold_macro_f1: tuple[float, ...]
    folds: int


@dataclass(frozen=True)
class PredictionRun:
    """Fitted pipeline and reproducible holdout/CV evaluation."""

    feature_set: FeatureSet
    pipeline: Pipeline
    class_counts: dict[str, int]
    train_size: int
    test_size: int
    train_indices: tuple[object, ...]
    test_indices: tuple[object, ...]
    metrics: ClassificationMetrics
    cross_validation: CrossValidationMetrics
    feature_names_after_encoding: tuple[str, ...]
    coefficients: pd.DataFrame
    mnlogit_p_values: pd.DataFrame | None
    mnlogit_status: str
    regularization_settings: Mapping[str, object]


@dataclass(frozen=True)
class ProfileProbabilityResult:
    """Class probabilities for one profile, with any imputer-based defaults identified."""

    probabilities: dict[str, float]
    message: str


@dataclass(frozen=True)
class ModelComparison:
    """Model A/B holdout and paired-fold differences; differences do not imply superiority."""

    model_a: PredictionRun
    model_b: PredictionRun
    holdout_accuracy_difference_b_minus_a: float
    holdout_macro_f1_difference_b_minus_a: float
    fold_accuracy_difference_mean: float
    fold_accuracy_difference_sd: float
    fold_macro_f1_difference_mean: float
    fold_macro_f1_difference_sd: float
    interpretation: str


def _feature_columns(feature_set: FeatureSet) -> tuple[str, ...]:
    """Return configured raw predictor names for one approved feature set."""
    if feature_set == "A":
        return MODEL_A_FEATURES
    if feature_set == "B":
        return MODEL_B_FEATURES
    raise ValueError("feature_set must be 'A' or 'B'.")


def _categorical_features(features: tuple[str, ...]) -> list[str]:
    """Select configured binary and nominal columns for one-hot encoding."""
    return [
        column
        for column in features
        if column in config.BINARY_COLUMNS or column in config.NOMINAL_COLUMNS
    ]


def _numeric_features(features: tuple[str, ...]) -> list[str]:
    """Select numeric and ordinal codes for median imputation and standardization."""
    return [column for column in features if column not in _categorical_features(features)]


def build_pipeline(feature_set: FeatureSet = "A") -> Pipeline:
    """Build a fold-safe impute/encode/scale/multinomial LogisticRegression pipeline.

    Numeric and ordinal predictors use median imputation and standard scaling. Configured binary
    and nominal predictors use most-frequent imputation and one-hot encoding. All fitted
    preprocessing lives within sklearn's Pipeline/ColumnTransformer and therefore is learned
    independently on each training split/fold. LogisticRegression uses L2 regularization,
    C=1.0, lbfgs, no class weighting, and a fixed 1,000-iteration limit.
    """
    features = _feature_columns(feature_set)
    categorical = _categorical_features(features)
    numeric = _numeric_features(features)
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
    classifier = LogisticRegression(
        C=1.0,
        l1_ratio=0.0,
        solver="lbfgs",
        class_weight=None,
        max_iter=1000,
        random_state=config.RANDOM_SEED,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def performance_categories(grades: pd.Series) -> pd.Series:
    """Label G3 using the fixed configured Low/Medium/High bands.

    The project currently uses fixed thresholds only. No quantile-based option is implemented;
    therefore category boundaries are independent of train/test splitting and use the values in
    `config.PERFORMANCE_BANDS`.
    """
    bands = config.PERFORMANCE_BANDS
    if bands["LOW_UPPER_EXCLUSIVE"] != bands["MEDIUM_LOWER_INCLUSIVE"]:
        raise PredictionError("Configured Low and Medium grade bands must meet at one boundary.")
    if bands["MEDIUM_UPPER_INCLUSIVE"] + 1 != bands["HIGH_LOWER_INCLUSIVE"]:
        raise PredictionError("Configured Medium and High grade bands must be contiguous.")
    values = pd.to_numeric(grades, errors="coerce")
    invalid = grades.notna() & values.isna()
    if invalid.any():
        raise PredictionError("G3 target contains non-numeric values.")
    non_finite = grades.notna() & values.notna() & ~np.isfinite(values)
    if non_finite.any():
        raise PredictionError("G3 target must contain finite values.")
    categories = pd.Series(pd.NA, index=grades.index, dtype="string", name="PerformanceCategory")
    categories.loc[values < bands["LOW_UPPER_EXCLUSIVE"]] = "Low"
    categories.loc[
        values.between(
            bands["MEDIUM_LOWER_INCLUSIVE"],
            bands["MEDIUM_UPPER_INCLUSIVE"],
            inclusive="both",
        )
    ] = "Medium"
    categories.loc[values >= bands["HIGH_LOWER_INCLUSIVE"]] = "High"
    return categories


def _validate_training_data(
    data: pd.DataFrame, feature_set: FeatureSet
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate expected features and create target labels without altering source data."""
    features = _feature_columns(feature_set)
    missing = [column for column in ("G3", *features) if column not in data.columns]
    if missing:
        raise PredictionError(f"Required prediction column(s) missing: {', '.join(missing)}.")
    if feature_set == "A" and set(MODEL_A_FEATURES).intersection(config.LEAKY_COLUMNS):
        raise PredictionError("Model A configuration includes a leaky grade column.")
    labels = performance_categories(data["G3"])
    complete = labels.notna()
    return data.loc[complete, list(features)].copy(), labels.loc[complete]


def _metrics(
    truth: pd.Series,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    training_labels: pd.Series,
) -> ClassificationMetrics:
    """Calculate imbalanced-class metrics, probability scores, and training-majority baseline."""
    precision, recall, f1_values, _ = precision_recall_fscore_support(
        truth,
        predicted,
        labels=list(CLASS_LABELS),
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1_values},
        index=pd.Index(CLASS_LABELS, name="class"),
    )
    matrix = confusion_matrix(truth, predicted, labels=list(CLASS_LABELS))
    confusion = pd.DataFrame(
        matrix,
        index=pd.Index(CLASS_LABELS, name="actual"),
        columns=pd.Index(CLASS_LABELS, name="predicted"),
    )
    majority = str(training_labels.value_counts().idxmax())
    baseline_predictions = np.repeat(majority, len(truth))
    _, _, base_f1, _ = precision_recall_fscore_support(
        truth,
        baseline_predictions,
        labels=list(CLASS_LABELS),
        zero_division=0,
    )
    one_hot_truth = np.column_stack(
        [(truth.to_numpy() == label).astype(float) for label in CLASS_LABELS]
    )
    brier = float(np.mean(np.sum((probabilities - one_hot_truth) ** 2, axis=1)))
    return ClassificationMetrics(
        accuracy=float(accuracy_score(truth, predicted)),
        per_class=per_class,
        macro_f1=float(
            f1_score(truth, predicted, labels=list(CLASS_LABELS), average="macro", zero_division=0)
        ),
        confusion_matrix=confusion,
        majority_class=majority,
        majority_baseline_accuracy=float(accuracy_score(truth, baseline_predictions)),
        majority_baseline_macro_f1=float(np.mean(base_f1)),
        log_loss=float(
            log_loss(
                truth,
                probabilities[:, [CLASS_LABELS.index(label) for label in sorted(CLASS_LABELS)]],
                labels=sorted(CLASS_LABELS),
            )
        ),
        multiclass_brier_score=brier,
    )


def _cross_validation(
    pipeline: Pipeline,
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    folds: int,
    seed: int,
    split_indices: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> CrossValidationMetrics:
    """Compute stratified-fold accuracy and macro-F1, each fold fitting its own preprocessing."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    cv_splits = (
        split_indices if split_indices is not None else list(splitter.split(features, labels))
    )
    scores = cross_validate(
        pipeline,
        features,
        labels,
        cv=cv_splits,
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
        return_train_score=False,
        error_score="raise",
    )
    accuracy_values = np.asarray(scores["test_accuracy"], dtype=float)
    macro_values = np.asarray(scores["test_macro_f1"], dtype=float)
    return CrossValidationMetrics(
        accuracy_mean=float(np.mean(accuracy_values)),
        accuracy_sd=float(np.std(accuracy_values, ddof=1)) if len(accuracy_values) > 1 else 0.0,
        macro_f1_mean=float(np.mean(macro_values)),
        macro_f1_sd=float(np.std(macro_values, ddof=1)) if len(macro_values) > 1 else 0.0,
        fold_accuracy=tuple(float(value) for value in accuracy_values),
        fold_macro_f1=tuple(float(value) for value in macro_values),
        folds=len(accuracy_values),
    )


def _stratified_holdout(
    indices: np.ndarray,
    labels: pd.Series,
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the reproducible stratified split or translate sklearn sizing errors helpfully."""
    try:
        return train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=labels.to_numpy(),
        )
    except ValueError as exc:
        raise InsufficientClassDataError(
            "A stratified train/test split cannot preserve every class at this test_size; "
            "increase the dataset or adjust test_size."
        ) from exc


def _coefficient_view(pipeline: Pipeline) -> pd.DataFrame:
    """Return class-versus-reference coefficient contrasts and odds ratios with a caveat."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    reference_class = str(classifier.classes_[0])
    records: list[dict[str, str | float]] = []
    for class_index, label in enumerate(classifier.classes_[1:], start=1):
        contrasts = classifier.coef_[class_index] - classifier.coef_[0]
        with np.errstate(over="ignore"):
            odds_ratios = np.exp(contrasts)
        records.extend(
            {
                "class": str(label),
                "reference_class": reference_class,
                "feature": str(feature),
                "coefficient_contrast": float(coefficient),
                "odds_ratio": float(odds_ratio),
                "interpretation_caveat": "model contribution, not causation",
            }
            for feature, coefficient, odds_ratio in zip(
                feature_names, contrasts, odds_ratios, strict=True
            )
        )
    return pd.DataFrame(records)


def _mnlogit_pvalues(
    pipeline: Pipeline,
    train_x: pd.DataFrame,
    train_y: pd.Series,
) -> tuple[pd.DataFrame | None, str]:
    """Attempt unregularized statsmodels MNLogit inference and report convergence explicitly."""
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(train_x)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    design = sm.add_constant(np.asarray(transformed, dtype=float), has_constant="add")
    labels = pd.Categorical(train_y, categories=list(CLASS_LABELS), ordered=True).codes
    model = sm.MNLogit(labels, design)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.fit(method="newton", maxiter=100, disp=False)
        if bool(result.mle_retvals.get("converged", False)):
            p_values = pd.DataFrame(
                np.asarray(result.pvalues),
                index=["const", *preprocessor.get_feature_names_out()],
                columns=CLASS_LABELS[1:],
            )
            status = (
                "statsmodels MNLogit converged on the training split; unregularized p-values "
                "are provided for Medium and High relative to Low, separately from the "
                "L2-regularized sklearn model."
            )
            if caught:
                status += " Warnings: " + "; ".join(str(item.message) for item in caught)
            return p_values, status
        warning_text = "; ".join(str(item.message) for item in caught)
        return (
            None,
            (
                "statsmodels MNLogit did not converge within 100 iterations; inferential p-values "
                "are omitted. " + warning_text
            ).strip(),
        )
    except (ValueError, np.linalg.LinAlgError, OverflowError) as exc:
        return None, (
            "statsmodels MNLogit did not converge or was not identifiable for these encoded "
            f"features; p-values are omitted ({exc})."
        )


def evaluate_model(
    data: pd.DataFrame,
    feature_set: FeatureSet = "A",
    *,
    test_size: float = config.DEFAULT_TEST_SIZE,
    cv_folds: int = config.DEFAULT_CV_FOLDS,
    seed: int = config.RANDOM_SEED,
) -> PredictionRun:
    """Fit and evaluate a feature set with a stratified holdout and stratified CV.

    Split and CV are deterministic for a fixed seed. Class thresholds are the configured fixed
    grade bands; preprocessing is fitted only inside each training split/fold via sklearn
    Pipeline. A class needs at least `cv_folds` samples overall, and each must occur in the
    holdout training and test partition.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    features, labels = _validate_training_data(data, feature_set)
    counts = labels.value_counts().reindex(CLASS_LABELS, fill_value=0)
    if (counts == 0).any():
        absent = [label for label, count in counts.items() if count == 0]
        raise InsufficientClassDataError(
            f"Cannot train a three-class model; class(es) absent: {', '.join(absent)}."
        )
    if counts.min() < cv_folds:
        raise InsufficientClassDataError(
            f"Stratified {cv_folds}-fold CV requires at least {cv_folds} observations per class; "
            f"smallest class has {int(counts.min())}."
        )
    indices = np.arange(len(features))
    train_indices, test_indices = _stratified_holdout(
        indices,
        labels,
        test_size=test_size,
        seed=seed,
    )
    train_y = labels.iloc[train_indices]
    test_y = labels.iloc[test_indices]
    train_counts = train_y.value_counts()
    if (train_counts.reindex(CLASS_LABELS, fill_value=0) == 0).any() or (
        test_y.value_counts().reindex(CLASS_LABELS, fill_value=0) == 0
    ).any():
        raise InsufficientClassDataError(
            "The stratified holdout must contain every class in both train and test partitions."
        )
    if train_counts.min() < cv_folds:
        raise InsufficientClassDataError(
            f"Training split has fewer than {cv_folds} examples in at least one class."
        )

    train_x = features.iloc[train_indices]
    test_x = features.iloc[test_indices]
    pipeline = build_pipeline(feature_set)
    pipeline.fit(train_x, train_y)
    predicted = pipeline.predict(test_x)
    class_indices = [
        int(np.flatnonzero(pipeline.named_steps["classifier"].classes_ == label)[0])
        for label in CLASS_LABELS
    ]
    probabilities = pipeline.predict_proba(test_x)[:, class_indices]
    metrics = _metrics(test_y, predicted, probabilities, train_y)
    cv_metrics = _cross_validation(
        build_pipeline(feature_set),
        train_x,
        train_y,
        folds=cv_folds,
        seed=seed,
    )
    feature_names = tuple(pipeline.named_steps["preprocessor"].get_feature_names_out())
    mnlogit_p_values, mnlogit_status = _mnlogit_pvalues(pipeline, train_x, train_y)
    return PredictionRun(
        feature_set=feature_set,
        pipeline=pipeline,
        class_counts={str(label): int(counts.loc[label]) for label in CLASS_LABELS},
        train_size=len(train_indices),
        test_size=len(test_indices),
        train_indices=tuple(features.index[train_indices].tolist()),
        test_indices=tuple(features.index[test_indices].tolist()),
        metrics=metrics,
        cross_validation=cv_metrics,
        feature_names_after_encoding=feature_names,
        coefficients=_coefficient_view(pipeline),
        mnlogit_p_values=mnlogit_p_values,
        mnlogit_status=mnlogit_status,
        regularization_settings=dict(REGULARIZATION_SETTINGS),
    )


def compare_models(
    data: pd.DataFrame,
    *,
    test_size: float = config.DEFAULT_TEST_SIZE,
    cv_folds: int = config.DEFAULT_CV_FOLDS,
    seed: int = config.RANDOM_SEED,
) -> ModelComparison:
    """Compare Model A/B on identical stratified holdout rows and paired CV folds.

    Returns B-minus-A metric differences and fold-wise spread without presuming either model is
    better. Identical labels, split seed, and fold indices are used for both feature sets.
    """
    features_a, labels_a = _validate_training_data(data, "A")
    _, labels_b = _validate_training_data(data, "B")
    if not labels_a.equals(labels_b):
        raise PredictionError("Model A and Model B must use identical labeled observations.")
    counts = labels_a.value_counts().reindex(CLASS_LABELS, fill_value=0)
    if (counts == 0).any() or counts.min() < cv_folds:
        raise InsufficientClassDataError(
            "Every class must have at least cv_folds observations for a paired model comparison."
        )
    indices = np.arange(len(labels_a))
    train_indices, _ = _stratified_holdout(
        indices,
        labels_a,
        test_size=test_size,
        seed=seed,
    )
    training_labels = labels_a.iloc[train_indices]
    if training_labels.value_counts().min() < cv_folds:
        raise InsufficientClassDataError(
            f"Training partition must have at least {cv_folds} observations per class."
        )
    stratified = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    relative_splits = list(stratified.split(features_a.iloc[train_indices], training_labels))

    model_a = evaluate_model(data, "A", test_size=test_size, cv_folds=cv_folds, seed=seed)
    model_b = evaluate_model(data, "B", test_size=test_size, cv_folds=cv_folds, seed=seed)
    if (
        model_a.train_indices != model_b.train_indices
        or model_a.test_indices != model_b.test_indices
    ):
        raise PredictionError("Model A and Model B were not evaluated on identical holdout rows.")

    # evaluate_model uses the same splitter; verify and compute paired fold differences.
    accuracy_differences = np.asarray(model_b.cross_validation.fold_accuracy) - np.asarray(
        model_a.cross_validation.fold_accuracy
    )
    macro_f1_differences = np.asarray(model_b.cross_validation.fold_macro_f1) - np.asarray(
        model_a.cross_validation.fold_macro_f1
    )
    if len(relative_splits) != len(accuracy_differences):
        raise PredictionError("Paired CV folds do not match between the model feature sets.")
    return ModelComparison(
        model_a=model_a,
        model_b=model_b,
        holdout_accuracy_difference_b_minus_a=(model_b.metrics.accuracy - model_a.metrics.accuracy),
        holdout_macro_f1_difference_b_minus_a=(model_b.metrics.macro_f1 - model_a.metrics.macro_f1),
        fold_accuracy_difference_mean=float(np.mean(accuracy_differences)),
        fold_accuracy_difference_sd=(
            float(np.std(accuracy_differences, ddof=1)) if len(accuracy_differences) > 1 else 0.0
        ),
        fold_macro_f1_difference_mean=float(np.mean(macro_f1_differences)),
        fold_macro_f1_difference_sd=(
            float(np.std(macro_f1_differences, ddof=1)) if len(macro_f1_differences) > 1 else 0.0
        ),
        interpretation=(
            "Differences are Model B minus Model A on identical splits/folds; positive or "
            "negative values describe this evaluation only and do not establish that either "
            "model is generally better."
        ),
    )


def predict_proba_for_profile(
    profile: Mapping[str, object],
    model: PredictionRun,
) -> ProfileProbabilityResult:
    """Return Low/Medium/High probabilities for a profile using fitted imputer defaults.

    Unspecified predictor fields and explicit `None`/NaN values are passed as missing to the
    pipeline, which imputes them using values learned from the training data. Extra profile
    fields are ignored; the fitted feature-set schema determines which inputs are used.
    """
    if not isinstance(profile, Mapping):
        raise ProfilePredictionError("profile must be a mapping of predictor names to values.")
    features = _feature_columns(model.feature_set)
    missing = [column for column in ("G3",) if column in profile]
    if missing:
        raise ProfilePredictionError(
            "G3 is the outcome used to define categories and cannot be supplied as a profile feature."
        )
    row = pd.DataFrame(
        [{column: profile.get(column, np.nan) for column in features}],
        columns=list(features),
    )
    try:
        values = model.pipeline.predict_proba(row)[0]
        class_order = model.pipeline.named_steps["classifier"].classes_
    except (ValueError, TypeError, KeyError) as exc:
        raise ProfilePredictionError(
            f"Profile values could not be scored by this fitted model: {exc}"
        ) from exc
    probabilities = {label: 0.0 for label in CLASS_LABELS}
    probabilities.update(
        {str(label): float(value) for label, value in zip(class_order, values, strict=True)}
    )
    if not np.isclose(sum(probabilities.values()), 1.0, rtol=1e-9, atol=1e-9):
        raise ProfilePredictionError(
            "The fitted model returned probabilities that do not sum to 1."
        )
    missing_features = [
        column for column in features if column not in profile or pd.isna(profile[column])
    ]
    message = (
        "Missing profile fields were filled by imputers fitted on the training data."
        if missing_features
        else "All configured model input fields were supplied."
    )
    return ProfileProbabilityResult(probabilities=probabilities, message=message)


def save_pipeline(model: PredictionRun, filename: str) -> Path:
    """Explicitly save a fitted pipeline beneath the repository's ignored `models/` directory."""
    safe_name = Path(filename).name
    if safe_name != filename or not re.fullmatch(r"[A-Za-z0-9_.-]+", safe_name):
        raise PredictionError("filename must be a simple file name without path components.")
    if not safe_name.lower().endswith((".joblib", ".pkl")):
        raise PredictionError("filename must end with .joblib or .pkl.")
    model_directory = Path(__file__).resolve().parents[1] / "models"
    model_directory.mkdir(parents=True, exist_ok=True)
    destination = model_directory / safe_name
    joblib.dump(model.pipeline, destination)
    return destination
