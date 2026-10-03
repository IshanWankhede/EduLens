"""Multiple linear regression with explicit encoding, diagnostics, and cautious interpretations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src import config
from src.assumptions import CheckResult, shapiro_check
from src.preprocessing import encode_binary, encode_nominal

OrdinalMode = Literal["numeric", "categorical"]


class RegressionError(ValueError):
    """Base exception for invalid or unidentifiable regression inputs."""


class LeakyPredictorError(RegressionError):
    """Raised when G1/G2 are requested without explicit Model B authorization."""


class InsufficientRegressionDataError(RegressionError):
    """Raised when too few complete observations remain to estimate the requested model."""


class CollinearPredictorError(RegressionError):
    """Raised when the encoded design matrix is rank deficient."""


@dataclass(frozen=True)
class RegressionDiagnostics:
    """Observation-level diagnostics, collinearity, and residual assumption checks."""

    fitted_values: pd.Series
    residuals: pd.Series
    standardized_residuals: pd.Series
    leverage: pd.Series
    cooks_distance: pd.Series
    vif_table: pd.DataFrame
    breusch_pagan: CheckResult
    shapiro_wilk: CheckResult
    assumptions: tuple[CheckResult, ...]
    alternatives: tuple[str, ...]


@dataclass(frozen=True)
class RegressionResult:
    """OLS estimates and model diagnostics, in TEST/STATISTIC/P-VALUE display order."""

    model_label: str
    target: str
    predictors: tuple[str, ...]
    test: str
    statistic: float
    p_value: float
    coefficients: pd.DataFrame
    r_squared: float
    adjusted_r_squared: float
    f_statistic: float
    f_p_value: float
    residual_standard_error: float
    sample_size: int
    residual_degrees_of_freedom: float
    confidence_level: float
    robust_se: Literal["nonrobust", "HC3"]
    ordinal_mode: OrdinalMode
    encoding: dict[str, object]
    interpretations: dict[str, str]
    diagnostics: RegressionDiagnostics
    decision: str
    interpretation: str


def _validate_columns(
    data: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
    allow_model_b: bool,
) -> tuple[str, ...]:
    """Validate names and gate leaky G1/G2 predictors behind the explicit Model B flag."""
    if data.empty:
        raise InsufficientRegressionDataError("Regression requires a non-empty dataset.")
    if target not in data.columns:
        raise RegressionError(f"Target column '{target}' is not present in the dataset.")
    selected = tuple(predictors)
    if not selected:
        raise RegressionError("At least one predictor is required.")
    if len(selected) != len(set(selected)):
        raise RegressionError("Predictor names must be unique.")
    if target in selected:
        raise RegressionError("The target column cannot also be a predictor.")
    missing = [column for column in selected if column not in data.columns]
    if missing:
        raise RegressionError(f"Predictor column(s) missing from dataset: {', '.join(missing)}.")
    return selected


def _encode_design(
    frame: pd.DataFrame,
    predictors: tuple[str, ...],
    ordinal_mode: OrdinalMode,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Encode predictors using project roles and return design columns with encoding metadata."""
    numeric_design = pd.DataFrame(index=frame.index)
    binary_metadata: dict[str, dict[str, object]] = {}
    nominal_columns: list[str] = []
    ordinal_categorical_columns: list[str] = []
    ordinal_metadata: dict[str, str] = {}

    for column in predictors:
        series = frame[column]
        if column in config.BINARY_COLUMNS:
            categories = sorted(series.astype(str).unique().tolist())
            if len(categories) != 2:
                raise RegressionError(
                    f"Binary predictor '{column}' must have exactly two observed categories; "
                    f"found {len(categories)}."
                )
            mapped = encode_binary(series, positive_label=categories[1]).astype(float)
            numeric_design[column] = mapped
            binary_metadata[column] = {
                "encoding": "0/1 indicator",
                "reference_level": categories[0],
                "positive_level": categories[1],
            }
        elif column in config.NOMINAL_COLUMNS:
            nominal_columns.append(column)
        elif column in config.ORDINAL_COLUMNS:
            if ordinal_mode == "numeric":
                numeric = pd.to_numeric(series, errors="coerce")
                if numeric.isna().any() or not np.isfinite(numeric).all():
                    raise RegressionError(
                        f"Ordinal predictor '{column}' must contain finite numeric category codes."
                    )
                numeric_design[column] = numeric.astype(float)
                ordinal_metadata[column] = "ordinal codes treated as numeric one-step increments"
            else:
                ordinal_categorical_columns.append(column)
        elif pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(
            series.dtype
        ):
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.isna().any() or not np.isfinite(numeric).all():
                raise RegressionError(f"Numeric predictor '{column}' must contain finite values.")
            numeric_design[column] = numeric.astype(float)
        else:
            raise RegressionError(
                f"Predictor '{column}' has no configured binary, nominal, ordinal, or numeric role."
            )

    encoding: dict[str, object] = {
        "binary": binary_metadata,
        "nominal": {},
        "ordinal": {"mode": ordinal_mode, "columns": ordinal_metadata},
    }
    if nominal_columns:
        nominal_encoded, references = encode_nominal(frame[nominal_columns], nominal_columns)
        dummy_columns = nominal_encoded
        numeric_design = numeric_design.join(dummy_columns)
        encoding["nominal"] = {
            column: {
                "encoding": "one-hot dummies with lexicographically first category as reference",
                "reference_level": references[column],
                "dummy_columns": [
                    name for name in dummy_columns.columns if name.startswith(f"{column}_")
                ],
            }
            for column in nominal_columns
        }
    for column in ordinal_categorical_columns:
        categories = sorted(frame[column].unique().tolist())
        categorical = pd.Categorical(frame[column], categories=categories, ordered=True)
        dummies = pd.get_dummies(
            categorical,
            prefix=column,
            drop_first=True,
            dtype=float,
        )
        dummies.index = frame.index
        numeric_design = numeric_design.join(dummies)
        encoding["ordinal"]["columns"][column] = {
            "mode": "categorical dummy coding",
            "reference_level": categories[0],
            "levels": categories,
        }

    if numeric_design.empty:
        raise RegressionError("The requested predictors produced no design-matrix columns.")
    numeric_design = numeric_design.astype(float)
    if not np.isfinite(numeric_design.to_numpy()).all():
        raise RegressionError("Encoded predictors must contain only finite values.")
    return numeric_design, encoding


def _vif_table(design: pd.DataFrame) -> pd.DataFrame:
    """Compute VIFs on encoded predictors, excluding the intercept."""
    columns = list(design.columns)
    if len(columns) == 1:
        return pd.DataFrame({"predictor": columns, "vif": [1.0]})
    matrix = design.to_numpy(dtype=float)
    with_intercept = sm.add_constant(matrix, has_constant="add")
    values = [
        float(variance_inflation_factor(with_intercept, index + 1)) for index in range(len(columns))
    ]
    return pd.DataFrame({"predictor": columns, "vif": values})


def _coefficient_interpretations(
    coefficients: pd.DataFrame,
    encoding: dict[str, object],
    target: str,
    confidence_level: float,
) -> dict[str, str]:
    """Create non-causal, reference-aware coefficient explanations."""
    details: dict[str, str] = {}
    binary = encoding["binary"]
    nominal = encoding["nominal"]
    ordinal = encoding["ordinal"]["columns"]
    for term, row in coefficients.iterrows():
        term_name = str(term)
        if term_name == "const":
            details[term_name] = (
                f"The intercept is the estimated {target} value at the reference levels and "
                "zero for numeric predictors; interpret only if that baseline is meaningful."
            )
            continue
        if term_name in binary:
            info = binary[term_name]
            wording = (
                f"the {info['positive_level']} level versus reference {info['reference_level']}"
            )
        elif term_name in nominal:
            info = nominal[term_name]
            prefix = f"{term_name}_"
            level = term_name.removeprefix(prefix) if term_name.startswith(prefix) else term_name
            wording = f"level {level} versus reference {info['reference_level']}"
        elif any(
            isinstance(info, dict) and term_name.startswith(f"{column}_")
            for column, info in ordinal.items()
        ):
            column = next(column for column in ordinal if term_name.startswith(f"{column}_"))
            info = ordinal[column]
            level = term_name.removeprefix(f"{column}_")
            wording = f"ordinal level {level} versus reference {info['reference_level']}"
        else:
            wording = "a one-unit increase"
        details[term_name] = (
            f"Holding the other included variables constant, {wording} is associated with an "
            f"estimated {row['coef']:.4g}-point change in {target} "
            f"({confidence_level:.0%} CI [{row['ci_lower']:.4g}, {row['ci_upper']:.4g}], "
            f"p={row['p_value']:.4g}); "
            "this is not a causal effect."
        )
    return details


def fit_ols(
    data: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
    *,
    exclude_leaky: bool = True,
    allow_model_b: bool = False,
    ordinal_mode: OrdinalMode = "numeric",
    confidence_level: float = config.DEFAULT_CONFIDENCE,
    robust_se: bool = False,
) -> RegressionResult:
    """Fit OLS with project-role encoding, diagnostics, optional HC3 standard errors.

    Missing rows among the selected variables are excluded as complete cases. Binary predictors
    use 0/1 indicators (sorted category 0 is reference); nominal predictors use one-hot dummies
    with the lexicographically first observed category as reference; ordinal predictors default
    to numeric codes, or use categorical dummies with `ordinal_mode="categorical"`. A G3 model
    containing G1/G2 is rejected unless `allow_model_b=True`, in which case it is labeled Model B.
    `exclude_leaky` is retained as a guard-compatible option: setting it False does not bypass the
    explicit Model B authorization. HC3 changes coefficient SEs, t/p values, and CIs, not fitted
    values or the OLS coefficient estimates.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if ordinal_mode not in {"numeric", "categorical"}:
        raise ValueError("ordinal_mode must be 'numeric' or 'categorical'.")
    selected = _validate_columns(data, target, predictors, allow_model_b)
    has_grade_predictor = target == "G3" and any(column in {"G1", "G2"} for column in selected)
    if has_grade_predictor and not allow_model_b:
        if exclude_leaky:
            message = (
                "G1 and G2 are excluded from factor-only Model A; set allow_model_b=True "
                "to explicitly run and label a Model B regression."
            )
        else:
            message = (
                "Disabling exclude_leaky does not authorize G1/G2; set allow_model_b=True "
                "explicitly."
            )
        raise LeakyPredictorError(message)

    missing = [column for column in (target, *selected) if data[column].isna().any()]
    complete = data.loc[:, [target, *selected]].dropna().copy()
    if complete.empty:
        raise InsufficientRegressionDataError("No complete cases remain for the regression.")
    try:
        y = pd.to_numeric(complete[target], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise RegressionError(f"Target '{target}' must be numeric.") from exc
    if not np.isfinite(y.to_numpy()).all():
        raise RegressionError(f"Target '{target}' must contain finite numeric values.")

    design, encoding = _encode_design(complete, selected, ordinal_mode)
    constant_columns = design.columns[design.nunique(dropna=False) < 2].tolist()
    if constant_columns:
        raise RegressionError(
            f"Constant predictor(s) cannot be estimated: {', '.join(constant_columns)}."
        )
    design = sm.add_constant(design, has_constant="add")
    n, parameter_count = design.shape
    if n <= parameter_count:
        raise InsufficientRegressionDataError(
            f"Regression has {n} complete rows for {parameter_count} coefficients; "
            "more observations than coefficients are required."
        )
    matrix = design.to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(matrix))
    if rank < parameter_count:
        raise CollinearPredictorError(
            "The encoded predictors are perfectly collinear; remove or combine redundant predictors."
        )

    ordinary = sm.OLS(y, design).fit()
    selected_fit = (
        ordinary.get_robustcov_results(cov_type="HC3", use_t=False) if robust_se else ordinary
    )
    alpha = 1 - confidence_level
    ci = np.asarray(selected_fit.conf_int(alpha=alpha), dtype=float)
    coefficient_table = pd.DataFrame(
        {
            "coef": np.asarray(selected_fit.params, dtype=float),
            "standard_error": np.asarray(selected_fit.bse, dtype=float),
            "t_value": np.asarray(selected_fit.tvalues, dtype=float),
            "p_value": np.asarray(selected_fit.pvalues, dtype=float),
            "ci_lower": ci[:, 0],
            "ci_upper": ci[:, 1],
        },
        index=design.columns,
    )
    coefficient_table.index.name = "term"

    influence = ordinary.get_influence()
    fitted = pd.Series(np.asarray(ordinary.fittedvalues), index=complete.index, name="fitted")
    residuals = pd.Series(np.asarray(ordinary.resid), index=complete.index, name="residual")
    standardized = pd.Series(
        np.asarray(influence.resid_studentized_internal),
        index=complete.index,
        name="standardized_residual",
    )
    leverage = pd.Series(
        np.asarray(influence.hat_matrix_diag), index=complete.index, name="leverage"
    )
    cooks = pd.Series(
        np.asarray(influence.cooks_distance[0]), index=complete.index, name="cooks_distance"
    )
    bp_lm, bp_p, _, _ = het_breuschpagan(ordinary.resid, ordinary.model.exog)
    bp_warning = bp_p < config.DEFAULT_ALPHA
    bp_check = CheckResult(
        name="Breusch–Pagan homoscedasticity",
        status="warning" if bp_warning else "ok",
        message=(
            f"Breusch–Pagan p={bp_p:.4g}; "
            + (
                "the result flags possible heteroscedasticity."
                if bp_warning
                else "the check did not flag heteroscedasticity."
            )
        ),
        statistic=float(bp_lm),
        p_value=float(bp_p),
    )
    shapiro = shapiro_check(
        ordinary.resid,
        name="Shapiro–Wilk residual normality",
        alpha=config.DEFAULT_ALPHA,
    )
    vif = _vif_table(design.drop(columns="const"))
    high_vif = vif.loc[vif["vif"] > 5, "predictor"].tolist()
    vif_check = CheckResult(
        name="Variance inflation factor",
        status="warning" if high_vif else "ok",
        message=(
            f"VIF exceeds the screening guideline of 5 for: {', '.join(map(str, high_vif))}."
            if high_vif
            else "All predictor VIF values are at or below the screening guideline of 5."
        ),
    )
    influence_warning = bool((cooks > 4 / n).any())
    influence_check = CheckResult(
        name="Influence screening (Cook's distance)",
        status="warning" if influence_warning else "ok",
        message=(
            f"At least one observation has Cook's distance above the screening threshold 4/n "
            f"({4 / n:.4g}); inspect influential points."
            if influence_warning
            else f"No observation exceeds the screening threshold 4/n ({4 / n:.4g})."
        ),
    )
    linearity_check = CheckResult(
        name="Linearity and independent errors",
        status="warning",
        message=(
            "These design assumptions are not established by an automatic test; inspect residuals "
            "versus fitted values and consider the sampling design."
        ),
    )
    assumptions = (linearity_check, bp_check, shapiro, vif_check, influence_check)
    alternatives: list[str] = []
    if bp_warning and not robust_se:
        alternatives.append(
            "Refit with robust_se=True (HC3) for heteroscedasticity-robust inference."
        )
    if shapiro.status == "warning":
        alternatives.append(
            "Inspect the residual Q–Q plot; consider a justified transformation or alternative model."
        )
    if high_vif:
        alternatives.append(
            "Review correlated predictors; combine, remove, or recode redundant terms."
        )
    if influence_warning:
        alternatives.append(
            "Inspect high-influence observations and verify their data; do not remove them automatically."
        )
    if not alternatives:
        alternatives.append(
            "No diagnostic-triggered alternative; inspect residual plots and study design."
        )

    is_model_b = target == "G3" and any(column in {"G1", "G2"} for column in selected)
    model_label = (
        "Model B (includes G1/G2)"
        if is_model_b
        else ("Model A (factor-only)" if target == "G3" else "OLS model")
    )
    interpretations = _coefficient_interpretations(
        coefficient_table, encoding, target, confidence_level
    )
    f_statistic = float(ordinary.fvalue) if ordinary.fvalue is not None else float("nan")
    f_p_value = float(ordinary.f_pvalue) if ordinary.f_pvalue is not None else float("nan")
    decision = (
        "Sufficient evidence that at least one included slope differs from zero"
        if f_p_value < config.DEFAULT_ALPHA
        else "Insufficient evidence that at least one included slope differs from zero"
    )
    model_interpretation = (
        f"{model_label} OLS explains R²={ordinary.rsquared:.4g} of the observed variation in "
        f"{target} (adjusted R²={ordinary.rsquared_adj:.4g}; F={f_statistic:.4g}, "
        f"p={f_p_value:.4g}, n={n}). Coefficients describe associations conditional on the "
        "other included variables and do not establish causation."
    )
    if missing:
        model_interpretation += (
            f" {len(data) - n} row(s) were excluded as incomplete or non-finite cases."
        )
    return RegressionResult(
        model_label=model_label,
        target=target,
        predictors=selected,
        test="Ordinary least squares multiple linear regression",
        statistic=f_statistic,
        p_value=f_p_value,
        coefficients=coefficient_table,
        r_squared=float(ordinary.rsquared),
        adjusted_r_squared=float(ordinary.rsquared_adj),
        f_statistic=f_statistic,
        f_p_value=f_p_value,
        residual_standard_error=float(np.sqrt(ordinary.mse_resid)),
        sample_size=int(n),
        residual_degrees_of_freedom=float(ordinary.df_resid),
        confidence_level=confidence_level,
        robust_se="HC3" if robust_se else "nonrobust",
        ordinal_mode=ordinal_mode,
        encoding=encoding,
        interpretations=interpretations,
        diagnostics=RegressionDiagnostics(
            fitted_values=fitted,
            residuals=residuals,
            standardized_residuals=standardized,
            leverage=leverage,
            cooks_distance=cooks,
            vif_table=vif,
            breusch_pagan=bp_check,
            shapiro_wilk=shapiro,
            assumptions=assumptions,
            alternatives=tuple(alternatives),
        ),
        decision=decision,
        interpretation=model_interpretation,
    )
