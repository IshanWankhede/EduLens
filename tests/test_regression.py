from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.regression import (
    CollinearPredictorError,
    InsufficientRegressionDataError,
    LeakyPredictorError,
    RegressionError,
    fit_ols,
)


@pytest.fixture
def linear_fixture() -> pd.DataFrame:
    x1 = np.arange(1.0, 16.0)
    x2 = np.array([2, 5, 1, 7, 4, 9, 3, 8, 6, 11, 10, 12, 14, 13, 15], dtype=float)
    noise = np.array(
        [0.2, -0.3, 0.1, -0.2, 0.4, -0.1, 0.3, -0.4, 0.2, -0.2, 0.1, 0.3, -0.3, 0.2, -0.1]
    )
    return pd.DataFrame({"outcome": 3.0 + 2.5 * x1 - 0.75 * x2 + noise, "x1": x1, "x2": x2})


def test_ols_matches_statsmodels_and_numpy_normal_equation(
    linear_fixture: pd.DataFrame,
) -> None:
    result = fit_ols(linear_fixture, "outcome", ["x1", "x2"])
    design = sm.add_constant(linear_fixture[["x1", "x2"]], has_constant="add")
    direct = sm.OLS(linear_fixture["outcome"], design).fit()
    expected_beta = np.linalg.solve(
        design.to_numpy().T @ design.to_numpy(),
        design.to_numpy().T @ linear_fixture["outcome"].to_numpy(),
    )

    assert result.model_label == "OLS model"
    assert result.sample_size == len(linear_fixture)
    assert result.coefficients["coef"].to_numpy() == pytest.approx(direct.params.to_numpy())
    assert result.coefficients["coef"].to_numpy() == pytest.approx(expected_beta)
    assert result.coefficients["standard_error"].to_numpy() == pytest.approx(direct.bse.to_numpy())
    assert result.coefficients["t_value"].to_numpy() == pytest.approx(direct.tvalues.to_numpy())
    assert result.coefficients["p_value"].to_numpy() == pytest.approx(direct.pvalues.to_numpy())
    assert result.r_squared == pytest.approx(direct.rsquared)
    assert result.adjusted_r_squared == pytest.approx(direct.rsquared_adj)
    assert result.f_statistic == pytest.approx(direct.fvalue)
    assert result.f_p_value == pytest.approx(direct.f_pvalue)
    assert result.residual_standard_error == pytest.approx(np.sqrt(direct.mse_resid))
    assert len(result.diagnostics.fitted_values) == result.sample_size
    assert len(result.diagnostics.residuals) == result.sample_size
    assert len(result.diagnostics.standardized_residuals) == result.sample_size
    assert len(result.diagnostics.leverage) == result.sample_size
    assert len(result.diagnostics.cooks_distance) == result.sample_size
    assert "Holding the other included variables constant" in result.interpretations["x1"]
    assert "causal" in result.interpretations["x1"]
    assert result.diagnostics.breusch_pagan.name.startswith("Breusch")
    assert result.diagnostics.shapiro_wilk.name.startswith("Shapiro")


def test_g3_factor_model_rejects_g1_g2_unless_model_b_is_explicit() -> None:
    x = np.arange(1.0, 16.0)
    data = pd.DataFrame(
        {
            "G3": 4 + 0.8 * x + np.sin(x),
            "G1": x,
            "G2": x + np.cos(x),
            "age": 15 + np.square(x) / 20,
        }
    )
    with pytest.raises(LeakyPredictorError, match="allow_model_b=True"):
        fit_ols(data, "G3", ["G1", "age"])

    result = fit_ols(data, "G3", ["G1", "G2", "age"], allow_model_b=True)
    assert result.model_label == "Model B (includes G1/G2)"
    assert "G1" in result.predictors and "G2" in result.predictors

    with pytest.raises(LeakyPredictorError, match="does not authorize"):
        fit_ols(data, "G3", ["G2", "age"], exclude_leaky=False)


def test_encoding_metadata_for_binary_nominal_and_ordinal_predictors() -> None:
    index = np.arange(20)
    data = pd.DataFrame(
        {
            "outcome": 2 + index * 0.4 + np.sin(index * 1.4),
            "sex": ["F", "M"] * 10,
            "school": ["GP", "GP", "MS", "MS"] * 5,
            "studytime": np.random.default_rng(42).choice([1, 2, 3, 4], size=20),
        }
    )
    result = fit_ols(data, "outcome", ["sex", "school", "studytime"])

    assert result.encoding["binary"]["sex"]["reference_level"] == "F"
    assert result.encoding["binary"]["sex"]["positive_level"] == "M"
    assert result.encoding["nominal"]["school"]["reference_level"] == "GP"
    assert result.ordinal_mode == "numeric"
    assert "numeric one-step increments" in result.encoding["ordinal"]["columns"]["studytime"]

    categorical = fit_ols(
        data,
        "outcome",
        ["sex", "school", "studytime"],
        ordinal_mode="categorical",
    )
    ordinal_info = categorical.encoding["ordinal"]["columns"]["studytime"]
    assert ordinal_info["reference_level"] == 1
    assert ordinal_info["levels"] == [1, 2, 3, 4]
    assert any(column.startswith("studytime_") for column in categorical.coefficients.index)


def test_hc3_changes_standard_errors_and_respects_configured_confidence_level(
    linear_fixture: pd.DataFrame,
) -> None:
    result = fit_ols(
        linear_fixture,
        "outcome",
        ["x1", "x2"],
        robust_se=True,
        confidence_level=0.90,
    )
    design = sm.add_constant(linear_fixture[["x1", "x2"]], has_constant="add")
    direct = sm.OLS(linear_fixture["outcome"], design).fit(cov_type="HC3")

    assert result.robust_se == "HC3"
    assert result.confidence_level == 0.90
    assert result.coefficients["standard_error"].to_numpy() == pytest.approx(direct.bse.to_numpy())
    assert result.coefficients["p_value"].to_numpy() == pytest.approx(
        direct.pvalues.to_numpy(), abs=1e-10
    )
    assert "90% CI" in result.interpretations["x1"]


def test_regression_rejects_collinearity_too_few_rows_and_constant_predictors(
    linear_fixture: pd.DataFrame,
) -> None:
    collinear = linear_fixture.assign(x3=linear_fixture["x1"] * 2)
    with pytest.raises(CollinearPredictorError, match="perfectly collinear"):
        fit_ols(collinear, "outcome", ["x1", "x3"])

    with pytest.raises(RegressionError, match="Constant predictor"):
        fit_ols(linear_fixture.assign(constant=1.0), "outcome", ["x1", "constant"])

    too_small = pd.DataFrame({"outcome": [1.0, 2.0], "x": [0.0, 1.0]})
    with pytest.raises(InsufficientRegressionDataError, match="more observations"):
        fit_ols(too_small, "outcome", ["x"])


def test_regression_drops_incomplete_rows_and_reports_n(linear_fixture: pd.DataFrame) -> None:
    data = linear_fixture.copy()
    data.loc[0, "x1"] = np.nan
    result = fit_ols(data, "outcome", ["x1", "x2"])

    assert result.sample_size == len(data) - 1
    assert "row(s) were excluded" in result.interpretation
    assert 0 not in result.diagnostics.fitted_values.index
