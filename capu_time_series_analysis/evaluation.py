# -*- coding: utf-8 -*-
"""
Evaluation utilities for time series forecasting models.

This module provides functions to evaluate forecasting models and analyze residuals.
"""

import base64
import io
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox


def evaluate_forecasts(
    test_series: pd.Series, forecasts: List[pd.Series], models: List[str]
) -> pd.DataFrame:
    """
    Evaluate forecast accuracy against test data.

    Args:
        test_series: Actual values (ground truth)
        forecasts: List of forecast series to evaluate
        models: List of model names corresponding to forecasts

    Returns:
        DataFrame with accuracy metrics for each forecast
    """
    results = []

    for i, forecast in enumerate(forecasts):
        # Calculate error metrics
        errors = test_series - forecast
        abs_errors = np.abs(errors)
        squared_errors = errors**2

        mae = abs_errors.mean()
        rmse = np.sqrt(squared_errors.mean())

        # Calculate MAPE if no zeros in test data
        if (test_series == 0).sum() == 0:
            mape = 100 * (abs_errors / test_series).mean()
        else:
            mape = np.nan

        results.append(
            {"Model": f"{models[i]}", "MAE": mae, "RMSE": rmse, "MAPE": mape}
        )

    return pd.DataFrame(results)


def analyze_residuals(
    model, train_series: pd.Series, model_name: str
) -> Dict[str, Any]:
    """
    Analyze residuals of a fitted model to check if they resemble white noise.

    Args:
        model: Fitted time series model (SeasonalNaive, ExponentialSmoothing, or SARIMAX)
        train_series: Original training data
        model_name: Name of the model for identification

    Returns:
        Dictionary with residual analysis metrics and plots
    """
    # Extract residuals based on model type
    if model_name == "Seasonal Naive":
        # Skip the first few observations where we can't calculate fitted values
        residuals = model.residuals
        fitted_values = model.fitted_values[model.seasonal_periods :]
        actual_values = train_series[model.seasonal_periods :]
    elif model_name == "ETS":
        residuals = train_series - model.fittedvalues
        fitted_values = model.fittedvalues
        actual_values = train_series
    elif model_name == "ARIMA":
        residuals = model.resid
        fitted_values = train_series - model.resid
        actual_values = train_series
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # Drop NaN values that might exist in residuals
    residuals = residuals.dropna()

    if len(residuals) == 0:
        return {
            "mean_residual": np.nan,
            "std_residual": np.nan,
            "ljung_box_stat": np.nan,
            "ljung_box_pvalue": np.nan,
            "residuals_normal_pvalue": np.nan,
            "residuals_plot": None,
            "acf_plot": None,
            "white_noise": False,
        }

    # 1. Basic statistics
    mean_residual = residuals.mean()
    std_residual = residuals.std()

    # 2. Ljung-Box test for autocorrelation
    # The null hypothesis is that the residuals are independently distributed
    lags = min(10, len(residuals) // 5)  # Rule of thumb for number of lags
    if lags > 0:
        lb_result = acorr_ljungbox(residuals, lags=[lags])
        lb_stat = lb_result["lb_stat"].iloc[0]
        lb_pvalue = lb_result["lb_pvalue"].iloc[0]
    else:
        lb_stat = np.nan
        lb_pvalue = np.nan

    # 3. Normality test (Shapiro-Wilk)
    if (
        len(residuals) >= 3
    ):  # Shapiro-Wilk test requires at least 3 observations
        _, norm_pvalue = stats.shapiro(residuals)
    else:
        norm_pvalue = np.nan

    # 4. Create residuals plot
    plt.figure(figsize=(10, 8))

    # Subplot 1: Residual time plot
    plt.subplot(2, 1, 1)
    plt.plot(residuals.index, residuals, "o-")
    plt.axhline(y=0, color="r", linestyle="-")
    plt.title(f"Residual Analysis for {model_name}")
    plt.ylabel("Residuals")

    # Subplot 2: ACF plot
    plt.subplot(2, 1, 2)
    if len(residuals) > 5:  # Need reasonable number of points for ACF
        plot_acf(
            residuals.values, lags=min(20, len(residuals) - 1), alpha=0.05
        )
        plt.title("ACF of Residuals")
    else:
        plt.text(
            0.5,
            0.5,
            "Insufficient data for ACF plot",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.title("ACF of Residuals (not enough data)")

    plt.tight_layout()

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")

    # Create a second figure for actual vs fitted
    plt.figure(figsize=(10, 6))
    plt.plot(actual_values.index, actual_values, "b-", label="Actual")
    plt.plot(fitted_values.index, fitted_values, "r--", label="Fitted")
    plt.title(f"Actual vs Fitted Values - {model_name}")
    plt.legend()
    plt.tight_layout()

    # Save the second plot to a bytes buffer
    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png")
    plt.close()
    buf2.seek(0)
    fit_img_str = base64.b64encode(buf2.read()).decode("utf-8")

    # Determine if residuals resemble white noise
    # Criteria: p-value of Ljung-Box test > 0.05, mean close to 0
    white_noise = (
        lb_pvalue > 0.05 if not np.isnan(lb_pvalue) else False
    ) and (abs(mean_residual) < 0.1 * std_residual)

    return {
        "mean_residual": mean_residual,
        "std_residual": std_residual,
        "ljung_box_stat": lb_stat,
        "ljung_box_pvalue": lb_pvalue,
        "residuals_normal_pvalue": norm_pvalue,
        "residuals_plot": img_str,
        "actual_fitted_plot": fit_img_str,
        "white_noise": white_noise,
    }


def calculate_residuals(
    fit_snaive: Any,
    fit_ets: Any,
    fit_arima: Any,
    train_series: pd.Series,
    residual_diagnostics: List[Dict[str, Any]],
    mt: str,
    resd: str,
    level: str,
) -> List[Dict[str, Any]]:
    """
    Calculate residuals for each model and add diagnostics to the collection.
    """
    snaive_residuals = analyze_residuals(
        fit_snaive, train_series, "Seasonal Naive"
    )
    ets_residuals = analyze_residuals(fit_ets, train_series, "ETS")
    arima_residuals = analyze_residuals(fit_arima, train_series, "ARIMA")

    # Add residual diagnostics to the collection
    for model_name, diag in [
        ("Seasonal Naive", snaive_residuals),
        ("ETS", ets_residuals),
        ("ARIMA", arima_residuals),
    ]:
        residual_diagnostics.append(
            {
                "Analysis_Type": mt,
                "Residency": resd,
                "Level": level,
                "Model": model_name,
                "Mean_Residual": diag["mean_residual"],
                "Std_Residual": diag["std_residual"],
                "Ljung_Box_Stat": diag["ljung_box_stat"],
                "Ljung_Box_pvalue": diag["ljung_box_pvalue"],
                "Normality_pvalue": diag["residuals_normal_pvalue"],
                "White_Noise": diag["white_noise"],
                "Residuals_Plot": diag["residuals_plot"],
                "Actual_Fitted_Plot": diag["actual_fitted_plot"],
            }
        )
    return residual_diagnostics
