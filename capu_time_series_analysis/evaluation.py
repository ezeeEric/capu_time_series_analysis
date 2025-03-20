# -*- coding: utf-8 -*-
"""
Evaluation utilities for time series forecasting models.

This module provides functions to evaluate forecasting models and analyze residuals.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from capu_time_series_analysis.plotting import (
    fig_to_base64,
    plot_actual_vs_fitted,
    plot_residuals_analysis,
    save_figure,
)


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
    model,
    train_series: pd.Series,
    model_name: str,
    save_plots: bool = False,
    plots_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze residuals of a fitted model to check if they resemble white noise.

    Args:
        model: Fitted time series model (SeasonalNaive, ExponentialSmoothing, or SARIMAX)
        train_series: Original training data
        model_name: Name of the model for identification
        save_plots: Whether to save plots to disk (default: False)
        plots_dir: Directory to save plots to (required if save_plots is True)

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
            "actual_fitted_plot": None,
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

    # 4. Create plots using the plotting utilities
    residuals_fig, acf_fig = plot_residuals_analysis(residuals, model_name)
    actual_fitted_fig = plot_actual_vs_fitted(
        actual_values, fitted_values, model_name
    )

    # Convert figures to base64 strings
    residuals_img_str = fig_to_base64(residuals_fig)
    acf_img_str = fig_to_base64(acf_fig)
    fit_img_str = fig_to_base64(actual_fitted_fig)

    # Save plots to disk if requested
    if save_plots and plots_dir is not None:
        os.makedirs(plots_dir, exist_ok=True)
        save_figure(
            residuals_fig,
            os.path.join(plots_dir, f"{model_name}_residuals.png"),
        )
        save_figure(acf_fig, os.path.join(plots_dir, f"{model_name}_acf.png"))
        save_figure(
            actual_fitted_fig,
            os.path.join(plots_dir, f"{model_name}_actual_vs_fitted.png"),
        )

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
        "residuals_plot": residuals_img_str,
        "acf_plot": acf_img_str,
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
    save_plots: bool = False,
    plots_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Calculate residuals for each model and add diagnostics to the collection.

    Args:
        fit_snaive: Fitted seasonal naive model
        fit_ets: Fitted ETS model
        fit_arima: Fitted ARIMA model
        train_series: Original training data
        residual_diagnostics: Collection to add diagnostics to
        mt: Analysis type
        resd: Residency type
        level: Level identifier
        save_plots: Whether to save plots to disk (default: False)
        plots_dir: Directory to save plots to (required if save_plots is True)
    """
    # If saving plots, create a specific directory for this analysis
    if save_plots and plots_dir is not None:
        analysis_dir = os.path.join(plots_dir, f"{mt}_{resd}_{level}")
        os.makedirs(analysis_dir, exist_ok=True)
    else:
        analysis_dir = None

    snaive_residuals = analyze_residuals(
        fit_snaive, train_series, "Seasonal Naive", save_plots, analysis_dir
    )
    ets_residuals = analyze_residuals(
        fit_ets, train_series, "ETS", save_plots, analysis_dir
    )
    arima_residuals = analyze_residuals(
        fit_arima, train_series, "ARIMA", save_plots, analysis_dir
    )

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
