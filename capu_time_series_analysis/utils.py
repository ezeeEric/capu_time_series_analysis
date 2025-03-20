# -*- coding: utf-8 -*-
"""
Contains utility functions for processing time series data.
"""
# -*- coding: utf-8 -*-
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Import functions used in process_timeseries but defined elsewhere
from capu_time_series_analysis.data_loader import (
    add_to_consolidated_df,
    load_data,
    prepare_time_series,
)
from capu_time_series_analysis.evaluation import (
    calculate_residuals,
    evaluate_forecasts,
)
from capu_time_series_analysis.models import fit_models

logger = logging.getLogger(__name__)


def process_timeseries(
    input_file: str,
    metrics: List[str],
    residencies: List[str],
    levels: List[str],
    forecast_steps: int = 9,
    model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    save_plots: bool = False,
    plots_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process time series data for different combinations of metrics, residencies, and levels.

    This function processes each combination of metrics, residencies, and levels through the following steps:
    1. Loads data from the input file for the specific combination
    2. Prepares and splits the time series into training and test sets
    3. Fits three forecasting models: Seasonal Naive, ETS, and ARIMA
    4. Analyzes residuals for each model to evaluate their fit quality
    5. Generates forecasts for the test period to validate model performance
    6. Creates future forecasts for the specified number of steps ahead
    7. Evaluates forecast accuracy on the test set using multiple metrics
    8. Consolidates all results into a single dataframe for easy analysis

    The process is repeated for all combinations of metrics, residencies, and levels,
    resulting in a comprehensive analysis across all specified dimensions.

    Args:
        input_file (str): Path to the input CSV file
        metrics (list): List of metrics to analyze
        residencies (list): List of residency statuses
        levels (list): List of academic levels
        forecast_steps (int): Number of steps to forecast into the future
        model_params (dict): Parameters for the forecasting models
        save_plots (bool): Whether to save plots to disk (default: False)
        plots_dir (str, optional): Directory to save plots to (required if save_plots is True)

    Returns:
        tuple: (consolidated_df, evaluation_results, residual_diagnostics)
            - consolidated_df: DataFrame containing all time series data, forecasts, and actuals
            - evaluation_results: List of dictionaries with forecast accuracy metrics
            - residual_diagnostics: List of dictionaries with residual analysis results
    """
    if model_params is None:
        model_params = {}

    total_combinations = len(levels) * len(residencies) * len(metrics)
    logger.info(
        "Processing %d combinations of levels, residencies, and metrics",
        total_combinations,
    )

    current_combination = 0

    # Initialize consolidated dataframe
    consolidated_df = pd.DataFrame(
        columns=[
            "Analysis_Type",
            "Residency",
            "Level",
            "Timestamp",
            "Model",
            "Entry_Type",
            "Entry",
        ]
    )

    # Dictionary to store evaluation metrics
    evaluation_results: List[Dict[str, Any]] = []

    # Dictionary to store residual diagnostics
    residual_diagnostics: List[Dict[str, Any]] = []

    # Process all combinations and build consolidated dataframe
    for mt in metrics:
        for resd in residencies:
            for level in levels:
                current_combination += 1
                logger.info(
                    "Processing %s - %s - %s (%d/%d)",
                    level,
                    resd,
                    mt,
                    current_combination,
                    total_combinations,
                )

                # Load data
                df_subset = load_data(input_file, level, resd, mt)
                if df_subset.empty:
                    logger.warning(
                        "No data for %s - %s - %s, skipping...",
                        level,
                        resd,
                        mt,
                    )
                    continue

                # Prepare time series data
                train_series, test_series = prepare_time_series(df_subset, mt)
                logger.debug(
                    "Split data into %d training and %d test samples",
                    len(train_series),
                    len(test_series),
                )

                # Add training data to consolidated df
                consolidated_df = add_to_consolidated_df(
                    consolidated_df, train_series, mt, resd, level, "Train"
                )

                # Add test data to consolidated df
                consolidated_df = add_to_consolidated_df(
                    consolidated_df, test_series, mt, resd, level, "Test"
                )

                # Fit models
                logger.info("Fitting models for %s - %s - %s", level, resd, mt)
                fit_snaive, fit_ets, fit_arima = fit_models(
                    train_series,
                    seasonal_naive_params=model_params.get(
                        "seasonal_naive_params", {}
                    ),
                    ets_params=model_params.get("ets_params", {}),
                    arima_params=model_params.get("arima_params", {}),
                )
                # Analyze residuals for each model
                logger.info(
                    "Analyzing residuals for %s - %s - %s", level, resd, mt
                )
                residual_diagnostics = calculate_residuals(
                    fit_snaive,
                    fit_ets,
                    fit_arima,
                    train_series,
                    residual_diagnostics,
                    mt,
                    resd,
                    level,
                    save_plots=save_plots,
                    plots_dir=plots_dir,
                )

                # Forecast test data
                logger.info(
                    "Creating forecasts for test period (%s - %s - %s)",
                    level,
                    resd,
                    mt,
                )
                snaive_forecast = fit_snaive.forecast(steps=len(test_series))
                ets_forecast = fit_ets.forecast(steps=len(test_series))
                arima_forecast = fit_arima.forecast(steps=len(test_series))

                for model_name, forecast in [
                    ("Seasonal Naive", snaive_forecast),
                    ("ETS", ets_forecast),
                    ("ARIMA", arima_forecast),
                ]:
                    consolidated_df = add_to_consolidated_df(
                        consolidated_df,
                        forecast,
                        mt,
                        resd,
                        level,
                        "Test",
                        model_name,
                    )

                # Forecast future (configurable steps ahead)
                logger.info(
                    "Creating future forecasts (%d steps ahead) for %s - %s - %s",
                    forecast_steps,
                    level,
                    resd,
                    mt,
                )
                future_snaive = fit_snaive.forecast(steps=forecast_steps)
                future_ets = fit_ets.forecast(steps=forecast_steps)
                future_arima = fit_arima.forecast(steps=forecast_steps)

                for model_name, forecast in [
                    ("Seasonal Naive", future_snaive),
                    ("ETS", future_ets),
                    ("ARIMA", future_arima),
                ]:
                    consolidated_df = add_to_consolidated_df(
                        consolidated_df,
                        forecast,
                        mt,
                        resd,
                        level,
                        "Forecast",
                        model_name,
                    )

                # Evaluate forecast accuracy on test set
                logger.info(
                    "Evaluating forecast accuracy for %s - %s - %s",
                    level,
                    resd,
                    mt,
                )
                forecast_evaluation = evaluate_forecasts(
                    test_series,
                    [snaive_forecast, ets_forecast, arima_forecast],
                    models=["Seasonal Naive", "ETS", "ARIMA"],
                )

                # Add metadata to evaluation results
                for _, row in forecast_evaluation.iterrows():
                    evaluation_results.append(
                        {
                            "Analysis_Type": mt,
                            "Residency": resd,
                            "Level": level,
                            "Model": row["Model"],
                            "MAE": row["MAE"],
                            "RMSE": row["RMSE"],
                            "MAPE": row["MAPE"],
                        }
                    )

    return consolidated_df, evaluation_results, residual_diagnostics
