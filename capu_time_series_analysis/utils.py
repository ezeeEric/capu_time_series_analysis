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
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process time series data for different combinations of metrics, residencies, and levels.

    Args:
        input_file (str): Path to the input CSV file
        metrics (list): List of metrics to analyze
        residencies (list): List of residency statuses
        levels (list): List of academic levels
        forecast_steps (int): Number of steps to forecast into the future
        model_params (dict): Parameters for the forecasting models

    Returns:
        tuple: (consolidated_df, evaluation_results, residual_diagnostics)
    """
    if model_params is None:
        model_params = {}

    total_combinations = len(levels) * len(residencies) * len(metrics)
    logger.info(
        f"Processing {total_combinations} combinations of levels, residencies, and metrics"
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
                    f"Processing {level} - {resd} - {mt} ({current_combination}/{total_combinations})"
                )

                # Load data
                df_subset = load_data(input_file, level, resd, mt)
                if df_subset.empty:
                    logger.warning(
                        f"No data for {level} - {resd} - {mt}, skipping..."
                    )
                    continue

                # Prepare time series data
                train_series, test_series = prepare_time_series(df_subset, mt)
                logger.debug(
                    f"Split data into {len(train_series)} training and {len(test_series)} test samples"
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
                logger.info(f"Fitting models for {level} - {resd} - {mt}")
                fit_snaive, fit_ets, fit_arima = fit_models(
                    train_series,
                    seasonal_naive_params=model_params.get(
                        "seasonal_naive_params", {}
                    ),
                    ets_params=model_params.get("ets_params", {}),
                    arima_params=model_params.get("arima_params", {}),
                )

                # Analyze residuals for each model
                logger.info(f"Analyzing residuals for {level} - {resd} - {mt}")
                residual_diagnostics = calculate_residuals(
                    fit_snaive,
                    fit_ets,
                    fit_arima,
                    train_series,
                    residual_diagnostics,
                    mt,
                    resd,
                    level,
                )

                # Forecast test data
                logger.info(
                    f"Creating forecasts for test period ({level} - {resd} - {mt})"
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
                    f"Creating future forecasts ({forecast_steps} steps ahead) for {level} - {resd} - {mt}"
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
                    f"Evaluating forecast accuracy for {level} - {resd} - {mt}"
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
