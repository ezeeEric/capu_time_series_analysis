# -*- coding: utf-8 -*-
"""
Time Series Forecasting Script for Capilano University Enrollment Data.

This script processes enrollment data from Capilano University, converts it into time series format,
and applies various forecasting models (Seasonal Naive, ETS, ARIMA). The results are logged to
Weights & Biases for visualization and comparison.

Original Work: Jiaqi Li (jiaqili@capilanou.ca)
Author: Eric Drechsler (dr.eric.drechsler@gmail.com)
version: 250301
"""
import logging
import os

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Import utility modules
from capu_time_series_analysis.data_loader import (
    add_to_consolidated_df,
    load_data,
    prepare_time_series,
)
from capu_time_series_analysis.evaluation import analyze_residuals, evaluate_forecasts
from capu_time_series_analysis.models import fit_models
from capu_time_series_analysis.visualization import log_results_to_wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


def calculate_residuals(
    fit_snaive,
    fit_ets,
    fit_arima,
    train_series,
    residual_diagnostics,
    mt,
    resd,
    level,
):
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


def process_timeseries(
    input_file,
    metrics,
    residencies,
    levels,
    forecast_steps=9,
    model_params=None,
):
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
    evaluation_results = []

    # Dictionary to store residual diagnostics
    residual_diagnostics = []

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


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main execution function for the time series processing pipeline.

    Processes enrollment data across different combinations of academic levels,
    residency statuses, and metrics. Converts data to time series format and logs
    results to Weights & Biases for visualization and analysis.

    Args:
        cfg (DictConfig): Hydra configuration
    """
    logger.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")

    input_file = cfg.input_file
    plot_dir = cfg.plot_dir
    levels = cfg.levels
    residencies = cfg.residencies
    metrics = cfg.metrics
    forecast_steps = cfg.forecast_steps
    model_params = {
        "seasonal_naive_params": cfg.models.seasonal_naive_params,
        "ets_params": cfg.models.ets_params,
        "arima_params": cfg.models.arima_params,
    }

    logger.info(
        f"Starting time series forecasting with data from {input_file}"
    )

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        logger.info(f"Created plot directory: {plot_dir}")

    # Process all combinations and build consolidated dataframe
    consolidated_df, evaluation_results, residual_diagnostics = (
        process_timeseries(
            input_file,
            metrics,
            residencies,
            levels,
            forecast_steps=forecast_steps,
            model_params=model_params,
        )
    )

    # Create evaluation dataframe
    evaluation_df = pd.DataFrame(evaluation_results)
    logger.info(
        f"Created evaluation dataframe with {len(evaluation_df)} entries"
    )

    # Create residual diagnostics dataframe
    residual_df = pd.DataFrame(residual_diagnostics)
    logger.info(
        f"Created residual diagnostics dataframe with {len(residual_df)} entries"
    )

    # Log results to wandb
    logger.info("Logging results to Weights & Biases")
    log_results_to_wandb(
        consolidated_df,
        evaluation_df,
        residual_df,
        metrics,
        levels,
        residencies,
    )
    logger.info("All combinations processed successfully")


if __name__ == "__main__":
    main()
