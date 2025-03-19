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
import pyrootutils
from omegaconf import DictConfig, OmegaConf

from capu_time_series_analysis.utils import process_timeseries

# Import utility modules
from capu_time_series_analysis.visualization import log_results_to_wandb

ROOT_PATH = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project_root",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


@hydra.main(config_path=f"{ROOT_PATH}/configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main execution function for the time series processing pipeline.

    Processes enrollment data across different combinations of academic levels,
    residency statuses, and metrics. Converts data to time series format and logs
    results to Weights & Biases for visualization and analysis.

    Args:
        cfg (DictConfig): Hydra configuration
    """
    logger.info("Configuration: \n%s", OmegaConf.to_yaml(cfg))

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
        "Starting time series forecasting with data from %s", input_file
    )

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        logger.info("Created plot directory: %s", plot_dir)

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
        "Created evaluation dataframe with %d entries", len(evaluation_df)
    )

    # Create residual diagnostics dataframe
    residual_df = pd.DataFrame(residual_diagnostics)
    logger.info(
        "Created residual diagnostics dataframe with %d entries",
        len(residual_df),
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
    # pylint: disable=no-value-for-parameter
    main()
