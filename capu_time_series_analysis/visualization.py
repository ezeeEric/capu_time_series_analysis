# -*- coding: utf-8 -*-
"""
Visualization and logging utilities for time series forecasting.

This module provides functions to log results to Weights & Biases and
create visualizations of time series data and forecasts.
"""

import base64
import logging
import os
from typing import List

import pandas as pd

import wandb

logger = logging.getLogger(__name__)


def log_results_to_wandb(
    consolidated_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    metrics: List[str],
    levels: List[str],
    residencies: List[str],
    project_name: str = "time_series_forecasting",
    entity_name: str = "ezeeeric",
):
    """
    Log results to Weights & Biases for visualization.

    Args:
        consolidated_df: DataFrame containing all time series data
        evaluation_df: DataFrame containing model evaluation metrics
        residual_df: DataFrame containing residual diagnostics
        metrics: List of analysis types/metrics used
        levels: List of academic levels
        residencies: List of residency statuses
        project_name: Name of the W&B project
        entity_name: Name of the W&B entity/username
    """
    # Initialize wandb
    logger.info(
        f"Initializing wandb project {project_name} with entity {entity_name}"
    )
    wandb.init(project=project_name, entity=entity_name)

    try:
        # Log consolidated data
        logger.info(
            f"Logging consolidated time series data ({len(consolidated_df)} rows)"
        )
        wandb.log(
            {
                "consolidated_time_series": wandb.Table(
                    dataframe=consolidated_df
                )
            }
        )

        # Log evaluation metrics
        logger.info("Logging forecast evaluation metrics")
        wandb.log(
            {
                "forecast_evaluation_metrics": wandb.Table(
                    dataframe=evaluation_df
                )
            }
        )

        # Log residual diagnostics (excluding image columns)
        residual_metrics_df = residual_df.drop(
            columns=["Residuals_Plot", "Actual_Fitted_Plot"]
        )
        logger.info("Logging residual diagnostics metrics")
        wandb.log(
            {
                "residual_diagnostics": wandb.Table(
                    dataframe=residual_metrics_df
                )
            }
        )

        # Log residual plots as images
        logger.info("Logging residual and fitted value plots")
        for i, row in residual_df.iterrows():
            if row["Residuals_Plot"] is not None:
                # Decode the base64 string to bytes
                image_bytes = base64.b64decode(row["Residuals_Plot"])
                # Create a temporary file to store the image
                with open("temp_residuals.png", "wb") as f:
                    f.write(image_bytes)
                # Log the image file
                wandb.log(
                    {
                        f"{row['Analysis_Type']}_{row['Level']}_{row['Residency']}_{row['Model']}_residuals": wandb.Image(
                            "temp_residuals.png"
                        )
                    }
                )

            if row["Actual_Fitted_Plot"] is not None:
                # Decode the base64 string to bytes
                image_bytes = base64.b64decode(row["Actual_Fitted_Plot"])
                # Create a temporary file to store the image
                with open("temp_fitted.png", "wb") as f:
                    f.write(image_bytes)
                # Log the image file
                wandb.log(
                    {
                        f"{row['Analysis_Type']}_{row['Level']}_{row['Residency']}_{row['Model']}_fitted": wandb.Image(
                            "temp_fitted.png"
                        )
                    }
                )

        # Create and log filtered views for easier visualization
        logger.info("Creating and logging filtered views of the data")
        for mt in metrics:
            # Log training and testing data
            for entry_type in ["Train", "Test", "Forecast"]:
                filtered_df = consolidated_df[
                    (consolidated_df["Analysis_Type"] == mt)
                    & (consolidated_df["Entry_Type"] == entry_type)
                ]
                if not filtered_df.empty:
                    logger.debug(
                        f"Logging {mt}_{entry_type.lower()}_data with {len(filtered_df)} rows"
                    )
                    wandb.log(
                        {
                            f"{mt}_{entry_type.lower()}_data": wandb.Table(
                                dataframe=filtered_df
                            )
                        }
                    )

            # Log evaluation results by metric
            metric_eval = evaluation_df[evaluation_df["Analysis_Type"] == mt]
            if not metric_eval.empty:
                wandb.log(
                    {f"{mt}_evaluation": wandb.Table(dataframe=metric_eval)}
                )

            # Log residual diagnostics by metric
            metric_resid = residual_metrics_df[
                residual_metrics_df["Analysis_Type"] == mt
            ]
            if not metric_resid.empty:
                wandb.log(
                    {
                        f"{mt}_residual_diagnostics": wandb.Table(
                            dataframe=metric_resid
                        )
                    }
                )

            # Log comparison tables for each level and residency
            for level in levels:
                for resd in residencies:
                    comparison_df = consolidated_df[
                        (consolidated_df["Analysis_Type"] == mt)
                        & (consolidated_df["Level"] == level)
                        & (consolidated_df["Residency"] == resd)
                    ]
                    if not comparison_df.empty:
                        logger.debug(
                            f"Logging comparison for {mt}_{level}_{resd} with {len(comparison_df)} rows"
                        )
                        wandb.log(
                            {
                                f"{mt}_{level}_{resd}_comparison": wandb.Table(
                                    dataframe=comparison_df
                                )
                            }
                        )
    finally:
        # Clean up temporary files if they exist
        for temp_file in ["temp_residuals.png", "temp_fitted.png"]:
            if os.path.exists(temp_file):
                logger.debug(f"Removing temporary file: {temp_file}")
                os.remove(temp_file)

        # Always finish wandb tracking, even if an exception occurs
        logger.info("Finishing wandb logging")
        wandb.finish()
