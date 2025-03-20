# -*- coding: utf-8 -*-
"""
Plotting utilities for time series analysis.

This module provides functions for creating plots to analyze time series data.
"""

import base64
import io
import os
from typing import Tuple

import matplotlib.dates as mdates

# Visualization of Results
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
from statsmodels.graphics.tsaplots import plot_acf

# Set plotting style
plt.style.use("ggplot")
sns.set_theme(style="whitegrid")


def plot_residuals_analysis(
    residuals: pd.Series, model_name: str
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Create residual analysis plots.

    Args:
        residuals: Series containing model residuals
        model_name: Name of the model for plot titles

    Returns:
        Tuple of (residuals_figure, acf_figure)
    """
    # Create figure for residuals time plot
    fig_residuals = plt.figure(figsize=(10, 6))
    plt.plot(residuals.index, residuals, "o-")
    plt.axhline(y=0, color="r", linestyle="-")
    plt.title(f"Residuals Over Time - {model_name}")
    plt.ylabel("Residuals")
    plt.xlabel("Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Create separate figure for ACF plot
    fig_acf = plt.figure(figsize=(10, 6))

    # Only attempt ACF plot if we have enough data
    if len(residuals) > 5:
        lags = min(20, len(residuals) - 1)
        # Ensure we use the correct array format for plot_acf
        plot_acf(
            residuals.values.squeeze(), lags=lags, alpha=0.05, ax=plt.gca()
        )
        plt.title(f"Autocorrelation Function of Residuals - {model_name}")
    else:
        plt.text(
            0.5,
            0.5,
            "Insufficient data for ACF plot",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.title(f"ACF of Residuals - {model_name} (not enough data)")

    plt.tight_layout()

    return fig_residuals, fig_acf


def plot_actual_vs_fitted(
    actual_values: pd.Series, fitted_values: pd.Series, model_name: str
) -> plt.Figure:
    """
    Create plot comparing actual values with fitted values from a model.

    Args:
        actual_values: Series containing actual values
        fitted_values: Series containing fitted/predicted values
        model_name: Name of the model for plot titles

    Returns:
        Figure object containing the plot
    """
    fig = plt.figure(figsize=(10, 6))
    plt.plot(actual_values.index, actual_values, "b-", label="Actual")
    plt.plot(fitted_values.index, fitted_values, "r--", label="Fitted")
    plt.title(f"Actual vs Fitted Values - {model_name}")
    plt.ylabel("Value")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def fig_to_base64(fig: plt.Figure) -> str:
    """
    Convert a matplotlib figure to a base64 encoded string.

    Args:
        fig: Matplotlib figure object

    Returns:
        Base64 encoded string of the figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def save_figure(fig: plt.Figure, filepath: str) -> None:
    """
    Save a matplotlib figure to disk.

    Args:
        fig: Matplotlib figure object
        filepath: Path where to save the figure
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath)
    plt.close(fig)


# Visualisations for Notebooks


# 1. Time Series Visualization by combination
def plot_time_series_combination(
    consolidated_df, level, residency, metric, figsize=(14, 8)
):
    """Plot actual values, test period, and forecasts for a specific combination"""

    # Filter data for the specified combination
    combo_data = consolidated_df[
        (consolidated_df["Level"] == level)
        & (consolidated_df["Residency"] == residency)
        & (consolidated_df["Analysis_Type"] == metric)
    ]

    if combo_data.empty:
        print(f"No data available for {level} - {residency} - {metric}")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Plot Actual Data (Training)
    train_data = combo_data[combo_data["Entry_Type"] == "Train"]
    ax.plot(
        pd.to_datetime(train_data["Timestamp"]),
        train_data["Entry"],
        "b-",
        linewidth=2,
        label="Actual (Training)",
    )

    # Plot Actual Data (Test)
    test_data = combo_data[
        (combo_data["Entry_Type"] == "Test")
        & (combo_data["Model"] == "Actual")
    ]
    if not test_data.empty:
        ax.plot(
            pd.to_datetime(test_data["Timestamp"]),
            test_data["Entry"],
            "g-",
            linewidth=2,
            label="Actual (Test)",
        )

    # Plot Model Forecasts
    for model_name in ["Seasonal Naive", "ETS", "ARIMA"]:
        # Test period forecasts
        test_forecast = combo_data[
            (combo_data["Entry_Type"] == "Test")
            & (combo_data["Model"] == model_name)
        ]
        if not test_forecast.empty:
            ax.plot(
                pd.to_datetime(test_forecast["Timestamp"]),
                test_forecast["Entry"],
                "--",
                label=f"{model_name} (Test)",
            )

        # Future forecasts
        # TODO could be added in the future.
        # future_forecast = combo_data[
        #     (combo_data['Entry_Type'] == 'Forecast') &
        #     (combo_data['Model'] == model_name)
        # ]
        # if not future_forecast.empty:
        #     ax.plot(pd.to_datetime(future_forecast['Timestamp']), future_forecast['Entry'],
        #             ':', label=f'{model_name} (Forecast)')

    # Formatting
    ax.set_title(
        f"Time Series Forecast: {level} - {residency} - {metric}", fontsize=14
    )
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig


# 2. Model Evaluation Metrics Visualization
def plot_evaluation_metrics(
    evaluation_df,
    metric="RMSE",
    figsize=(20, 14),
    cols_per_row=4,
    independent_scaling=True,
):
    """
    Plot evaluation metrics in a grid layout with custom configuration

    Args:
        evaluation_df: DataFrame with evaluation metrics
        metric: Which metric to visualize ('MAE', 'RMSE', or 'MAPE')
        figsize: Size of the figure
        cols_per_row: Number of columns per row (default: 4)
        independent_scaling: Whether to use independent y-axes scaling (default: True)
    """
    if metric not in ["MAE", "RMSE", "MAPE"]:
        raise ValueError("Metric must be one of 'MAE', 'RMSE', or 'MAPE'")

    # Get unique values for each dimension
    levels = sorted(evaluation_df["Level"].unique())
    residencies = sorted(evaluation_df["Residency"].unique())
    analysis_types = sorted(evaluation_df["Analysis_Type"].unique())

    # Create a list of all combinations
    combinations = []
    for level in levels:
        for residency in residencies:
            combinations.append((level, residency))

    # Calculate grid dimensions
    n_combos = len(combinations)
    n_cols = cols_per_row
    n_rows = (n_combos + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with subplots
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, sharey=not independent_scaling
    )

    # Handle case with single row or column
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    # Flatten axes if needed
    if n_rows > 1 or n_cols > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes[0][0]]

    # Loop through each combination
    for i, (level, residency) in enumerate(combinations):
        if i < len(axes_flat):
            ax = axes_flat[i]

            # Filter data for this Level/Residency combination
            combo_data = evaluation_df[
                (evaluation_df["Level"] == level)
                & (evaluation_df["Residency"] == residency)
            ]

            if not combo_data.empty:
                # Create grouped bar chart for this subplot
                sns.barplot(
                    data=combo_data,
                    x="Model",
                    y=metric,
                    hue="Analysis_Type",
                    ax=ax,
                )

                # Set subplot title and format
                ax.set_title(f"{level} - {residency}", fontsize=12)
                ax.set_xlabel("")

                # If using independent scaling, don't set consistent y-limit
                if not independent_scaling:
                    max_value = evaluation_df[metric].max() * 1.1
                    ax.set_ylim(0, max_value)

                # Format x-ticks
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

                # Only show legend for the first subplot
                if i > 0:
                    ax.get_legend().remove()
            else:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])

    # Hide unused subplots
    for j in range(len(combinations), len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Add a single legend at the top of the figure
    if len(axes_flat) > 0 and hasattr(
        axes_flat[0], "get_legend_handles_labels"
    ):
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(analysis_types),
            bbox_to_anchor=(0.5, 0.98),
            title="Analysis Type",
        )

    plt.suptitle(f"Model Comparison by {metric}", fontsize=16, y=0.99)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the title and legend

    return fig


# 3. Residual Diagnostics Visualization
def display_residual_plots(residual_df):
    """
    Display residual plots from the base64-encoded images in the residual diagnostics
    """
    # Select a specific combination to display residual plots
    combinations = (
        residual_df[["Level", "Residency", "Analysis_Type"]]
        .drop_duplicates()
        .values
    )

    if len(combinations) == 0:
        print("No residual diagnostics available")
        return

    # Function to convert base64 to HTML img tag
    def base64_to_img(base64_str, width=800):
        if not base64_str or pd.isna(base64_str):
            return "<p>No plot available</p>"
        return (
            f'<img src="data:image/png;base64,{base64_str}" width="{width}px">'
        )

    # Display for the first combination as an example
    first_combo = combinations[0]
    level, residency, analysis_type = first_combo

    print(f"## Residual Diagnostics: {level} - {residency} - {analysis_type}")

    # Filter for this combination
    combo_residuals = residual_df[
        (residual_df["Level"] == level)
        & (residual_df["Residency"] == residency)
        & (residual_df["Analysis_Type"] == analysis_type)
    ]

    # Display for each model
    for _, row in combo_residuals.iterrows():
        model = row["Model"]
        print(f"\n### {model} Model")

        # Statistical information
        stats_info = (
            f"Mean Residual: {row['Mean_Residual']:.4f}\n"
            f"Std Residual: {row['Std_Residual']:.4f}\n"
            f"Ljung-Box p-value: {row['Ljung_Box_pvalue']:.4f}\n"
            f"White Noise Test: {'Passed' if row['White_Noise'] else 'Failed'}"
        )
        print(stats_info)

        # Display the actual vs fitted plot if available
        if "Actual_Fitted_Plot" in row and not pd.isna(
            row["Actual_Fitted_Plot"]
        ):
            print("\nActual vs Fitted Values:")
            display(HTML(base64_to_img(row["Actual_Fitted_Plot"])))

        # Display the residuals plot if available
        if "Residuals_Plot" in row and not pd.isna(row["Residuals_Plot"]):
            print("\nResiduals Plot:")
            display(HTML(base64_to_img(row["Residuals_Plot"])))
