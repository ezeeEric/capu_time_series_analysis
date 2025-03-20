# -*- coding: utf-8 -*-
"""
Plotting utilities for time series analysis.

This module provides functions for creating plots to analyze time series data.
"""

import base64
import io
import os
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf


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
