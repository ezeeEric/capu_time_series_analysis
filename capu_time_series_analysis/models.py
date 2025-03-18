# -*- coding: utf-8 -*-
"""
Time series forecasting models.

This module provides custom time series forecasting models and wrappers
for other statistical models.
"""

from typing import Tuple

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SeasonalNaive:
    """
    Seasonal Naive forecasting model.

    This model forecasts future values based on the last observed value from the same
    season/period, similar to R's snaive() function.
    """

    def __init__(self, seasonal_periods: int = 3):
        """
        Initialize the Seasonal Naive model.

        Args:
            seasonal_periods: Number of periods in a seasonal cycle (default: 3 for trimester data)
        """
        self.seasonal_periods = seasonal_periods
        self.data = None
        self.fitted_values = None
        self.residuals = None

    def fit(self, data: pd.Series) -> "SeasonalNaive":
        """
        Fit the Seasonal Naive model to the provided time series data.

        Args:
            data: Time series data as a pandas Series

        Returns:
            Self for method chaining
        """
        self.data = data

        # Generate fitted values (shifted by seasonal period)
        self.fitted_values = data.shift(self.seasonal_periods)

        # Calculate residuals
        self.residuals = (
            data[self.seasonal_periods :]
            - self.fitted_values[self.seasonal_periods :]
        )

        return self

    def forecast(self, steps: int) -> pd.Series:
        """
        Generate forecasts for future periods.

        Args:
            steps: Number of periods to forecast

        Returns:
            Series containing forecast values
        """
        if self.data is None:
            raise ValueError("Model must be fit before forecasting")

        # Get last values for each season
        seasons = {}
        n = len(self.data)

        # Find the most recent value for each season position
        for i in range(self.seasonal_periods):
            # Find the most recent observation for this season
            pos = n - 1 - ((n - 1 - i) % self.seasonal_periods)
            seasons[i] = self.data.iloc[pos]

        # Generate forecasts
        forecasts = []
        for i in range(steps):
            season_pos = i % self.seasonal_periods
            forecasts.append(seasons[season_pos])

        # Create forecasts series with appropriate index
        if isinstance(self.data.index, pd.DatetimeIndex):
            # Continue the date pattern
            last_date = self.data.index[-1]
            if (
                hasattr(self.data.index, "freq")
                and self.data.index.freq is not None
            ):
                freq = self.data.index.freq
                forecast_index = pd.date_range(
                    start=last_date + freq, periods=steps, freq=freq
                )
            else:
                # If frequency is not defined, attempt to infer it
                inferred_freq = pd.infer_freq(self.data.index)
                forecast_index = pd.date_range(
                    start=last_date, periods=steps + 1, freq=inferred_freq
                )[1:]
        else:
            # Use integer index continuation
            forecast_index = range(len(self.data), len(self.data) + steps)

        return pd.Series(forecasts, index=forecast_index)


def fit_models(
    train_series,
    seasonal_naive_params=None,
    ets_params=None,
    arima_params=None,
):
    """
    Fit time series forecasting models to the training data.

    Args:
        train_series: Time series training data
        seasonal_naive_params: Parameters for Seasonal Naive model
        ets_params: Parameters for ETS model
        arima_params: Parameters for ARIMA model

    Returns:
        tuple: Fitted models (seasonal naive, ETS, ARIMA)
    """
    # Apply default parameters if none provided
    if seasonal_naive_params is None:
        seasonal_naive_params = {}
    if ets_params is None:
        ets_params = {}
    if arima_params is None:
        arima_params = {}

    # Get the seasonal period (default: 3)
    seasonal_period = seasonal_naive_params.get("seasonal_periods", 3)

    # Fit Seasonal Naive model
    fit_snaive = SeasonalNaive(seasonal_periods=seasonal_period)
    fit_snaive.fit(train_series)

    # Fit ETS model with configurable parameters
    fit_ets = ExponentialSmoothing(
        train_series,
        seasonal=ets_params.get("seasonal", "add"),
        damped_trend=ets_params.get("damped_trend", True),
        **{
            k: v
            for k, v in ets_params.items()
            if k not in ["seasonal", "damped_trend"]
        }
    ).fit()

    # Fit ARIMA model with configurable parameters
    fit_arima = SARIMAX(
        train_series,
        order=arima_params.get("order", (1, 1, 1)),
        seasonal_order=arima_params.get("seasonal_order", (1, 1, 1, 3)),
    ).fit(disp=False)

    return fit_snaive, fit_ets, fit_arima
