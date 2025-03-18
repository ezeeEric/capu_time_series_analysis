# -*- coding: utf-8 -*-
"""
Data loading and preprocessing utilities for time series analysis.

This module provides functions to load, filter, and prepare enrollment data
for time series analysis.
"""

from typing import Tuple

import pandas as pd


def load_data(file_path: str, level: str, resd: str, mt: str) -> pd.DataFrame:
    """
    Load and filter enrollment data from CSV file.

    Args:
        file_path: Path to the CSV file containing enrollment data
        level: Academic level to filter (e.g., 'CapU', 'AS', 'BPS')
        resd: Residency status to filter (e.g., 'Domestic', 'International')
        mt: Metric type to extract (e.g., 'Headcount', 'CourseEnrolment')

    Returns:
        DataFrame containing filtered data with TermCode and selected metric
    """
    df_raw = pd.read_csv(file_path)
    df_subset = df_raw[
        (df_raw["Level"] == level) & (df_raw["Residency"] == resd)
    ]
    df_subset = (
        df_subset[["TermCode", mt]]
        .sort_values(by="TermCode", ascending=True)
        .reset_index(drop=True)
    )
    return df_subset


def prepare_time_series(
    df_subset: pd.DataFrame, mt: str
) -> Tuple[pd.Series, pd.Series]:
    """
    Convert filtered data into time series format and split into train/test sets.

    Args:
        df_subset: DataFrame containing TermCode and metric data
        mt: Metric column name to convert to time series

    Returns:
        Tuple containing (train_series, test_series) where:
            - train_series: First 36 periods of data as a time series
            - test_series: Remaining periods after the first 36
    """
    df_start_year = int(str(df_subset["TermCode"][0])[:4])
    df_start_term = int(str(df_subset["TermCode"][0])[4])
    ts_df = pd.Series(
        df_subset[mt].values,
        index=pd.date_range(
            start=pd.Timestamp(year=df_start_year, month=df_start_term, day=1),
            periods=len(df_subset),
            freq="4ME",  # 4-month frequency (trimester)
        ),
    )
    train = ts_df[:36]
    test = ts_df[36:]
    return train, test


def add_to_consolidated_df(
    consolidated_df: pd.DataFrame,
    series_data: pd.Series,
    analysis_type: str,
    residency: str,
    level: str,
    entry_type: str,
    model: str = "Actual",
) -> pd.DataFrame:
    """
    Add a time series to the consolidated dataframe.

    Args:
        consolidated_df: The dataframe to add data to
        series_data: Time series data to add
        analysis_type: Type of analysis (e.g., 'Headcount', 'CourseEnrolment')
        residency: Residency status ('Domestic' or 'International')
        level: Academic level (e.g., 'CapU', 'AS')
        entry_type: Type of entry ('Train', 'Test', or 'Forecast')
        model: Model name ('Actual', 'Seasonal Naive', 'ETS', or 'ARIMA')

    Returns:
        Updated consolidated dataframe
    """
    temp_df = pd.DataFrame(
        {
            "Analysis_Type": analysis_type,
            "Residency": residency,
            "Level": level,
            "Timestamp": series_data.index,
            "Model": model,
            "Entry_Type": entry_type,
            "Entry": series_data.values,
        }
    )

    return pd.concat([consolidated_df, temp_df], ignore_index=True)
