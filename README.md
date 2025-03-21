# Capilano University Time Series Analysis

This package provides tools for time series forecasting of enrollment data from Capilano University. It implements multiple forecasting models (Seasonal Naive, ETS, ARIMA) and provides utilities for evaluation, visualization, and comparison.

## Table of Contents

- [Capilano University Time Series Analysis](#capilano-university-time-series-analysis)
  - [Table of Contents](#table-of-contents)
  - [Technical Setup](#technical-setup)
    - [Installation](#installation)
    - [Project Structure](#project-structure)
  - [From R to Python: Adapting the Original Analysis](#from-r-to-python-adapting-the-original-analysis)
    - [Original Analysis Steps vs. Python Implementation](#original-analysis-steps-vs-python-implementation)
  - [Using the Package](#using-the-package)
    - [Basic Usage](#basic-usage)
    - [Customizing Analysis](#customizing-analysis)
  - [Modules Explained](#modules-explained)
    - [data_loader.py](#data_loaderpy)
    - [models.py](#modelspy)
    - [evaluation.py](#evaluationpy)
    - [visualization.py](#visualizationpy)
  - [Model Selection and Optimization](#model-selection-and-optimization)
  - [Scaling and Maintaining Models](#scaling-and-maintaining-models)
    - [Future Scaling with Small Error Rates](#future-scaling-with-small-error-rates)
    - [Retraining Models and Adding Data](#retraining-models-and-adding-data)
      - [When to Retrain](#when-to-retrain)
      - [When to Simply Add Data](#when-to-simply-add-data)
    - [Batch Processing](#batch-processing)
  - [Understanding Model Diagnostics](#understanding-model-diagnostics)
    - [Autocorrelation Analysis](#autocorrelation-analysis)
    - [Residual Analysis](#residual-analysis)
  - [Accessing Results](#accessing-results)
  - [Future Improvements](#future-improvements)
  - [Credits](#credits)

## Technical Setup

Python 3.11 is required to run this package. The following instructions assume you have Python installed on your system.

### Installation

```bash
# Clone the repository
git clone https://github.com/ezeeEric/capu_time_series_analysis
cd capu_time_series_analysis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt
python -m pip install -e .

# Install pre-commit hooks
pre-commit install

# Add project to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$PWD
```

### Project Structure

```
capu_time_series_analysis/
├── configs/                   # Configuration files
│   └── default.yaml           # Default configuration
├── capu_time_series_analysis/ # Core package modules
│   ├── data_loader.py         # Data loading and preprocessing utilities
│   ├── evaluation.py          # Model evaluation and residual analysis
│   ├── models.py              # Time series forecasting models
│   ├── visualization.py       # Visualization and logging utilities
│   └── plotting.py            # Specialized plotting functions for time series data
├── data/                      # Data directory
│   ├── input/                 # Input data files
│   └── jiaqi_original/        # Original R script and data
├── notebooks/                 # Jupyter notebooks
│   ├── exploratory_analysis.ipynb  # Initial data exploration
│   ├── model_comparison.ipynb     # Detailed model comparison and analysis
│   └── forecast_visualization.ipynb  # Interactive forecast visualizations
├── scripts/                   # Execution scripts
│   └── run_timeseries_analysis.py  # Main execution script
└── output/                    # Output directory
    ├── plots/                 # Generated static plots
    ├── models/                # Saved model objects
    └── forecasts/             # Forecast output data
```

## From R to Python: Adapting the Original Analysis

The original analysis, developed by Jiaqi Li, was implemented in R using the `fpp2` package. This Python implementation preserves the core analysis workflow while adding:

1. **Modular, maintainable code structure**: The monolithic R script has been divided into specialized Python modules.
2. **Configuration-driven execution**: Using [Hydra](https://hydra.cc/docs/intro/) for configuration management instead of hardcoded parameters.
3. **Enhanced visualization**: Results are logged to [Weights & Biases](https://docs.wandb.ai/) for interactive visualization and experiment tracking.
4. **Improved scalability**: The Python implementation can handle multiple analysis combinations efficiently.

### Original Analysis Steps vs. Python Implementation

The original R analysis follows these steps:

01. Install packages and set variables
02. Import and prepare data
03. Visualize data
04. Fit models to training set
05. Compare test set accuracy
06. Check training set residuals
07. Select the best model
08. Forecast future using the best model
09. (Optional) Compare forecasting results
10. Export results to Excel

Our Python implementation follows a similar workflow but with a more structured approach:

1. **Configuration Management**: Parameters defined in YAML files
2. **Data Loading**: `data_loader.py` handles import and preparation
3. **Model Definition**: `models.py` contains implementations of forecasting algorithms
4. **Model Evaluation**: `evaluation.py` provides metrics and residual analysis
5. **Visualization**: `visualization.py` logs results to W&B
6. **Orchestration**: `run_timeseries_analysis.py` coordinates the entire process

## Using the Package

### Basic Usage

Run the analysis using the default configuration:

```bash
python scripts/run_timeseries_analysis.py
```

### Customizing Analysis

Modify the default configuration or override parameters:

```bash
# Override specific parameters
python scripts/run_timeseries_analysis.py metrics=[Headcount,CourseEnrolment] levels=[CapU]

# Use a different config file
python scripts/run_timeseries_analysis.py --config-name=custom_config
```

## Modules Explained

### data_loader.py

Provides functions to:

- Load and filter enrollment data from CSV files
- Convert data into time series format
- Split data into training and test sets
- Consolidate results into a standardized format

### models.py

Implements time series forecasting models:

- `SeasonalNaive`: Uses the last observed value from the same season
- Wrappers for `ExponentialSmoothing` (ETS) and `SARIMAX` (ARIMA) from statsmodels
- Model fitting and forecasting utilities

### evaluation.py

Provides tools to:

- Calculate forecast accuracy metrics (MAE, RMSE, MAPE)
- Analyze model residuals
- Generate diagnostic plots for model validation
- Test for white noise characteristics

### visualization.py

Handles:

- Logging results to Weights & Biases
- Creating filtered views of data for easier exploration
- Converting diagnostic plots to format suitable for W&B

## Model Selection and Optimization

The framework implements several strategies for optimizing and selecting the best forecasting models:

- **Comparative Metrics**: Models are compared using multiple accuracy metrics (MAE, RMSE, MAPE) to identify the best performer for each data series.
- **Model Optimization**: Automated parameter tuning is available for ARIMA models through `auto.arima` equivalents in Python.
- **Large-scale Modeling**: Weights & Biases (W&B) integration supports tracking performance across numerous model combinations and datasets, especially helpful when scaling to program-level forecasting.
- **Validation Against Original Results**: The implementation includes verification against Jiaqi's original R-based analysis to ensure consistency in forecasting performance.

## Scaling and Maintaining Models

### Future Scaling with Small Error Rates

To improve forecast accuracy when scaling to more detailed levels:

- **External Variables**: Support for incorporating additional contextual variables:
  - Demographic trends
  - Economic indicators
  - Policy changes
- **Hybrid Models**: Implementation of ensemble approaches combining ARIMA with deep learning models like LSTM
- **Periodic Retraining**: Framework for regularly updating models with fresh data

### Retraining Models and Adding Data

#### When to Retrain

- **New Data Available**: Keep models up-to-date with latest observations
- **Significant Trend Shifts**: After policy changes or economic events that alter enrollment patterns
- **Scheduled Retraining**: Regular quarterly or annual recalibration
- **Performance Decline**: When accuracy metrics worsen beyond acceptable thresholds

#### When to Simply Add Data

- **Minor or Seasonal Variations**: When new data follows expected seasonal patterns
- **Short-Term Updates**: For incremental updates when underlying trends remain stable

### Batch Processing

The framework supports batch processing multiple model configurations:

- **Parameterizable Components**:
  - Input data files
  - Training and testing date ranges
  - Model parameters (with support for automatic ARIMA adjustment)
  - Seasonal frequency
  - Forecast horizon
  - Output file paths

## Understanding Model Diagnostics

### Autocorrelation Analysis

- **Definition**: Measures correlation of a time series with its own past values
- **Purpose**: Helps identify underlying patterns like trends and seasonality
- **Diagnostic Use**: Used to verify if a model has captured all significant patterns in the data

### Residual Analysis

- **Definition**: Differences between observed values and model predictions
- **Purpose**: Assesses model fit quality
- **Diagnostic Use**: Good models produce residuals that:
  - Are uncorrelated (resembling white noise)
  - Have a mean of zero
  - Show no systematic patterns

The `evaluation.py` module provides comprehensive tools for both autocorrelation and residual analysis to ensure model validity.

## Accessing Results

Results can be visualized and explored in the Weights & Biases dashboard:

1. Navigate to https://wandb.ai/ezeeeric/time_series_forecasting
2. Select the most recent run to view:
   - Time series data and forecasts
   - Evaluation metrics
   - Residual diagnostics
   - Comparison between different models

## Future Improvements

- Add more forecasting models (Prophet, LSTM, etc.)
- Implement automated model selection
- Add export to Excel functionality
- Support for more complex seasonality patterns

## Credits

- Original Analysis: Jiaqi Li (jiaqili@capilanou.ca)
- Python Implementation: Eric Drechsler (dr.eric.drechsler@gmail.com)
