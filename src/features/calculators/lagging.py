import os
import sys

from pandas import DataFrame

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
sys.path.insert(0, project_root)

from src.config.schemas.features import LaggingConfig


def get_lagging_features(
    df: DataFrame,
    config: LaggingConfig
) -> DataFrame:
    """
    Create lagged versions of a time series.

    Parameters
    ---
    df : Dataframe
        Complete Dataframe with already created features
    config : LaggingConfig
        Configuration instance with column name and maximum number
        of lags to create (will create lags 1 to `config.period`)

    Returns
    ---
    DataFrame: DataFrame with lagged features

    Raises
    ---
    ValueError: If `config.period` is less than 1
    """
    # Validate input parameter
    if config.period < 1:
        raise ValueError(
            "Lagging period must be at least 1, "
            f"while {config.period} is obtained."
        )

    # Select the required feature by column name
    col = df[config.column]

    # Creating a dataframe for storing lags
    lagging_features = DataFrame()

    # Create lags from 1 to config.period
    for lag in range(1, config.period + 1):
        lagging_features[f"{col.name}_lag{lag}"] = col.shift(lag)
    return lagging_features