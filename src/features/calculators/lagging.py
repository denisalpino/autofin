from pandas import DataFrame, Series


def get_lagging_features(
    ser: Series,
    max_lag: int
) -> DataFrame:
    """
        Create lagged versions of a time series.

        Parameters
        ---
        ser: Input time series
        max_lag: Maximum number of lags to create (will create lags 1 to max_lag)

        Returns
        ---
        DataFrame: DataFrame with lagged features

        Raises
        ---
        ValueError: If max_lag is less than 1
    """
    # Validate input parameter
    if max_lag < 1:
        raise ValueError("Parameter `max_lag` must be at least 1.")

    lagging_features = DataFrame()

    # Create lags from 1 to max_lag
    for lag in range(1, max_lag + 1):
        lagging_features[f"{ser.name}_lag{lag}"] = ser.shift(lag)
    return lagging_features