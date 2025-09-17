from pandas import DataFrame, Series, concat

from src.data.processors.encoding import encode_cyclic


def create_time_features(
    timestamps: Series,
    minute: bool = False,
    hour: bool = False,
    day: bool = False,
    day_of_week: bool = False,
    month: bool = False
) -> DataFrame:
    """
    Calculates cyclic time-features based on `cos` and `sin` functions.

    Parameters
    ---
    timestamps : Series
        Series containing datetime values
    minute : bool, default=False
        Whether to include minute-based features
    hour : bool, default=False
        Whether to include hour-based features
    day : bool, default=False
        Whether to include day-of-month features
    day_of_week : bool, default=False
        Whether to include day-of-week features
    month : bool, default=False
        Whether to include month features

    Returns
    ---
    DataFrame :
        DataFrame with cyclic time features encoded as sin/cos pairs

    Raises
    ---
    ValueError: If no time features are specified for encoding
    """
    time_features = DataFrame()

    # Validate that at least one time feature is requested
    if not any([minute, hour, day, day_of_week, month]):
        raise ValueError("No time features specified for encoding.")

    # Encode minute as cyclic feature using sin/cos transformation
    if minute:
        time_features = concat([
            time_features,
            encode_cyclic(
                timestamps.dt.minute.to_numpy(),
                col_name="minute",
                max_val=60
            )
        ], axis=1)

    # Encode hour as cyclic feature using sin/cos transformation
    if hour:
        time_features = concat([
            time_features,
            encode_cyclic(
                timestamps.dt.hour.to_numpy(),
                col_name="hour",
                max_val=24
            )
        ], axis=1)

    # Encode day as cyclic feature using sin/cos transformation
    if day:
        time_features = concat([
            time_features,
            encode_cyclic(
                timestamps.dt.day.to_numpy(),
                col_name="day",
                max_val=timestamps.dt.days_in_month.to_numpy()
            )
        ], axis=1)

    # Encode day of week as cyclic feature using sin/cos transformation
    if day_of_week:
        time_features = concat([
            time_features,
            encode_cyclic(
                timestamps.dt.day_of_week.to_numpy(),
                col_name="day_of_week",
                max_val=7
            )
        ], axis=1)

    # Encode month as cyclic feature using sin/cos transformation
    if month:
        time_features = concat([
            time_features,
            encode_cyclic(
                timestamps.dt.month.to_numpy(),
                col_name="month",
                max_val=12
            )
        ], axis=1)

    return time_features
