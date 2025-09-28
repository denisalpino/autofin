from typing import Dict, Optional, Sequence, Union, overload

from pandas import DataFrame, Series, Timedelta

from ...config.schemas.splitting import SplitIndices, SplitResult


def _parse_interval(interval_str: str) -> Timedelta:
    """
    Parse time interval string into pandas Timedelta object.

    Parameters
    ----------
    interval_str : str
        Time interval string. Supported units:
        - 'm' - minutes (e.g., '15m' for 15 minutes)
        - 'h' - hours (e.g., '12h' for 12 hours)
        - 'd' - days (e.g., '7d' for 7 days)
        - 'M' - months (e.g., '1M' for 1 month)

    Returns
    -------
    Timedelta
        Parsed time interval.

    Raises
    ------
    ValueError
        If unit is not supported.
    """
    n, unit = int(interval_str[:-1]), interval_str[-1]
    if unit == 'm': return Timedelta(minutes=n)  # pyright: ignore[reportReturnType]
    if unit == 'h': return Timedelta(hours=n)    # pyright: ignore[reportReturnType]
    if unit == 'd': return Timedelta(days=n)     # pyright: ignore[reportReturnType]
    if unit == 'M': return Timedelta(days=30*n)  # pyright: ignore[reportReturnType] # Approximation
    raise ValueError("Unsupported unit. Use 'm', 'h', 'd', or 'M'")


@overload
def train_val_test_split(
    features: DataFrame,
    split: Sequence[int],
    test_interval: None = None,
    val_interval: None = None,
    groups: Optional[Series] = None,
    timestamps: Optional[Series] = None
) -> Dict[str, SplitResult]:
    ...


@overload
def train_val_test_split(
    features: DataFrame,
    split: None = None,
    test_interval: str = ...,
    val_interval: Optional[str] = None,
    groups: Optional[Series] = None,
    timestamps: Optional[Series] = None
) -> Dict[str, SplitResult]:
    ...


def train_val_test_split(
    features: DataFrame,
    split: Optional[Sequence[int]] = None,
    test_interval: Optional[str] = None,
    val_interval: Optional[str] = None,
    groups: Optional[Series] = None,
    timestamps: Optional[Series] = None
) -> Dict[str, SplitResult]:
    """
    Splits data into training, validation and test sets with group support.

    Supports two splitting modes:
    1. Percentage-based: using `split` parameter with three integers
    2. Time-based: using `test_interval` and `val_interval` parameters

    Parameters
    ----------
    features : DataFrame
        Feature matrix to be split.

    split : Sequence[int], optional
        Sequence of three integers defining the train/val/test percentage ratio.
        Must sum to 100. Example: [70, 15, 15] for 70% train, 15% val, 15% test.
        Required for percentage-based splitting.

    test_interval : str, optional
        Time interval for test set. Required for time-based splitting.
        Supported units: 'm' (minutes), 'h' (hours), 'd' (days), 'M' (months).
        Example: '7d' for 7 days.

    val_interval : str, optional
        Time interval for validation set. Used only in time-based splitting.
        Same format as test_interval. If not provided, no validation set is created.

    groups : Series, optional
        Group labels for each sample. If not provided, all data will be
        treated as a single group named 'default_group'.

    timestamps : Series, optional
        Timestamps for each sample. Required for time-based splitting.
        For percentage-based splitting, if provided, data within each group
        will be sorted by time before splitting.

    Returns
    -------
    Dict[str, SplitResult]
        Dictionary where keys are group names and values are SplitResult objects
        containing splits for each group.

    Raises
    ------
    ValueError
        - If neither `split` nor `test_interval` is provided
        - If both `split` and `test_interval` are provided
        - If `split` doesn't sum to 100 (percentage-based mode)
        - If `timestamps` is not provided (time-based mode)
        - If invalid interval format is provided

    Examples
    --------
    Percentage-based splitting:
    >>> results = train_val_test_split(data, [60, 20, 20])

    Time-based splitting with validation:
    >>> results = train_val_test_split(
    ...     data,
    ...     test_interval='7d',
    ...     val_interval='2d',
    ...     timestamps=timestamps_series
    ... )

    Time-based splitting without validation:
    >>> results = train_val_test_split(
    ...     data,
    ...     test_interval='1M',
    ...     timestamps=timestamps_series
    ... )
    """
    # Determine splitting mode
    has_split = split is not None
    has_test_interval = test_interval is not None

    if not has_split and not has_test_interval:
        raise ValueError("Either `split` or `test_interval` must be provided")

    if has_split and has_test_interval:
        raise ValueError("Cannot use both `split` and `test_interval` parameters")

    # Percentage-based splitting
    if has_split:
        return _percentage_split(features, split, groups, timestamps)
    # Time-based splitting
    else:
        return _time_based_split(
            features,
            test_interval, # pyright: ignore[reportArgumentType]
            val_interval,
            groups,
            timestamps
        )


def _percentage_split(
    features: DataFrame,
    split: Sequence[int],
    groups: Optional[Series] = None,
    timestamps: Optional[Series] = None
) -> Dict[str, SplitResult]:
    """Percentage-based splitting implementation."""
    # Check parameters
    if (curr_sum := sum(split)) != 100:
        raise ValueError(
            f"All elements of the `split` sequence must summing to 100, "
            f"current sum: {curr_sum}."
        )

    # Convert to shares
    train_size, val_size, test_size = [i / 100 for i in split]

    # If no groups provided, create a single group
    if groups is None:
        groups = Series(['default_group'] * len(features))

    results = {}

    for group_name in groups.unique():
        group_mask = groups == group_name
        group_features = features[group_mask]

        # Sort by timestamps if provided
        if timestamps is not None:
            group_timestamps = timestamps[group_mask]
            group_features = group_features.iloc[group_timestamps.argsort()]

        # Calculate number of samples for each set
        num_trainable = int(len(group_features) * train_size)
        num_validatable = int((len(group_features) - num_trainable) *
                            (val_size / (val_size + test_size)))

        # Create slices for each set
        trainable_idx = group_features.iloc[:num_trainable].index.tolist()
        validatable_idx = group_features.iloc[
            num_trainable:num_trainable + num_validatable
        ].index.tolist()
        testable_idx = group_features.iloc[
            num_trainable + num_validatable:
        ].index.tolist()

        # Create SplitResult for the group
        split_indices = SplitIndices(
            train_indices=trainable_idx,
            validation_indices=validatable_idx,
            test_indices=testable_idx,
            group=group_name
        )

        results[group_name] = SplitResult(
            group=group_name,
            train_test_split=split_indices,
            validation_splits=[split_indices]
        )

    return results


def _time_based_split(
    features: DataFrame,
    test_interval: str,
    val_interval: Optional[str] = None,
    groups: Optional[Series] = None,
    timestamps: Optional[Series] = None
) -> Dict[str, SplitResult]:
    """Time-based splitting implementation."""
    if timestamps is None:
        raise ValueError("`timestamps` must be provided for time-based splitting")

    # Parse intervals
    try:
        test_delta = _parse_interval(test_interval)
        val_delta = _parse_interval(val_interval) if val_interval else None
    except ValueError as e:
        raise ValueError(f"Invalid interval format: {e}")

    # If no groups provided, create a single group
    if groups is None:
        groups = Series(['default_group'] * len(features))

    results = {}

    for group_name in groups.unique():
        group_mask = groups == group_name
        group_features = features[group_mask]
        group_timestamps = timestamps[group_mask]

        # Sort by timestamps
        sort_indices = group_timestamps.argsort()
        group_features = group_features.iloc[sort_indices]
        group_timestamps = group_timestamps.iloc[sort_indices] # pyright: ignore[reportAttributeAccessIssue]

        # Get time boundaries
        max_time = group_timestamps.max()
        test_start = max_time - test_delta

        if val_delta is not None:
            val_start = test_start - val_delta
        else:
            val_start = test_start  # No validation period

        # Split indices based on time boundaries
        train_mask = group_timestamps < val_start
        train_indices = group_features[train_mask].index.tolist()

        if val_delta is not None:
            val_mask = (group_timestamps >= val_start) & (group_timestamps < test_start)
            val_indices = group_features[val_mask].index.tolist()
        else:
            val_indices = []

        test_mask = group_timestamps >= test_start
        test_indices = group_features[test_mask].index.tolist()

        # Create SplitResult for the group
        split_indices = SplitIndices(
            train_indices=train_indices,
            validation_indices=val_indices,
            test_indices=test_indices,
            group=group_name
        )

        results[group_name] = SplitResult(
            group=group_name,
            train_test_split=split_indices,
            validation_splits=[split_indices]
        )

    return results
