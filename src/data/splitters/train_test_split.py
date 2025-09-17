from typing import Dict, Optional, Sequence

from pandas import DataFrame, Series

from ...config.schemas.splitting import SplitIndices, SplitResult


def train_val_test_split(
    features: DataFrame,
    split: Sequence[int],
    groups: Optional[Series] = None,
    timestamps: Optional[Series] = None
) -> Dict[str, SplitResult]:
    """
    Splits data into training, validation and test sets with group support.

    This function implements simple train/val/test data splitting while preserving
    group structure. Each group is processed independently, ensuring no cross-group
    data contamination.

    Key Features:
    - Group-aware splitting: Each group is processed independently
    - Temporal sorting: Optional timestamp-based sorting within groups
    - Unified API: Returns results in the same format as GroupTimeSeriesSplit
    - Percentage-based splitting: Precise splitting according to specified percentages

    Parameters
    ----------
    features : DataFrame
        Feature matrix to be split.

    split : Sequence[int]
        Sequence of three integers defining the train/val/test percentage ratio.
        Must sum to 100.
        Example: [70, 15, 15] for 70% train, 15% val, 15% test.

    groups : Series, optional
        Group labels for each sample. If not provided, all data will be
        treated as a single group named 'default_group'.

    timestamps : Series, optional
        Timestamps for each sample. If provided, data within each group
        will be sorted by time before splitting.

    Returns
    -------
    Dict[str, SplitResult]
        Dictionary where keys are group names and values are SplitResult objects
        containing splits for each group.

    Examples
    --------
    Basic usage without groups:

    >>> from pandas import DataFrame
    >>> data = DataFrame({'feature1': [1, 2, 3, 4, 5], 'feature2': [6, 7, 8, 9, 10]})
    >>> results = train_val_test_split(data, [60, 20, 20])
    >>> for group_name, result in results.items():
    >>>     print(f"Group: {group_name}")
    >>>     print(f"Train indices: {result.train_test_split.train_idx}")
    >>>     print(f"Test indices: {result.train_test_split.test_idx}")
    >>>     for val_split in result.validation_splits:
    >>>         print(f"Validation indices: {val_split.val_idx}")

    Usage with groups and timestamps:

    >>> import pandas as pd
    >>> data = DataFrame({
    ...     'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
    ...     'feature2': [9, 10, 11, 12, 13, 14, 15, 16]
    ... })
    >>> groups = pd.Series(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
    >>> timestamps = pd.Series(pd.date_range('2023-01-01', periods=8, freq='D'))
    >>> results = train_val_test_split(data, [50, 25, 25], groups, timestamps)
    >>> for group_name, result in results.items():
    >>>     print(f"Group {group_name}:")
    >>>     print(f"  Train size: {len(result.train_test_split.train_idx)}")
    >>>     print(f"  Val size: {len(result.validation_splits[0].val_idx)}")
    >>>     print(f"  Test size: {len(result.train_test_split.test_idx)}")

    Usage with time series data:

    >>> # Create time series data with multiple groups
    >>> dates = pd.date_range('2023-01-01', '2023-01-30')
    >>> group_a = DataFrame({
    ...     'value': range(30),
    ...     'group': 'A',
    ...     'timestamp': dates
    ... })
    >>> group_b = DataFrame({
    ...     'value': range(30, 60),
    ...     'group': 'B',
    ...     'timestamp': dates
    ... })
    >>> data = pd.concat([group_a, group_b])
    >>> results = train_val_test_split(
    ...     data,
    ...     [60, 20, 20],
    ...     groups=data['group'],
    ...     timestamps=data['timestamp']
    ... )
    >>> # Train model on group A data
    >>> group_a_result = results['A']
    >>> X_train = data.loc[group_a_result.train_test_split.train_idx]
    >>> X_val = data.loc[group_a_result.validation_splits[0].val_idx]
    >>> X_test = data.loc[group_a_result.train_test_split.test_idx]

    Notes
    ----------
    - For time series data, ensure data is sorted by time before calling
      the function or use the `timestamps` parameter
    - To standardize the output interface, the `SplitResult` object is returned,
      which, as with `GroupTimeSeriesSplit`, contains two attributes: `train_test_split`
      and `validation_splits`. With a simple split, these attributes contain
      the same `SplitIndices` object!
    - Splitting is performed sequentially (not randomly) to preserve
      temporal order
    - If data has no natural groups, the function will create a single group
      named 'default_group'
    """
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
            group_features   = group_features.iloc[group_timestamps.argsort()]

        # Calculate number of samples for each set
        num_trainable   = int(len(group_features) * train_size)
        num_validatable = int((len(group_features) - num_trainable) * (val_size / (val_size + test_size)))

        # Create slices for each set
        trainable_idx   = group_features.iloc[:num_trainable].index.tolist()
        validatable_idx = group_features.iloc[num_trainable:num_trainable + num_validatable].index.tolist()
        testable_idx    = group_features.iloc[num_trainable + num_validatable:].index.tolist()

        # Create SplitResult for the group
        train_val_test_split = SplitIndices(
            train_idx=trainable_idx,
            val_idx=validatable_idx,
            test_idx=testable_idx,
            group=group_name
        )

        results[group_name] = SplitResult(
            group=group_name,
            train_test_split=train_val_test_split,
            validation_splits=[train_val_test_split]
        )

    return results
