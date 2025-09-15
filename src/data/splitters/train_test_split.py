from typing import Sequence, Tuple

from pandas import DataFrame


def train_val_test_split(
    features: DataFrame,
    split: Sequence[int]
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Разделяет данные на train/val/test по заданным процентам"""
    # Check parameters
    if (curr_sum := sum(split)) != 100:
        raise ValueError(
            f"All elements of the `split` sequence must summing to 100, "
            f"current sum: {curr_sum}."
        )

    # Convert to shares
    train_size, val_size, test_size = [i / 100 for i in split]

    # Calculate number of samples for training and validation
    num_trainable   = int(len(features) * train_size)
    num_validatable = int((len(features) - num_trainable) * (val_size / (val_size + test_size)))

    # Create slices for each set
    trainable   = features.iloc[:num_trainable]
    validatable = features.iloc[num_trainable:num_trainable + num_validatable]
    testable    = features.iloc[num_trainable + num_validatable:]

    return trainable, validatable, testable
