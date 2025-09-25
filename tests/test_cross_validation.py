import sys
import os

import pandas as pd
import pytest

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
sys.path.insert(0, project_root)

from src.data.splitters.cross_validation import GroupTimeSeriesSplit, SplitIndices, SplitResult


# Test data and expected results
TEST_CASES = [
    {
        "id": "single_validation_split",
        "description": "Single validation split with test interval",
        "dates": pd.date_range('2023-01-01', '2023-01-10', freq='D'),
        "groups": ['AAPL'] * 10,
        "params": {"k_folds": 1, "test_interval": "2d", "val_interval": '2d'},
        "expected": {
            'AAPL': SplitResult(
                group='AAPL',
                train_test_split=SplitIndices(
                    train_indices=[0, 1, 2, 3, 4, 5, 6, 7],
                    validation_indices=None,
                    test_indices=[8, 9],
                    group='AAPL'
                ),
                validation_splits=[
                    SplitIndices(
                        train_indices=[0, 1, 2, 3, 4, 5],
                        validation_indices=[6, 7],
                        test_indices=None,
                        group='AAPL'
                    )
                ]
            )
        }
    },
    {
        "id": "multiple_validation_folds",
        "description": "Multiple validation folds with larger interval",
        "dates": pd.date_range('2023-01-01', '2023-01-12', freq='D'),
        "groups": ['AAPL'] * 12,
        "params": {"k_folds": 2, "test_interval": "2d", "val_interval": '3d'},
        "expected": {
            'AAPL': SplitResult(
                group='AAPL',
                train_test_split=SplitIndices(
                    train_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    validation_indices=None,
                    test_indices=[10, 11],
                    group='AAPL'
                ),
                validation_splits=[
                    SplitIndices(
                        train_indices=[0, 1, 2, 3],
                        validation_indices=[4, 5, 6],
                        test_indices=None,
                        group='AAPL'
                    ),
                    SplitIndices(
                        train_indices=[0, 1, 2, 3, 4, 5, 6],
                        validation_indices=[7, 8, 9],
                        test_indices=None,
                        group='AAPL'
                    )
                ]
            )
        }
    },
    {
        "id": "different_intervals",
        "description": "Different intervals for validation folds",
        "dates": pd.date_range('2023-01-01', '2023-01-12', freq='D'),
        "groups": ['AAPL'] * 12,
        "params": {"k_folds": 3, "test_interval": "2d", "val_interval": '1d'},
        "expected": {
            'AAPL': SplitResult(
                group='AAPL',
                train_test_split=SplitIndices(
                    train_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    validation_indices=None,
                    test_indices=[10, 11],
                    group='AAPL'
                ),
                validation_splits=[
                    SplitIndices(
                        train_indices=[0, 1, 2, 3, 4, 5, 6],
                        validation_indices=[7],
                        test_indices=None,
                        group='AAPL'
                    ),
                    SplitIndices(
                        train_indices=[0, 1, 2, 3, 4, 5, 6, 7],
                        validation_indices=[8],
                        test_indices=None,
                        group='AAPL'
                    ),
                    SplitIndices(
                        train_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                        validation_indices=[9],
                        test_indices=None,
                        group='AAPL'
                    )
                ]
            )
        }
    },
    {
        "id": "rolling_window",
        "description": "Rolling window without test split",
        "dates": pd.date_range('2023-01-01', '2023-01-16', freq='D'),
        "groups": ['AAPL'] * 16,
        "params": {"k_folds": 2, "val_interval": '3d', "train_interval": "6d", "window": 'rolling'},
        "expected": {
            'AAPL': SplitResult(
                group='AAPL',
                train_test_split=None,
                validation_splits=[
                    SplitIndices(
                        train_indices=[3, 4, 5, 6, 7, 8, 9],
                        validation_indices=[10, 11, 12],
                        test_indices=None,
                        group='AAPL'
                    ),
                    SplitIndices(
                        train_indices=[6, 7, 8, 9, 10, 11, 12],
                        validation_indices=[13, 14, 15],
                        test_indices=None,
                        group='AAPL'
                    )
                ]
            )
        }
    },
    {
        "id": "multiple_groups",
        "description": "Multiple groups with separate splits",
        "dates": pd.date_range('2023-01-01', '2023-01-10', freq='D'),
        "groups": ['AAPL'] * 10 + ['MSFT'] * 10,
        "timestamps": lambda: pd.Series(list(pd.date_range('2023-01-01', '2023-01-10', freq='D')) * 2),
        "params": {"k_folds": 2, "test_interval": "2d", "val_interval": '2d'},
        "expected": {
            'AAPL': SplitResult(
                group='AAPL',
                train_test_split=SplitIndices(
                    train_indices=[0, 1, 2, 3, 4, 5, 6, 7],
                    validation_indices=None,
                    test_indices=[8, 9],
                    group='AAPL'
                ),
                validation_splits=[
                    SplitIndices(
                        train_indices=[0, 1, 2, 3],
                        validation_indices=[4, 5],
                        test_indices=None,
                        group='AAPL'
                    ),
                    SplitIndices(
                        train_indices=[0, 1, 2, 3, 4, 5],
                        validation_indices=[6, 7],
                        test_indices=None,
                        group='AAPL'
                    )
                ]
            ),
            'MSFT': SplitResult(
                group='MSFT',
                train_test_split=SplitIndices(
                    train_indices=[10, 11, 12, 13, 14, 15, 16, 17],
                    validation_indices=None,
                    test_indices=[18, 19],
                    group='MSFT'
                ),
                validation_splits=[
                    SplitIndices(
                        train_indices=[10, 11, 12, 13],
                        validation_indices=[14, 15],
                        test_indices=None,
                        group='MSFT'
                    ),
                    SplitIndices(
                        train_indices=[10, 11, 12, 13, 14, 15],
                        validation_indices=[16, 17],
                        test_indices=None,
                        group='MSFT'
                    )
                ]
            )
        }
    }
]

@pytest.mark.parametrize("test_case", TEST_CASES, ids=[case["id"] for case in TEST_CASES])
def test_group_time_series_split(test_case):
    """Test GroupTimeSeriesSplit with various configurations"""
    # Prepare test data
    dates = test_case["dates"]
    groups = test_case["groups"]

    # Handle special case for multiple groups with custom timestamps
    if "timestamps" in test_case:
        timestamps = test_case["timestamps"]()
    else:
        timestamps = pd.Series(dates)

    # Initialize splitter
    cv = GroupTimeSeriesSplit(**test_case["params"])

    # Execute split
    splits = cv.split(
        X=None,
        y=None,
        groups=pd.Series(groups),
        timestamps=timestamps
    )

    # Verify results
    expected = test_case["expected"]

    assert splits.keys() == expected.keys()

    for group in splits:
        assert splits[group].group == expected[group].group
        assert splits[group].train_test_split == expected[group].train_test_split
        assert splits[group].validation_splits == expected[group].validation_splits
