import sys
import os

import pandas as pd
import pytest

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
sys.path.insert(0, project_root)

from src.preprocessing.cv import GroupTimeSeriesSplit, SplitIndices


# Параметризованные тест-кейсы
TEST_CASES = [
    # Тест 1: Одиночное разбиение
    {
        "name": "single_split",
        "dates": pd.date_range('2023-01-01', '2023-01-10', freq='D'),
        "params": {"val_folds": 1, "test_folds": 0, "interval": '2d'},
        "expected": [
            SplitIndices(train_idx=[0, 1, 2, 3, 4, 5, 6, 7], val_idx=[8, 9], test_idx=None)
        ]
    },
    # Тест 2: Множественное разбиение
    {
        "name": "multiple_splits",
        "dates": pd.date_range('2023-01-01', '2023-01-20', freq='D'),
        "params": {"val_folds": 3, "test_folds": 0, "interval": '3d'},
        "expected": [
            SplitIndices(train_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], val_idx=[11, 12, 13], test_idx=None),
            SplitIndices(train_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], val_idx=[14, 15, 16], test_idx=None),
            SplitIndices(train_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], val_idx=[17, 18, 19], test_idx=None)
        ]
    },
    # Тест 3: Скользящее окно с тестовыми фолдами
    {
        "name": "rolling_window_with_test_folds",
        "dates": pd.date_range('2023-01-01', '2023-01-16', freq='D'),
        "params": {"val_folds": 2, "test_folds": 1, "interval": '3d', "train_interval": "4d", "window": 'rolling'},
        "expected": [
            SplitIndices(train_idx=[3, 4, 5, 6], val_idx=[7, 8, 9], test_idx=None),
            SplitIndices(train_idx=[6, 7, 8, 9], val_idx=[10, 11, 12], test_idx=None),
            SplitIndices(train_idx=[9, 10, 11, 12], val_idx=None, test_idx=[13, 14, 15])
        ]
    }
]

@pytest.mark.parametrize("test_case", TEST_CASES, ids=[tc["name"] for tc in TEST_CASES])
def test_group_time_series_split(test_case):
    """Параметризованный тест для GroupTimeSeriesSplit"""
    dates = test_case["dates"]
    groups = ['A'] * len(dates)

    cv = GroupTimeSeriesSplit(**test_case["params"])
    splits = list(cv.split(
        X=None,
        y=None,
        groups=pd.Series(groups),
        timestamps=pd.Series(dates)
    ))

    expected = test_case["expected"]

    # Проверяем количество сплитов
    assert len(splits) == len(expected)

    # Проверяем каждый сплит
    for i, (actual_split, expected_split) in enumerate(zip(splits, expected)):
        assert actual_split.train_idx == expected_split.train_idx, f"Train index mismatch in split {i}"
        assert actual_split.val_idx == expected_split.val_idx, f"Validation index mismatch in split {i}"
        assert actual_split.test_idx == expected_split.test_idx, f"Test index mismatch in split {i}"
