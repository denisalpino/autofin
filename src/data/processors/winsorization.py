from typing import Tuple

from pandas import Series
from numpy import clip, quantile


def winsorize(
    y_train: Series,
    y_val: Series,
    y_test: Series,
    winsorize_percent: int
) -> Tuple[Series, Series, Series]:
    """Применяет winsorization к целевым переменным"""
    winsorize_frac = winsorize_percent / 200
    low, high = quantile(y_train, [winsorize_frac, 1 - winsorize_frac])

    y_train_clipped = Series(clip(y_train, low, high))
    y_val_clipped = Series(clip(y_val, low, high))
    y_test_clipped = Series(clip(y_test, low, high))

    return y_train_clipped, y_val_clipped, y_test_clipped
