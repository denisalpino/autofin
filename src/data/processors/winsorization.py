from typing import Tuple

from pandas import Series

def winsorize(
    y_train: Series,
    y_val: Series,
    y_test: Series,
    winsorize_percent: int | float
) -> Tuple[Series, Series, Series]:
    """Применяет winsorization к целевым переменным"""
    p = winsorize_percent / 100.0
    lower_q = (1.0 - p) / 2.0
    upper_q = 1.0 - lower_q

    low = y_train.quantile(lower_q)
    high = y_train.quantile(upper_q)

    y_train_clipped = y_train.clip(lower=low, upper=high)
    y_val_clipped   = y_val.clip(lower=low, upper=high)
    y_test_clipped  = y_test.clip(lower=low, upper=high)

    return y_train_clipped, y_val_clipped, y_test_clipped
