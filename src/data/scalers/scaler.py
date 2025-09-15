import os, sys
from typing import List, Set, Literal

import pandas as pd

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
sys.path.insert(0, project_root)

from src.config.constants import SCALERS, NON_SCALABLE_COLS


def get_scalable_cols(
    columns: pd.Index,
    non_scalable_cols: Set[str] = NON_SCALABLE_COLS
) -> List[str]:
    """Определяет какие колонки можно масштабировать"""
    scalable = []
    for col in columns:
        if not any([col.startswith(nonsc) for nonsc in non_scalable_cols]):
            scalable.append(col)
    return scalable


def apply_scaling(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    scaling_method: Literal["minmax", "standard", "robust"],
    non_scalable_cols: Set[str] = NON_SCALABLE_COLS,
    include_target: bool = True,
    task: str = "returns"
) -> tuple:
    """Применяет масштабирование к данным"""
    # Select columns for scaling
    scalable_cols = get_scalable_cols(X_train.columns, non_scalable_cols)

    # Scale features
    X_scaler = SCALERS[scaling_method]()
    X_scaler.fit(X_train[scalable_cols])

    # Copy data to prevent inplace modifications
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    # Scale features
    X_train_scaled[scalable_cols] = X_scaler.transform(X_train[scalable_cols])
    X_val_scaled[scalable_cols] = X_scaler.transform(X_val[scalable_cols])
    X_test_scaled[scalable_cols] = X_scaler.transform(X_test[scalable_cols])

    # Scale target if needed
    y_scaler = None
    if include_target and task != "direction":
        y_scaler = SCALERS[scaling_method]()
        y_scaler.fit(y_train.values.reshape(-1, 1))

        y_train_scaled = y_scaler.transform(y_train.values.reshape(-1, 1)).flatten()
        y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).flatten()
        y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
    else:
        y_train_scaled = y_train
        y_val_scaled = y_val
        y_test_scaled = y_test

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled, X_scaler, y_scaler)