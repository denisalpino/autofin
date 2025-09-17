import os
import sys

from pandas import Series
from numpy import log

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
sys.path.insert(0, project_root)

from src.config.schemas.features import ReturnsMethod


def calculate_returns(
    price: Series,
    method: ReturnsMethod = ReturnsMethod.LOG,
    period: int = 1
) -> Series:
    """
    Calculate price returns using specified method.

    Parameters
    ---
    price : Series
        Series of price values
    method : ReturnsMethod, default=ReturnsMethod.LOG
        Method for calculating returns:
        - ReturnsMethod.RAW: absolute price difference
        - ReturnsMethod.PERCENT: percentage change
        - ReturnsMethod.LOG: logarithmic ratio of current to previous price
    period : int, default=1
        Number of periods to calculate returns over

    Returns
    ---
    Series:
        Series of calculated returns

    Raises
    ---
    ValueError: If invalid method is provided or log is used with unsupported method
    """
    if period < 1:
        raise ValueError(
            "The period must be at least 1 to prevent data leakage, "
            f"while {period=} is obtained."
        )
    # Logarithmic method: simple price ratio
    if method == ReturnsMethod.LOG:
        # Apply log transformation
        return Series(log(price / price.shift(period)))
    # Standard percentage change method
    elif method == ReturnsMethod.PERCENT:
        return price.pct_change(period)
    # Absolute price difference method
    elif method == ReturnsMethod.RAW:
        return price - price.shift(period)
    # Handle unknown method
    raise ValueError(f"Unknown method: {method}. Please consult the function docstring.")