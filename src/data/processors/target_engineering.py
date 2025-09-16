import os
import sys
from typing import Literal

from pandas import DataFrame, Series

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
sys.path.insert(0, project_root)

from src.config.schemas.features import ReturnsMethod
from src.features.calculators.returns import calculate_returns


def create_direction_target(
    df: DataFrame,
    price_col_name: str,
    min_step: int = 1,
) -> Series:
    """
    Create directional target variable indicating price movement direction.

    Parameters
    ---
    df : DataFrame
        DataFrame containing price data
    price_col_name : str
        Name of the price column to use
    min_step : int, default=1
        Minimum number of periods for a price movement to be considered significant

    Returns
    ---
    Series: Binary series indicating price direction (1 for up, 0 for down)
    """
    s = df[price_col_name]
    # Identify runs where price doesn't change for min_step periods
    run_id = (s != s.shift(min_step)).cumsum()
    # Get the first value of each run
    run_first = s.groupby(run_id).first()
    # Get the first value of the next run
    next_run_first = run_first.shift(-1)
    # Determine if next run's first value is higher (1) or lower (0)
    directions = (next_run_first > run_first).astype("Int8")
    # Map directions back to original index
    target = run_id.map(directions).astype("Int8")
    target.name = "target"
    return target


def create_price_target(
    df: DataFrame,
    price_col_name: str = "close",
    returns: ReturnsMethod | Literal[False] = ReturnsMethod.LOG
) -> Series:
    """
    Create price-based target variable.

    Parameters
    ---
    df : DataFrame
        DataFrame containing price data
    price_col_name : str, default="close"
        Name of the price column to use
    returns : ReturnsMethod | Literal[False], default=ReturnsMethod.LOG:
        Method for calculating the target variable:
        - ReturnsMethod.RAW: absolute price difference
        - ReturnsMethod.PERCENT: percentage change
        - ReturnsMethod.LOG: logarithmic ratio of current to previous price
        - False: next price directly

    Returns
    ---
    Series: Series of percentage price changes
    """
    price = df[price_col_name]

    # Returns-based target
    if returns:
        target = calculate_returns(price, returns)
    else: # Plain prices
        target = price

    target = target.shift(-1)
    target = target.rename("target")
    return target
