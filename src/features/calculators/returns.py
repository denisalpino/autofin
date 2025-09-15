from typing import Literal

from pandas import Series
from numpy import log


def calculate_returns(
    price: Series,
    period: int = 1,
    method: Literal["momentum", "pct_change", "price_change"] = "pct_change",
    log=False
) -> Series:
    """
    Calculate price returns using specified method.

    Parameters
    ---
    price :
        Series of price values
    period:
        Number of periods to calculate returns over
    method:
        Method for calculating returns:
            - "momentum": price[t] / price[t-period]
            - "pct_change": percentage change
            - "price_change": absolute price difference
    log :
        Whether to apply logarithmic transformation (only for momentum method)

    Returns
    ---
    pd.Series: Series of calculated returns

    Raises
    ---
    ValueError: If invalid method is provided or log is used with unsupported method
    """
    # Momentum method: simple price ratio
    if method == "momentum":
        returns = price / price.shift(period)
        # Apply log transformation if requested
        return Series(log(returns)) if log else returns - 1
    # Validate that log is only used with momentum method
    elif log:
        raise ValueError("Using `log=True` is only available with `method='momentum'`.")
    # Standard percentage change method
    elif method == "pct_change":
        return price.pct_change(period)
    # Absolute price difference method
    elif method == "price_change":
        return price - price.shift(period)
    # Handle unknown method
    raise ValueError(f"Unknown method: {method}. Please consult the function docstring.")