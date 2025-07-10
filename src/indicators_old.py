import pandas as pd
import numpy as np
from typing import Literal


def calculate_rsi(
        prices: pd.Series,
        window: int = 14
) -> pd.Series:
    """
    This function calculate RSI Indicator using correspondent value from the `window` param.
    """

    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean() # type: ignore
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean() # type: ignore
    rsi = 1 - (100 / (1 + gain / loss)) / 100
    return rsi


def calculate_bollinger_bands_position(
        prices: pd.Series,
        window: int = 14,
        std: float = 2,
        kind: Literal["simple", "exponential"] = "exponential"
) -> pd.Series:
    """Docstring"""

    bb_middle = calculate_moving_average(prices, window, kind)
    bb_std = prices.ewm(span=window).std() if kind == "exponential" else prices.rolling(window=window).std()

    bb_upper = bb_middle + (std * bb_std)
    bb_lower = bb_middle - (std * bb_std)

    bb_position = (prices - bb_lower) / (bb_upper - bb_lower)
    return bb_position


def calculate_moving_average(
        prices: pd.Series,
        window: int = 14,
        kind: Literal["simple", "exponential"] = "exponential"
) -> pd.Series:
    """Docstring"""

    if kind == "exponential":
        return prices.ewm(span=window).mean()
    elif kind == "simple":
        return prices.rolling(window=window).mean()
    raise ValueError(
        f"Unknown moving average kind: {kind}. Please read docstring of calculate_moving_average() function.")


def calculate_moving_average_convergence_divergence(
        prices: pd.Series,
        short_window: int = 12,
        long_window: int = 26,
        signal_window: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """
    Function calculate MACD and MACD Signal based on short and long Exponential Moving Average (EMA) lines

    :param prices: ...
    :param short_window: ...
    :param ong_window: ...
    :param ignal_window: ...

    :returns:
    * pandas.Series:
        Main Line
    * pandas.Series:
        Signal Line
    """

    short_ema = calculate_moving_average(prices, short_window, "exponential")
    long_ema = calculate_moving_average(prices, long_window, "exponential")
    main_line = short_ema - long_ema
    signal = calculate_moving_average(main_line, signal_window, "exponential")

    return main_line, signal


def calculate_true_range(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        average: bool = False,
        window: int = 14,
        kind: Literal["simple", "exponential"] = "exponential"
) -> pd.Series:
    """
    If average = True, ATR will be calculated. Params window and kind specifies only
    for ATR calculation
    """

    # True Range calculation
    true_range = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    # Average True Range (ATR) calculation if needed
    if average:
        return calculate_moving_average(true_range, window=window, kind=kind)
    return true_range


def calculate_average_directional_movement(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        window: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculates the ADX (Average Directional Movement Index) indicator and its components +DI (PDI) and -DI (NDI).

    :params:
    ---
    `close`: `pd.Series`
        array of closing prices
    :param `high`: `pd.Series`
        `pd.Series` of maximum prices
    `low`: `pd.Series`
        array of minimum prices
    `window`: `int`
        Calculation period (default is 14)

    :returns:
    * `tuple` of three `pd.Series`:
        - PDI: Positive directional indicator;
        - NDI: Negative directional indicator;
        - ADX: Average directional movement index.

    Algotithm:
    ---
    1. Calculation of Directional Movement (+DM и -DM)
    2. Calculation of True Range (TR)
    3. Smooth DM and TR using Wilder method
    4. Calculation of Directional Indicators (+DI, -DI)
    5. Calculation of Directional Movement Index (DX)
    6. Smooth DX for getting ADX
    """

    # Directional Movement
    up_move = high.diff(1)
    down_move = -low.diff(1)
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0) # type: ignore
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0) # type: ignore

    # True Range
    tr = calculate_true_range(high, low, close, average=False, window=window)

    # Wilder Smoothing method
    def wilder_smoothing(series: pd.Series, window: int) -> pd.Series:
        smooth = series.copy()
        smooth.iloc[window - 1] = series.iloc[:window - 1].sum()

        for i in range(window, len(series)):
            smooth.iloc[i] = smooth.iloc[i-1] - (smooth.iloc[i-1] / window) + series.iloc[i]
        return smooth

    atr = wilder_smoothing(tr, window)
    plus_dm_smoothed = wilder_smoothing(plus_dm, window)
    minus_dm_smoothed = wilder_smoothing(minus_dm, window)

    # Directional Indicators
    plus_di = plus_dm_smoothed / atr
    minus_di = minus_dm_smoothed / atr

    # Directional Movement Index
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    dx.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ADX (smoothed DX)
    adx = wilder_smoothing(dx.dropna(), window)
    adx = adx.reindex(close.index, fill_value=np.nan)

    return plus_di, minus_di, adx


def calculate_price_channel(
        high: pd.Series,
        low: pd.Series,
        window: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """
    Calculates the Price Channel (Donchian Channel), an indicator that determines support and resistance levels.

    :param high: `pd.Series` maximal prices
    :param low: `pd.Series` minimal prices
    :param window: period for calculation (default 20)

    :return: `tuple` of 2 `pd.Series` (upper and lower channels)
    """
    upper_channel = high.rolling(window=window).max()
    lower_channel = low.rolling(window=window).min()

    return upper_channel, lower_channel


def calculate_elder_ray_power(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        window: int = 13
) -> tuple[pd.Series, pd.Series]:
    """
    Calculates Bears Power and Bulls Power indicators from Elder Ray system.

    :param close: `pd.Series` of close prices
    :param high: `pd.Series` maximal prices
    :param low: `pd.Series` minimal prices
    :param window: period for EMA (default 13)

    :return: `tuple` of two `pd.Series` (Bears Power, Bulls Power)
    """
    ema = calculate_moving_average(close, window, "exponential")
    bulls_power = high - ema
    bears_power = low - ema

    return bears_power, bulls_power


def calculate_smoothed_rate_of_change(
        prices: pd.Series,
        roc_period: int = 12,
        smooth_period: int = 6,
        kind: Literal["simple", "exponential"] = "exponential"
) -> pd.Series:
    """
    Calculates the smoothed Rate of Change, a modification of the standard ROC with smoothing

    :param prices: `pd.Series` of prices (close by default)
    :param roc_period: priod for ROC (по умолчанию 12)
    :param smooth_period: smoothing period for MA (default 6)
    :param kind: Тип скользящей средней (по умолчанию exponential)

    :return: `pd.Series` of SROC
    """
    # Standard ROC
    roc = (prices / prices.shift(roc_period) - 1) * 100

    # Smoothing
    sroc = calculate_moving_average(roc, smooth_period, kind)
    return sroc


def calculate_mass_index(
        high: pd.Series,
        low: pd.Series,
        short_window: int = 9,
        long_window: int = 25,
        sum_period: int = 25
) -> pd.Series:
    """
    Calculates Mass Index - indicator, describing reversal points based on volatility.

    :param high: `pd.Series` of maximum prices
    :param low: `pd.Series` of minimum prices
    :param short_window: period of short EMA (default 9)
    :param long_window: period of long EMA (default 25)
    :param period: summing period (default 25)

    :return: `pd.Series`
    """
    high_low_ratio = (high - low) / low * 100
    ema_short = calculate_moving_average(high_low_ratio, short_window, "exponential")
    ema_long = calculate_moving_average(ema_short, long_window, "exponential")

    ratio = ema_short / ema_long
    mass_index = ratio.rolling(window=sum_period).sum()
    return mass_index