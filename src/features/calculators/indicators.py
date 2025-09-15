import pandas_ta as indicators
from pandas import DataFrame, Series, concat


def calculate_indicator(
    df: DataFrame,
    name: str,
    config: dict,
    base_columns: dict
) -> DataFrame | Series:
    """
    Calculate an indicator based on the provided configuration.

    Parameters
    ---
    df: Input DataFrame with OHLC price data
    name: Name of the indicator to calculate
    config: Configuration dictionary for the indicator
    base_columns: Dictionary mapping standard column names to actual column names

    Returns
    ---
    DataFrame | Series:
        Calculated indicator values

    Raises
    ---
    ValueError: If unknown indicator name is provided
    KeyError: If required columns are missing from base_columns
    """
    # Extract OHLC columns using base_columns mapping
    # This ensures flexibility in column naming across different datasets
    open, high = df[base_columns["open"]], df[base_columns["high"]]
    low, close = df[base_columns["low"]], df[base_columns["close"]]

    # Average True Range - measures market volatility
    if name.startswith("ATR"):
        atr = indicators.atr(
            high,
            low,
            close,
            length=config["window"],
        mamode=config["kind"]
        )
        return atr  # type: ignore

    # Relative Strength Index - momentum oscillator
    elif name.startswith("RSI"):
        # Use specified price column (e.g., close, open, etc.)
        prices = df[config["price"]]
        rsi = indicators.rsi(prices, length=config["window"], scalar=1)
        return rsi  # type: ignore

    # Bollinger Bands - volatility bands around a moving average
    elif name.startswith("BBP"):
        bb = indicators.bbands(
            close,
            length=config["window"],
            std=config["std"],
            mamode=config["kind"]
        ).iloc[:, -2:]  # type: ignore
        bb.columns = ["BBB", "BBP"]
        return bb[config["output"]]  # type: ignore

    # Moving Average Convergence Divergence (including signal and histogram)
    # Trend-following momentum indicator
    elif name.startswith("MACD"):
        prices = df[config["price"]]
        macd = indicators.macd(
            prices,
            fast=config["short_window"],
            slow=config["long_window"],
            signal=config["signal_window"]
        )  # type: ignore
        macd.columns = ["MACD", "MACD_Signal", "MACD_Hist"]  # type: ignore
        return macd[config["output"]]  # type: ignore

    # Average Directional Index - measures trend strength
    elif name.startswith("ADX"):
        adx = indicators.adx(
            high, low, close,
            length=config["base_window"],
            lensig=config["signal_window"],
            mamode=config["kind"],
            scalar=1)
        adx.columns = ["ADX", "+DI", "-DI"]  # type: ignore
        return adx[config["output"]]  # type: ignore

    # Donchian Channels - volatility indicator showing highest high and lowest low
    elif name.startswith("DC"):
        dc = indicators.donchian(
            high,
            low,
            lower_length=config["window"],
            upper_length=config["window"])
        dcr = DataFrame()

        # Convert to DataFrame with proper column names
        for col, ser in dc.items():  # type: ignore
            dcr[col] = ser

        dcr.columns = ["LowDC", "MidDC", "UpDC"]
        return dcr[config["output"]]

    # Elder Ray Power - measures buying and selling pressure
    elif name.startswith("ERP"):
        ema = indicators.ema(close, length=config["window"])
        # Bull power measures the ability to push prices above EMA
        bears_power = high - ema
        # Bear power measures the ability to push prices below EMA
        bulls_power = low - ema
        bears_power.name = "BePo"
        bulls_power.name = "BuPo"
        return concat([bears_power, bulls_power], axis=1)

    # Mass Index (MI) - identifies trend reversals by measuring volatility
    elif name.startswith("MI"):
        mi = indicators.massi(
            df.high,
            df.low,
            fast=config["short_window"],
            slow=config["long_window"]
        )
        return mi  # type: ignore

    # Moving Average (including few kinds) - smooths price data
    elif name.startswith("MA_"):
        prices = df[config["price"]]
        ma = indicators.ma(config["kind"], prices, length=config["window"])
        return ma

    # Volatility Ratio - measures intraday volatility
    elif name == "volatility_ratio":
        vr = high / low - 1
        vr.name = "volatility_ratio"
        return vr

    # Candle Strength - measures how strong a candle is relative to its range
    elif name == "candle_strength":
        cs = ((close - open) / (high - low + 1e-5)).rename("candle_strength")
        return cs

    # Absolute Body Difference - measures the absolute size of the candle body
    elif name == "body":
        body = (open - close).abs().rename("body")
        return body

    # Upper Wick (diff between high and open or close)
    elif name == "upper_wick":
        oc = concat([open, close], axis=1)
        upper_wick = (high - oc.max(axis=1)).rename("upper_wick")
        return upper_wick

    # Lower Wick (diff between low and open or close)
    elif name == "lower_wick":
        oc = concat([open, close], axis=1)
        lower_wick = (oc.min(axis=1) - low).rename("lower_wick")
        return lower_wick

    # Close-to-close differance (absolute return)
    elif name == "cc_diff":
        cc_diff = (close - close.shift(1)).rename("cc_diff")
        return cc_diff

    # Ratio of Close - EMA and EMA - shows deviation from moving average
    elif name.startswith("price_to_ema"):
        ma = indicators.ma(config["kind"], close, length=config["window"])
        price_to_ema = ((close - ma) / ma).rename(f"price_to_ema_{config['window']}")
        return price_to_ema

    # Z-Score close - shows how many standard deviations close is from mean
    elif name.startswith("ZScore_"):
        ma_rolling = close.rolling(config["window"]).mean()
        std_rolling = close.rolling(config["window"]).std()
        zscore = ((close - ma_rolling) / std_rolling).rename(f"ZScore_{config['window']}")
        return zscore

    # Keltner Channels - volatility-based envelopes around moving average
    elif name.startswith("KC"):
        kc = indicators.kc(high, low, close, length=config["window"], scalar=config["scalar"])
        kcr = DataFrame()

        for col, ser in kc.items():  # type: ignore
            kcr[col] = ser

        kcr.columns = ["LowKC", "MidKC", "UpKC"]
        return kcr[config["output"]]

    # Momentum with window - measures the rate of price change
    elif name.startswith("MOM_"):
        mom = close.diff(config["window"]).rename(f"MOM_{config['window']}")
        return mom

    # Skew of absolute returns per rolling window - measures return
    # distribution asymmetry
    elif name.startswith("ret_skew_"):
        ret_skew = (
            (close - close.shift())
            .rolling(config["window"])
            .skew()
            .rename(f"ret_skew_{config['window']}")
        )
        return ret_skew

    # Kurtosis of absolute returns per rolling window - measures tail
    # heaviness of return distribution
    elif name.startswith("ret_kurt_"):
        ret_kurt = (
            (close - close.shift())
            .rolling(config["window"])
            .kurt()
            .rename(f"ret_kurt_{config['window']}")
        )
        return ret_kurt

    # Standart deviation of absolute returns per rolling window - measures volatility
    elif name.startswith("ret_std_"):
        ret_std = (
            (close - close.shift())
            .rolling(config["window"])
            .std()
            .rename(f"ret_std_{config['window']}")
        )
        return ret_std

    # Mean of absolute returns per rolling window - measures average return
    elif name.startswith("ret_mean_"):
        ret_mean = (
            (close - close.shift())
            .rolling(config["window"])
            .mean()
            .rename(f"ret_mean_{config['window']}")
        )
        return ret_mean
    raise ValueError(f"Unknown indicator {name} detected!")
