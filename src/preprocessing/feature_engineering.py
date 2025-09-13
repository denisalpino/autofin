import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pandas_ta as indicators

from typing import Literal


def create_features(
    df: pd.DataFrame,
    features_config: dict
) -> pd.DataFrame:
    df = df.copy()
    original_index = df.index
    base_columns = features_config.get("base_columns", {})
    feature_frames = []

    # Include OHLC features if needed
    if ohlc := features_config.get("ohlc"):
        for price, include in ohlc.items():
            if include:
                col_name = base_columns[price]
                feature_frames.append(df[col_name].copy())

    # Time-based feature engineering
    if tf_cfg := features_config.get("time_features"):
        col = base_columns["timestamps"]
        bool_params = [
            param in tf_cfg
            for param in ["minute", "hour", "day", "day_of_week", "month"]
        ]
        time_feats = create_time_features(df[col], *bool_params)
        feature_frames.append(time_feats)

    # Returns feature engineering
    if ret_cfg := features_config.get("returns"):
        ret = calculate_returns(
            df[ret_cfg["column"]],
            period=ret_cfg["period"],
            method=ret_cfg["method"],
            log=ret_cfg["log"]
        )
        ret.name = "returns"
        feature_frames.append(ret)

    # Indicator-based feature engineering
    if ind_cfg := features_config.get("indicators"):
        if isinstance(ind_cfg, dict):
            ind_frames = [
                calculate_indicator(df, name, cfg, base_columns)
                for name, cfg in ind_cfg.items()
            ]
            feature_frames.append(pd.concat(ind_frames, axis=1))
        else:
            raise TypeError(f"Unsupported indicator configuration type: {type(ind_cfg)}")

    # Concatenate features built so far
    feature_frames = [frame for frame in feature_frames if not frame.empty]
    if feature_frames:
        features = pd.concat(feature_frames, axis=1).reindex(df.index)
    else:
        features = pd.DataFrame(index=df.index)

    features = pd.concat(feature_frames, axis=1)

    # Lagging feature engineering
    if lags := features_config.get("lags"):
        for col, lag in lags.items():
            lag_feats = get_lagging_features(features[col], max_lag=lag)
            features = pd.concat([features, lag_feats], axis=1)

    return features.reindex(original_index)


def calculate_indicator(
    df: pd.DataFrame,
    name: str,
    config: dict,
    base_columns: dict
) -> pd.DataFrame | pd.Series:
    """
    Calculate an indicator based on the provided configuration.
    """
    open, high = df[base_columns["open"]], df[base_columns["high"]]
    low, close = df[base_columns["low"]], df[base_columns["close"]]

    # Average True Range
    if name.startswith("ATR"):
        atr = indicators.atr(
            high,
            low,
            close,
            length=config["window"],
        mamode=config["kind"]
        )
        return atr  # type: ignore

    # Relative Strength Index
    elif name.startswith("RSI"):
        prices = df[config["price"]]
        rsi = indicators.rsi(prices, length=config["window"], scalar=1)
        return rsi  # type: ignore

    # Bollinger Bands
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

    # Average Directional Index
    elif name.startswith("ADX"):
        adx = indicators.adx(
            high, low, close,
            length=config["base_window"],
            lensig=config["signal_window"],
            mamode=config["kind"],
            scalar=1)
        adx.columns = ["ADX", "+DI", "-DI"]  # type: ignore
        return adx[config["output"]]  # type: ignore

    # Donchain Channels
    elif name.startswith("DC"):
        dc = indicators.donchian(
            high,
            low,
            lower_length=config["window"],
            upper_length=config["window"])
        dcr = pd.DataFrame()

        for col, ser in dc.items():  # type: ignore
            dcr[col] = ser

        dcr.columns = ["LowDC", "MidDC", "UpDC"]
        return dcr[config["output"]]

    # Elder Ray Power
    elif name.startswith("ERP"):
        ema = indicators.ema(close, length=config["window"])
        bears_power = high - ema
        bulls_power = low - ema
        bears_power.name = "BePo"
        bulls_power.name = "BuPo"
        return pd.concat([bears_power, bulls_power], axis=1)
    # Mass Index (MI)
    elif name.startswith("MI"):
        mi = indicators.massi(
            df.high,
            df.low,
            fast=config["short_window"],
            slow=config["long_window"]
        )
        return mi  # type: ignore
    # Moving Average (including few kinds)
    elif name.startswith("MA_"):
        prices = df[config["price"]]
        ma = indicators.ma(config["kind"], prices, length=config["window"])
        return ma

    # Volatility Ratio
    elif name == "volatility_ratio":
        vr = high / low - 1
        vr.name = "volatility_ratio"
        return vr

    # Candle Strength
    elif name == "candle_strength":
        cs = ((close - open) / (high - low + 1e-5)).rename("candle_strength")
        return cs

    # Absolute Body Differance
    elif name == "body":
        body = (open - close).abs().rename("body")
        return body

    # Upper Wick (diff between high and open or close)
    elif name == "upper_wick":
        oc = pd.concat([open, close], axis=1)
        upper_wick = (high - oc.max(axis=1)).rename("upper_wick")
        return upper_wick

    # Lower Wick (diff between low and open or close)
    elif name == "lower_wick":
        oc = pd.concat([open, close], axis=1)
        lower_wick = (oc.min(axis=1) - low).rename("lower_wick")
        return lower_wick

    # Close-to-close differance (absolute return)
    elif name == "cc_diff":
        cc_diff = (close - close.shift(1)).rename("cc_diff")
        return cc_diff

    # Ratio of Close - EMA and EMA
    elif name.startswith("price_to_ema"):
        ma = indicators.ma(config["kind"], close, length=config["window"])
        price_to_ema = ((close - ma) / ma).rename(f"price_to_ema_{config['window']}")
        return price_to_ema

    # Z-Score close
    elif name.startswith("ZScore_"):
        ma_rolling = close.rolling(config["window"]).mean()
        std_rolling = close.rolling(config["window"]).std()
        zscore = ((close - ma_rolling) / std_rolling).rename(f"ZScore_{config['window']}")
        return zscore

    # Keltner Channels
    elif name.startswith("KC"):
        kc = indicators.kc(high, low, close, length=config["window"], scalar=config["scalar"])
        kcr = pd.DataFrame()

        for col, ser in kc.items():  # type: ignore
            kcr[col] = ser

        kcr.columns = ["LowKC", "MidKC", "UpKC"]
        return kcr[config["output"]]

    # Momentum with window
    elif name.startswith("MOM_"):
        mom = close.diff(config["window"]).rename(f"MOM_{config['window']}")
        return mom

    # Skew of absolute returns per rolling window
    elif name.startswith("ret_skew_"):
        ret_skew = (
            (close - close.shift())
            .rolling(config["window"])
            .skew()
            .rename(f"ret_skew_{config['window']}")
        )
        return ret_skew

    # Kurtosis of absolute returns per rolling window
    elif name.startswith("ret_kurt_"):
        ret_kurt = (
            (close - close.shift())
            .rolling(config["window"])
            .kurt()
            .rename(f"ret_kurt_{config['window']}")
        )
        return ret_kurt

    # Standart deviation of absolute returns per rolling window
    elif name.startswith("ret_std_"):
        ret_std = (
            (close - close.shift())
            .rolling(config["window"])
            .std()
            .rename(f"ret_std_{config['window']}")
        )
        return ret_std

    # Mean of absolute returns per rolling window
    elif name.startswith("ret_mean_"):
        ret_mean = (
            (close - close.shift())
            .rolling(config["window"])
            .mean()
            .rename(f"ret_mean_{config['window']}")
        )
        return ret_mean
    raise ValueError(f"Unknown indicator {name} detected!")


def create_time_features(
    timestamps: pd.Series,
    minute: bool = False,
    hour: bool = False,
    day: bool = False,
    day_of_week: bool = False,
    month: bool = False
) -> pd.DataFrame:
    """Calculates cyclic time-features based on `cos` and `sin` functions"""
    time_features = pd.DataFrame()

    if not any([minute, hour, day, day_of_week, month]):
        raise ValueError("No time features specified for encoding.")

    if minute:
        time_features = pd.concat([
            time_features,
            encode_cyclic(
                timestamps.dt.minute.to_numpy(),
                col_name="minute",
                max_val=60
            )
        ], axis=1)
    if hour:
        time_features = pd.concat([
            time_features,
            encode_cyclic(
                timestamps.dt.hour.to_numpy(),
                col_name="hour",
                max_val=24
            )
        ], axis=1)
    if day:
        time_features = pd.concat([
            time_features,
            encode_cyclic(
                timestamps.dt.day.to_numpy(),
                col_name="day",
                max_val=timestamps.dt.days_in_month.to_numpy()
            )
        ], axis=1)
    if day_of_week:
        time_features = pd.concat([
            time_features,
            encode_cyclic(
                timestamps.dt.day_of_week.to_numpy(),
                col_name="day_of_week",
                max_val=7
            )
        ], axis=1)
    if month:
        time_features = pd.concat([
            time_features,
            encode_cyclic(
                timestamps.dt.month.to_numpy(),
                col_name="month",
                max_val=12
            )
        ], axis=1)

    return time_features


def encode_cyclic(
    values: NDArray,
    col_name: str,
    max_val: int | NDArray
):
    """Encode cyclic features using sin and cos transformations"""
    encoded_features = pd.DataFrame()
    encoded_features[f"{col_name}_sin"] = np.sin(2 * np.pi * values / max_val)
    encoded_features[f"{col_name}_cos"] = np.cos(2 * np.pi * values / max_val)
    return encoded_features


def calculate_returns(
    price: pd.Series,
    period: int = 1,
    method: Literal["momentum", "pct_change", "price_change"] = "pct_change",
    log=False
) -> pd.Series:
    if method == "momentum":
        returns = price / price.shift(period)
        return pd.Series(np.log(returns)) if log else returns - 1
    elif log:
        raise ValueError("Using `log=True` is only available with `method='momentum'`.")
    elif method == "pct_change":
        return price.pct_change(period)
    elif method == "price_change":
        return price - price.shift(period)
    raise ValueError(f"Unknown method: {method}. Please consult the function docstring.")


def get_lagging_features(
    ser: pd.Series,
    max_lag: int
) -> pd.DataFrame:
    if max_lag < 1:
        raise ValueError("Parameter `max_lag` must be at least 1.")
    lagging_features = pd.DataFrame()
    for lag in range(1, max_lag + 1):
        lagging_features[f"{ser.name}_lag{lag}"] = ser.shift(lag)
    return lagging_features


def create_direction_target(
    df: pd.DataFrame,
    price_col_name: str,
    min_step: int = 1,
) -> pd.Series:
    s = df[price_col_name]
    run_id = (s != s.shift(min_step)).cumsum()
    run_first = s.groupby(run_id).first()
    next_run_first = run_first.shift(-1)
    directions = (next_run_first > run_first).astype("Int8")
    target = run_id.map(directions).astype("Int8")
    target.name = "target"
    return target


def create_price_target(
    df: pd.DataFrame,
    price_col_name: str
) -> pd.Series:
    price = df[price_col_name]
    diff = price - price.shift(-1)
    target = (diff / price.shift(-1)).rename("target")
    return target
