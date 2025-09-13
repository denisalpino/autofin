import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pandas_ta as indicators

from typing import Literal


def create_features(
    df: pd.DataFrame,
    features_config: dict
) -> pd.DataFrame:
    """
        Creates feature matrix based on configuration dictionary.

        Parameters
        ---
        df: Input DataFrame with raw financial data
        features_config: Configuration dictionary specifying which features to create
            - base_columns: Dictionary mapping standard names to actual column names
            - ohlc: Dictionary specifying which OHLC prices to include as features
            - time_features: Dictionary specifying which time-based features to create
            - returns: Dictionary configuring returns calculation
            - indicators: Dictionary configuring technical indicator calculations
            - lags: Dictionary specifying lag features to create

        Returns
        ---
        pd.DataFrame: DataFrame with engineered features

        Raises
        ---
        KeyError: If required base_columns are missing from features_config
        TypeError: If indicators configuration is not a dictionary
        ValueError: If time_features are requested but no parameters are specified
    """
    df = df.copy()
    # Store original index to ensure proper alignment at the end
    original_index = df.index
    # Extract base column mappings with default empty dict
    base_columns = features_config.get("base_columns", {})
    feature_frames = []

    # Include OHLC features if needed - these are the raw price values
    if ohlc := features_config.get("ohlc"):
        for price, include in ohlc.items():
            if include:
                # Get actual column name from base_columns mapping
                col_name = base_columns[price]
                # Add the price column as a feature
                feature_frames.append(df[col_name].copy())

    # Time-based feature engineering - creates cyclic time features
    if tf_cfg := features_config.get("time_features"):
        col = base_columns["timestamps"]
        # Create boolean list of which time features to generate
        bool_params = [
            param in tf_cfg
            for param in ["minute", "hour", "day", "day_of_week", "month"]
        ]
        # Generate time features using the timestamp column
        time_feats = create_time_features(df[col], *bool_params)
        feature_frames.append(time_feats)

    # Returns feature engineering - calculates price returns
    if ret_cfg := features_config.get("returns"):
        ret = calculate_returns(
            df[ret_cfg["column"]],
            period=ret_cfg["period"],
            method=ret_cfg["method"],
            log=ret_cfg["log"]
        )
        # Name the returns series for easier identification
        ret.name = "returns"
        feature_frames.append(ret)

    # Indicator-based feature engineering - calculates technical indicators
    if ind_cfg := features_config.get("indicators"):
        if isinstance(ind_cfg, dict):
            # Calculate all configured indicators
            ind_frames = [
                calculate_indicator(df, name, cfg, base_columns)
                for name, cfg in ind_cfg.items()
            ]
            # Concatenate all indicator results
            feature_frames.append(pd.concat(ind_frames, axis=1))
        else:
            raise TypeError(f"Unsupported indicator configuration type: {type(ind_cfg)}")

    # Filter out any empty DataFrames before concatenation
    feature_frames = [frame for frame in feature_frames if not frame.empty]
    # Concatenate all features, ensuring alignment with original index
    if feature_frames:
        features = pd.concat(feature_frames, axis=1).reindex(df.index)
    else:
        features = pd.DataFrame(index=df.index)

    # Lagging feature engineering - creates lagged versions of features
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

        Parameters
        ---
        df: Input DataFrame with OHLC price data
        name: Name of the indicator to calculate
        config: Configuration dictionary for the indicator
        base_columns: Dictionary mapping standard column names to actual column names

        Returns
        ---
        pd.DataFrame | pd.Series:
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
        dcr = pd.DataFrame()

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
        return pd.concat([bears_power, bulls_power], axis=1)

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
        kcr = pd.DataFrame()

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


def create_time_features(
    timestamps: pd.Series,
    minute: bool = False,
    hour: bool = False,
    day: bool = False,
    day_of_week: bool = False,
    month: bool = False
) -> pd.DataFrame:
    """
        Calculates cyclic time-features based on `cos` and `sin` functions.

        Parameters
        ---
        timestamps: Series containing datetime values
        minute: Whether to include minute-based features
        hour: Whether to include hour-based features
        day: Whether to include day-of-month features
        day_of_week: Whether to include day-of-week features
        month: Whether to include month features

        Returns
        ---
        pd.DataFrame: DataFrame with cyclic time features encoded as sin/cos pairs

        Raises
        ---
        ValueError: If no time features are specified for encoding
    """
    time_features = pd.DataFrame()

    # Validate that at least one time feature is requested
    if not any([minute, hour, day, day_of_week, month]):
        raise ValueError("No time features specified for encoding.")

    # Encode minute as cyclic feature using sin/cos transformation
    if minute:
        time_features = pd.concat([
            time_features,
            encode_cyclic(
                timestamps.dt.minute.to_numpy(),
                col_name="minute",
                max_val=60
            )
        ], axis=1)

    # Encode hour as cyclic feature using sin/cos transformation
    if hour:
        time_features = pd.concat([
            time_features,
            encode_cyclic(
                timestamps.dt.hour.to_numpy(),
                col_name="hour",
                max_val=24
            )
        ], axis=1)

    # Encode day as cyclic feature using sin/cos transformation
    if day:
        time_features = pd.concat([
            time_features,
            encode_cyclic(
                timestamps.dt.day.to_numpy(),
                col_name="day",
                max_val=timestamps.dt.days_in_month.to_numpy()
            )
        ], axis=1)

    # Encode day of week as cyclic feature using sin/cos transformation
    if day_of_week:
        time_features = pd.concat([
            time_features,
            encode_cyclic(
                timestamps.dt.day_of_week.to_numpy(),
                col_name="day_of_week",
                max_val=7
            )
        ], axis=1)

    # Encode month as cyclic feature using sin/cos transformation
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
) -> pd.DataFrame:
    """
        Encode cyclic features using sin and cos transformations.

        Parameters
        ---
        values: Array of values to encode
        col_name: Base name for the output columns
        max_val: Maximum value for the cyclic feature (used for normalization)

        Returns
        ---
        pd.DataFrame: DataFrame with sin and cos encoded features
    """
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
    """
        Calculate price returns using specified method.

        Parameters
        ---
        price: Series of price values
        period: Number of periods to calculate returns over
        method: Method for calculating returns:
            - "momentum": price[t] / price[t-period]
            - "pct_change": percentage change
            - "price_change": absolute price difference
        log: Whether to apply logarithmic transformation (only for momentum method)

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
        return pd.Series(np.log(returns)) if log else returns - 1
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


def get_lagging_features(
    ser: pd.Series,
    max_lag: int
) -> pd.DataFrame:
    """
        Create lagged versions of a time series.

        Parameters
        ---
        ser: Input time series
        max_lag: Maximum number of lags to create (will create lags 1 to max_lag)

        Returns
        ---
        pd.DataFrame: DataFrame with lagged features

        Raises
        ---
        ValueError: If max_lag is less than 1
    """
    # Validate input parameter
    if max_lag < 1:
        raise ValueError("Parameter `max_lag` must be at least 1.")

    lagging_features = pd.DataFrame()

    # Create lags from 1 to max_lag
    for lag in range(1, max_lag + 1):
        lagging_features[f"{ser.name}_lag{lag}"] = ser.shift(lag)
    return lagging_features


def create_direction_target(
    df: pd.DataFrame,
    price_col_name: str,
    min_step: int = 1,
) -> pd.Series:
    """
        Create directional target variable indicating price movement direction.

        Parameters
        ---
        df: DataFrame containing price data
        price_col_name: Name of the price column to use
        min_step: Minimum number of periods for a price movement to be considered significant

        Returns
        ---
        pd.Series: Binary series indicating price direction (1 for up, 0 for down)
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
    df: pd.DataFrame,
    price_col_name: str
) -> pd.Series:
    """
        Create price-based target variable representing percentage change.

        Parameters
        ---
        df: DataFrame containing price data
        price_col_name: Name of the price column to use

        Returns
        ---
        pd.Series: Series of percentage price changes
    """
    price = df[price_col_name]
    # Calculate price difference between current and next period
    diff = price - price.shift(-1)
    # Calculate percentage change
    target = (diff / price.shift(-1)).rename("target")
    return target
