import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pandas_ta as indicators
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from mlxtend.evaluate import GroupTimeSeriesSplit

import os
import joblib
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Sequence, Literal, Set

scaling = {
    "method": "minmax",
    "non_scalable_cols": {"ticker"},
    "include_target": True
}
NON_SCALABLE_COLS = {"target", "ticker"}

SCALERS = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
    "robust": RobustScaler
}

DEFAULT_FUTURES_CONFIG = {
    "base_columns": {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close"
    },
    "timestamps": {
        "column": "timestamp",
        "features": ["minute", "hour", "day", "day_of_week"] #  "month"
    },
    "lags": {
        "returns": 5
    },
    "returns": {
        "column": "close",
        "method": "pct_change",
        "period": 1,
        "log": False
    },
    "indicators": {
        "RSI": {
            "price": "close",
            "window": 14
        },
        "BBP": {
            "window": 14,
            "std": 2,
            "kind": "ema",
            "output": ["BBB", "BBP"]
        },
        "MA_1": {
            "price": "close",
            "kind": "ema",
            "window": 4
        },
        "MA_2": {
            "price": "close",
            "kind": "ema",
            "window": 8
        },
        "MA_3": {
            "price": "close",
            "kind": "ema",
            "window": 16
        },
        "MA_4": {
            "price": "close",
            "kind": "ema",
            "window": 48
        },
        "MA_5": {
            "price": "close",
            "kind": "ema",
            "window": 96
        },
        "MACD": {
            "price": "close",
            "short_window": 12,
            "long_window": 26,
            "signal_window": 9,
            "output": ["MACD", "MACD_Signal", "MACD_Hist"] # "MACD_Hist" one more option
        },
        "ATR": {
            "window": 14,
            "kind": "rma",
        },
        "ADX": {
            "base_window": 14,
            "signal_window": 14,
            "kind": "rma",
            "output": ["ADX", "+DI", "-DI"]
        },
        "DC": { # Donchain Channel Range (based on range between Donchain Channels with current Close price)
            "window": 20,
            "output": ["LowDCR", "UpDCR"] # "MidDCR" - one more option
        },
        "ERP": {
            "window": 13
        },
        "MI": {
            "short_window": 9,
            "long_window": 25
        },
    },
    "labels_encoding": None
}

@dataclass
class Dataset:
    raw: Optional[pd.DataFrame]
    features: Optional[pd.DataFrame]
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    scalers: dict


def preprocessing_pipe(
        df: Optional[pd.DataFrame] = None, # type: ignore
        file: Optional[str] = None,
        dir: Optional[str] = None,
        loader_config: dict = {},
        task: Literal["direction", "returns"] = "returns",
        features_config: dict = DEFAULT_FUTURES_CONFIG,
        split: Sequence[int] = (75, 15, 10),
        use_cv: bool = True,
        winsorize_percent: Optional[int] = 1,
        scaling_method: Optional[Literal["minmax", "standard", "robust"]] = None,
        non_scalable_cols: Set[str] = NON_SCALABLE_COLS,
        include_target: bool = True,
        save_features_dir: Optional[str] = None,
        save_ready_data_dir: Optional[str] = None,
        return_raw_data: bool = False,
        return_features: bool = False,
) -> Dataset:
    """
    High level function for preprocessing pipeline configuration.
    """
    merged_df = pd.DataFrame()

    if use_cv:
        train_for_cv = pd.DataFrame()
        test_for_cv  = pd.DataFrame()
    else:
        merged_train = pd.DataFrame()
        merged_val   = pd.DataFrame()
        merged_test  = pd.DataFrame()
    scalers = {}

    DEFAULT_FUTURES_CONFIG.update(features_config)
    features_config = DEFAULT_FUTURES_CONFIG

    NON_SCALABLE_COLS.update(non_scalable_cols)
    non_scalable_cols = NON_SCALABLE_COLS

    # Select scaler for target and features if scaling is needed
    if scaling_method and use_cv:
        feature_scaler = SCALERS[scaling_method]()
        target_scaler  = SCALERS[scaling_method]()

    # Load data from single file or directory
    dfs = load_data(file, dir, loader_config)

    for df in dfs:
        # Converting timestamps from string to the DateTime object and then sorting
        timestamps_col     = features_config["timestamps"]["column"]
        df[timestamps_col] = pd.to_datetime(df[timestamps_col])
        df.sort_values(by=timestamps_col, ignore_index=True, inplace=True)

        # Create features
        features           = create_features(df, features_config=features_config)
        features["ticker"] = df["ticker"].astype("category")

        # Create target
        if task == "direction":
            features["target"] = create_direction_target(df, price_col_name="close")
            df["target"]       = features["target"]
        elif task == "returns":
            features["target"] = create_price_target(df, price_col_name="close")
            df["target"]       = features["target"]
        else:
            raise ValueError("Unknown type of task. Choose one of the next tasks: 'direction' or 'returns'")

        # Drop NA values
        mask         = ~features.isna().any(axis=1).values # type: ignore
        features, df = features[mask], df[mask]

        # Merge full df containing features with all tickers df's
        df              = pd.concat([df, features], axis=1)
        merged_df       = pd.concat([merged_df, df], axis=0, ignore_index=True)
        merged_features = pd.concat([merged_features, features], axis=0, ignore_index=True)

        # TODO: There is no alignment applies if tickers have unequal timings
        # Split dataset to the train / validation / test samples
        train, val, test = train_val_test_split(features, split)

        if use_cv:
            train_for_cv = pd.concat([train_for_cv, train, val], axis=0)
            test_for_cv  = pd.concat([test_for_cv, test], axis=0)
        else:
            merged_train = pd.concat([merged_train, train], axis=0, ignore_index=True)
            merged_val   = pd.concat([merged_val, val], axis=0, ignore_index=True)
            merged_test  = pd.concat([merged_test, test], axis=0, ignore_index=True)

    if not use_cv and scaling_method:
        # Select scalers
        feature_scaler = SCALERS[scaling_method]()
        target_scaler  = SCALERS[scaling_method]()

        # Select scalable columns
        others   = []
        scalable = []
        for col in features.columns:
            condition_starts = not any([col.startswith(nonsc) for nonsc in non_scalable_cols])
            condition_ends = not (col.endswith("_cos") or col.endswith("_sin"))
            (others, scalable)[condition_starts and condition_ends].append(col)

        # Scale features and target (optionally)
        for ind, merged_sample in enumerate([merged_train, merged_val, merged_test]):
            if ind == 0:
                # Fit scalers only on train features
                feature_scaler = feature_scaler.fit(merged_sample[scalable])
                scalers["features"] = feature_scaler
                # Fit scalers only on train features target (optionally)
                if include_target and task != "direction":
                    target_array  = merged_sample["target"].values.reshape(-1, 1)
                    target_scaler = target_scaler.fit(target_array)
                elif include_target:
                    raise ValueError("Scaling target detected during 'direction' task. Turn of target scaling or change type of task.")
            # Scale validation and train samples
            merged_sample.loc[:, scalable] = feature_scaler.transform(merged_sample[scalable])
            if include_target:
                target_array            = merged_sample["target"].values.reshape(-1, 1)
                merged_sample["target"] = target_scaler.transform(target_array).flatten()
    elif use_cv:
        # Calculate ratio of training and validating samples for cross-validation
        train_size, val_size, _ = split

        ratio = train_size / val_size
        frac  = Fraction(ratio).limit_denominator(100)
        train_size, val_size = frac.numerator, frac.denominator

        # Prepare features, target and groups
        X = train_for_cv.drop(columns=["target"])
        y = train_for_cv["target"]
        groups = train_for_cv["ticker"]

        gtscv = GroupTimeSeriesSplit(val_size, train_size=train_size, n_splits=5)

        for fold, (train_idx, val_idx) in enumerate(gtscv.split(X, y, groups=groups)):
            # Sampling by inexies
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # TODO: Think about winsorize lags of log-returns too
            # 2) Winsorize target by quantiles on train if needed
            if winsorize_percent:
                winsorize_percent: float = winsorize_percent / 200
                low, high    = np.quantile(y_train, [winsorize_percent, 1 - winsorize_percent])
                y_train = np.clip(y_train, low, high)
                y_val   = np.clip(y_val,   low, high)

            # 3) Фитим StandardScaler только на X_train и y_train_clip
            numeric_feats = [c for c in X.columns if c != "ticker"]
            feat_scaler   = StandardScaler().fit(X_train[numeric_feats])
            targ_scaler   = StandardScaler().fit(y_train.values.reshape(-1,1))

            # 4) Трансформируем train и val
            X_train_scaled = X_train.copy()
            X_val_scaled   = X_val.copy()
            X_train_scaled[numeric_feats] = feat_scaler.transform(X_train[numeric_feats])
            X_val_scaled[numeric_feats]   = feat_scaler.transform(X_val[numeric_feats])

            y_train_scaled = targ_scaler.transform(y_train.values.reshape(-1,1)).ravel()
            y_val_scaled   = targ_scaler.transform(y_val.values.reshape(-1,1)).ravel()




    dataset = Dataset(
        raw=merged_df if return_raw_data else None, # type: ignore
        features=merged_features if return_features else None, # type: ignore
        train=merged_train, # type: ignore
        val=merged_val, # type: ignore
        test=merged_test, # type: ignore
        scalers=scalers
    )
    return dataset

def train_val_test_split(
        features: pd.DataFrame,
        split: Sequence[int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Validate params
    if (curr_sum := sum(split)) != 1:
        raise ValueError(f"All elements of the `split` sequence must summing to 1, when current sum is {curr_sum}.")

    # Convert to shares
    train_size, val_size, test_size = [i / 100 for i in split]

    # Calaculate number of training and validating samples
    num_trainable = int(len(features) * train_size)
    num_validatable = int((len(features) - num_trainable) * (val_size / (val_size + test_size)))

    # Make slices for each set
    trainable = features.iloc[:num_trainable]
    validatable = features.iloc[num_trainable:num_trainable + num_validatable]
    testable = features.iloc[num_trainable + num_validatable:]

    return trainable, validatable, testable


def load_data(
        file: Optional[str] = None,
        dir: Optional[str] = None,
        loader_config: dict = {},
) -> list[pd.DataFrame]:
    """
    Function for loading data from single file or all directory with assigning
    files in the new column named 'ticker' by symbols before the first '_' (underline).
    """
    dfs = []

    if file:
        df = pd.read_csv(file, **loader_config)

        # Assign ticker to the new column
        ticker_name = file.split("_")[0]
        df["ticker"] = ticker_name
        df["ticker"] = pd.Categorical(df["ticker"])

        dfs.append(df)
    elif dir:
        file_names = os.listdir(dir)
        dir = dir if dir.endswith("/") else dir + "/"

        for file_name in file_names:
            df = pd.read_csv(dir + file_name, **loader_config)

            # Assign ticker to the new column
            ticker_name = file_name.split("_")[0]
            df["ticker"] = ticker_name
            df["ticker"] = pd.Categorical(df["ticker"])

            dfs.append(df)
    else:
        raise ValueError("Need to directory or file with data.")
    return dfs


def create_direction_target(
        df: pd.DataFrame,
        price_col_name: str,
        min_step: int = 1,
) -> pd.Series:
    """
    :param int, default = 1 min_step:
        minimal step for differance calculation. If there is no differance search for first next
        item with changed value
    """
    s = df[price_col_name]

    # Определяем ранги одинаковых значений
    run_id = (s != s.shift(min_step)).cumsum()
    # Для каждого run_id берём первую цену
    run_first = s.groupby(run_id).first()
    # Вычисляем цену следующего отличающегося ранга
    next_run_first = run_first.shift(-1)
    # Сравниваем: 1, если next > current, 0 если <, и NA если next отсутствует
    directions = (next_run_first > run_first).astype("Int8")
    # Разворачиваем обратно: каждому элементу в ранге присваиваем его direction
    target = run_id.map(directions).astype("Int8")
    target.name = "target"

    return target


def create_price_target(
        df: pd.DataFrame,
        price_col_name: str
) -> pd.Series:
    price = df[price_col_name]
    target = np.log(price / price.shift(1)).rename("target") # type: ignore
    return target


def create_features(
        df: pd.DataFrame,
        indicators: Optional[list[str]] = None,
        features_config: dict = {}
) -> pd.DataFrame:
    df = df.copy()
    features = pd.DataFrame()

    # Time-based feature engineering
    if timestamps := features_config["timestamps"]:
        # Configs unpacking
        column = timestamps["column"]
        params = timestamps["features"]
        boolean_params = [
            param in params
            for param in ["minute", "hour", "day", "day_of_week", "month"]
        ]
        # Calculate time-based features and concatenate with others
        time_features = create_time_features(df[column], *boolean_params)
        features = pd.concat([features, time_features], axis=1)

    # Returns feature engineering
    if returns := features_config["returns"]:
        # Configs unpacking
        column = returns["column"]
        period = returns["period"]
        method = returns["method"]
        log = returns["log"]

        # Calculate feature and concatenate with the main feature DataFrame
        returns_feature = calculate_returns(df[column], period=period, method=method, log=log)
        returns_feature.name = "returns"
        features = pd.concat([features, returns_feature], axis=1)

    # Indicator-based feature engineering
    if indicators := features_config["indicators"]:
        if isinstance(indicators, dict):
            indicator_features = pd.DataFrame()

            # For each indicator in indicator configs calculate correspondent values
            for name, config in indicators.items():
                indicator = calculate_indicator(
                    df, name=name, config=config,
                    base_columns=features_config["base_columns"]
                )
                indicator_features = pd.concat([indicator_features, indicator], axis=1)
        else:
            raise TypeError(f"Unsupported object of indicator configuration field: {type(indicators)}")
        # Concatenate indicators with other features
        features = pd.concat([features, indicator_features], axis=1)

    # Lagging feature engineering
    if lags := features_config["lags"]:
        # Configs unpacking
        for col, lag in lags.items():
            # Calculate lagging features and concatenate with others
            lagging_features = get_lagging_features(features[col], max_lag=lag)
            features = pd.concat([features, lagging_features], axis=1)

    return features # type: ignore

# This function is something like a shit code because it restricts any possibility
# of adding new user indicators. Maybe I'll rewrite ogic in the future, but now it's okay
# cause it's a small pet-project
def calculate_indicator(
        df: pd.DataFrame,
        name: str,
        config: dict,
        base_columns: dict
) -> pd.DataFrame | pd.Series:
    """
    Select correspondent function for indicator calculation based on `name` parameter,
    parse configuration for this indicator, insert configs into function, and, then,
    postprocess resulting values adding correspondent name to the Series.

    :param df: base given for preprocessing `pd.DataFrame`
    :param name: short name of current indicator
    :param config: user configurations for correspondent indicator
    :param base_columns: user configurations about names of OHLC-columns

    :return: `pd.Series` | `pd.DataFrame`
    """
    open, high = df[base_columns["open"]], df[base_columns["high"]]
    low, close = df[base_columns["low"]], df[base_columns["close"]]

    if name == "ATR":
        # Configs unpacking
        window = config["window"]
        kind = config["kind"]

        atr = indicators.atr(high, low, close, length=window, mamode=kind)
        return atr # type: ignore
    elif name == "RSI":
        # Configs unpacking
        prices = df[config["price"]]
        window = config["window"]

        rsi = rsi = indicators.rsi(prices, length=window, scalar=1)
        return rsi # type: ignore
    elif name == "BBP":
        # Configs unpacking
        window = config["window"]
        std = config["std"]
        kind = config["kind"]
        output = config["output"]

        bb = indicators.bbands(close, length=window, std=std, mamode=kind).iloc[:, -2:] # type: ignore
        bb.columns = ["BBB", "BBP"]

        return bb[output] # type: ignore
    elif name == "MACD":
        # Configs unpacking
        prices = df[config["price"]]
        short_window = config["short_window"]
        long_window = config["long_window"]
        signal_window = config["signal_window"]
        output = config["output"]

        macd = indicators.macd(prices, fast=short_window, slow=long_window, signal=signal_window) # type: ignore
        macd.columns = ["MACD", "MACD_Signal", "MACD_Hist"] # type: ignore

        return macd[output] # type: ignore
    elif name == "ADX":
        # Configs unpacking
        base_window = config["base_window"]
        signal_window = config["signal_window"]
        kind = config["kind"]
        output = config["output"]

        adx = indicators.adx(high, low, close, length=base_window, lensig=signal_window, mamode=kind, scalar=1)
        adx.columns = ["ADX", "+DI", "-DI"] # type: ignore

        return adx[output] # type: ignore
    elif name == "DC":
        # Configs unpacking
        window = config["window"]
        output = config["output"]

        dc = indicators.donchian(high, low, lower_length=window, upper_length=window)
        dcr = pd.DataFrame()
        for col, ser in dc.items(): # type: ignore
            dcr[col] = (ser / close) - 1
        dcr.columns = ["LowDCR", "MidDCR", "UpDCR"] # type: ignore

        return dcr[output]
    elif name == "ERP": # Elder Ray Power
        # Configs unpacking
        window = config["window"]

        ema = indicators.ema(close, length=window)
        bears_power = high - ema
        bulls_power = low - ema
        bears_power.name = "BePo"
        bulls_power.name = "BuPo"
        return pd.concat([bears_power, bulls_power], axis=1)

        # Calculate and concatenate with others
        bears_power, bulls_power = calculate_elder_ray_power(close, high, low, window=window)
        bears_power.name = "BePo"
        bulls_power.name = "BuPo"
        return pd.concat([bears_power, bulls_power], axis=1)
    elif name == "MI":
        # Configs unpacking
        short_window = config["short_window"]
        long_window = config["long_window"]

        mi = indicators.massi(df.high, df.low, fast=short_window, slow=long_window)
        return mi # type: ignore

        # Calculate and concatenate with others
        mi = calculate_mass_index(high, low, short_window=short_window, long_window=long_window, sum_period=sum_period)
        mi.name = name
        return mi
    elif "MA_" in name:
        # Configs unpacking
        prices = df[config["price"]]
        kind = config["kind"]
        window = config["window"]

        ma = indicators.ma(kind, prices, length=window)
        return ma
    elif name == "volatility_ratio":
        vr = high / low - 1
        vr.name = "volatility_ratio"
        return vr
    elif name == "candle_strength":
        cs = (close - open) / (high - low) # type: ignore
        cs.name = "candle_strength"
        return cs
    raise ValueError(f"Unknown indicator {name} detected")


# Maybe shit code too, but it works nicely
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

    # If function called without any target-feature it will raise an error
    if not any([minute, hour, day, day_of_week, month]):
        raise ValueError(
            "Time feature engineering without any target-feature has been detected.")

    # Use cyclic encoding based on cos and sin functions
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
        # Use `days_in_month` for each item
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
    """Encoding cyclic features basd on `cos` and `sin` math functions"""

    encoded_features = pd.DataFrame()
    encoded_features[col_name + '_sin'] = np.sin(2 * np.pi * values / max_val)
    encoded_features[col_name + '_cos'] = np.cos(2 * np.pi * values / max_val)

    return encoded_features


def calculate_returns(
        price: pd.Series,
        period: int = 1,
        method: Literal["momentum", "pct_change", "price_change"] = "pct_change",
        log=False
) -> pd.Series:
    """
    Docstring
    """

    if method == "momentum":
        returns = price / price.shift(period) # type: ignore
        return pd.Series(np.log(returns)) if log else returns
    elif log:
        raise ValueError(f"Using `log=True` only available with `method='momentum'`.")
    elif method == "pct_change":
        return price.pct_change(period)
    elif method == "price_chenge":
        return price - price.shift(period)
    raise ValueError(f"Unknown method: {method}. Please read docstring of calculate_returns() function.")


def get_lagging_features(
        ser: pd.Series,
        max_lag: int
) -> pd.DataFrame:
    if max_lag < 2:
        raise ValueError("Parameter `max_lag` cannot be less than 2.")

    lagging_features = pd.DataFrame()

    for lag in range(1, max_lag):
        lagging_features[f"{ser.name}_lag{lag}"] = ser.shift(lag)

    return lagging_features


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

class PipelineConfig:
    n_splits: int = 5
    scaler_method: str = 'standard'  # 'standard' or 'robust'
    target_quantile: float = 0.99
    task: str = 'returns'


def preprocess_and_cv(dfs: list[pd.DataFrame], cfg: PipelineConfig):
    # 1. Создание фич и таргета для всех тикеров
    all_feats = []
    all_meta = []  # для ticker и timestamps
    for df in dfs:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        feats = create_features(df)
        # log-return
        df['log_r'] = np.log(df['close'] / df['close'].shift(1))
        feats['target'] = df['log_r']
        feats['ticker'] = df['ticker']
        feats['timestamp'] = df['timestamp']
        all_feats.append(feats.dropna())

    data = pd.concat(all_feats, ignore_index=True)

    # 2. Удаление экстремумов на уровне квантилей
    q = data['target'].quantile(cfg.target_quantile)
    data['target'] = data['target'].clip(lower=-q, upper=q)

    # 3. CV с учётом time series и группы ticker
    gts = GroupTimeSeriesSplit(n_splits=cfg.n_splits)
    splits = []
    for train_idx, test_idx in gts.split(data, groups=data['ticker']):
        splits.append((train_idx, test_idx))

    # 4. Применение масштабирования внутри фолдов
    fold_data = []
    for i, (tr, te) in enumerate(splits):
        df_train = data.iloc[tr].copy()
        df_test = data.iloc[te].copy()

        # Feature scaler глобально
        if cfg.scaler_method == 'standard':
            feat_scaler = StandardScaler()
        else:
            feat_scaler = RobustScaler()
        X_cols = [c for c in data.columns if c not in ['ticker', 'timestamp', 'target']]
        feat_scaler.fit(df_train[X_cols])
        df_train[X_cols] = feat_scaler.transform(df_train[X_cols])
        df_test[X_cols] = feat_scaler.transform(df_test[X_cols])

        # Target scaler по тикеру
        target_scalers = {}
        for t in df_train['ticker'].unique():
            ts = StandardScaler()
            mask = df_train['ticker'] == t
            ts.fit(df_train.loc[mask, ['target']])
            target_scalers[t] = ts
            df_train.loc[mask, 'target'] = ts.transform(df_train.loc[mask, ['target']])
        # Применяем тот же scaler на тесте
        for t, ts in target_scalers.items():
            mask = df_test['ticker'] == t
            df_test.loc[mask, 'target'] = ts.transform(df_test.loc[mask, ['target']])

        fold_data.append((df_train, df_test))

    return fold_data

# Пример вызова:
# cfg = PipelineConfig()
# dfs = load_all_dfs(...)  # список df по тикерам
# folds = preprocess_and_cv(dfs, cfg)
