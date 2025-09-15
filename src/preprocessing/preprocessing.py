import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    accuracy_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    auc,
    log_loss,
    r2_score
)
import xgboost as xgb

from dataclasses import dataclass
from fractions import Fraction
from typing import (
    Callable, Optional, Literal,
    Set, List,
    Iterable, Sequence, Tuple
)

from src.data.splitters.cross_validation import GroupTimeSeriesSplit
from preprocessing.data_loading import load_data, align_by_timestamps
from src.preprocessing.feature_engineering import create_features, create_direction_target, create_price_target

scaling = {
    "method": "minmax",
    "non_scalable_cols": {"ticker"},
    "include_target": True
}

NON_SCALABLE_COLS = {
    "target", "ticker", "minute_sin", "minute_cos",
    "day_sin", "day_cos", "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos", "hour_sin", "hour_cos",
    "direction"
}

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
        "close": "close",
        "timestamps": "timestamps"
    },
    "ohlc": {
        "open": True,
        "high": True,
        "low": True,
        "close": True,
    },
    "time_features": ["minute", "hour", "day", "day_of_week"],  # "month"
    "lags": None,
    "returns": {
        "column": "close",
        "method": "momentum",
        "period": 1,
        "log": True
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
            # "MACD_Hist" one more option
            "output": ["MACD", "MACD_Signal", "MACD_Hist"]
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
        "DC": {  # Donchain Channel Range (based on range between Donchain Channels with current Close price)
            "window": 20,
            "output": ["LowDCR", "UpDCR"]  # "MidDCR" - one more option
        },
        "ERP": {
            "window": 13
        },
        "MI": {
            "short_window": 9,
            "long_window": 25
        },
    },
    "labels_encoding": None,
    "apply_pca": True
}

CV_CONFIG = {
    "val_folds": 5,
    "test_folds": 0,
    "interval": "1d",
    "window": "expanding"
}


def pipeline(
    df: Optional[pd.DataFrame] = None,  # type: ignore
    file: Optional[str] = None,
    dir: Optional[str] = None,
    loader_config: dict = {},
    task: Literal["direction", "returns"] = "returns",
    features_config: dict = DEFAULT_FUTURES_CONFIG,
    align_timestamps: bool = True,
    split: Optional[Sequence[int]] = None, # (80, 10, 10),
    cv_config: Optional[dict] = CV_CONFIG,
    winsorize_percent: Optional[int] = 1,
    scaling_method: Optional[Literal["minmax", "standard", "robust"]] = None,
    non_scalable_cols: Set[str] = NON_SCALABLE_COLS,
    include_target: bool = True,
    save_features_dir: Optional[str] = None,
    save_ready_data_dir: Optional[str] = None,
    return_raw_data: bool = False,
    return_features: bool = False,
    params: dict = {},
    metric: Optional[Callable] = None,
    arch: Literal["multi", "solo"] = "multi"
):
    DEFAULT_FUTURES_CONFIG.update(features_config)
    features_config = DEFAULT_FUTURES_CONFIG

    NON_SCALABLE_COLS.update(non_scalable_cols)
    non_scalable_cols = NON_SCALABLE_COLS

    dataset = preprocessing_pipe(
        df=df,
        file=file,
        dir=dir,
        loader_config=loader_config,
        task=task,
        features_config=features_config,
        align_timestamps=align_timestamps,
        split=split,
        cv_config=cv_config,
        save_features_dir=save_features_dir,
        save_ready_data_dir=save_ready_data_dir,
        return_raw_data=return_raw_data,
        return_features=return_features,
    )
    print("Features engineered")

    result = training_pipe(
        dataset=dataset,
        task=task,
        scaling_method=scaling_method,
        non_scalable_cols=non_scalable_cols,
        include_target=include_target,
        winsorize_percent=winsorize_percent,
        cv_config=cv_config,
        params=params,
        metric=metric,
        arch=arch,
        ts_col=features_config["base_columns"]["timestamps"]
    )
    print("Model trained")
    return result, dataset


def preprocessing_pipe(
        df: Optional[pd.DataFrame] = None,  # type: ignore
        file: Optional[str] = None,
        dir: Optional[str] = None,
        loader_config: dict = {},
        task: Literal["direction", "returns"] = "returns",
        features_config: dict = DEFAULT_FUTURES_CONFIG,
        align_timestamps: bool = True,
        split: Optional[Sequence[int]] = None, # (80, 10, 10),
        cv_config: Optional[dict] = CV_CONFIG,
        save_features_dir: Optional[str] = None,
        save_ready_data_dir: Optional[str] = None,
        return_raw_data: bool = False,
        return_features: bool = False
) -> Dataset:
    '''
    High-level function for preprocessing pipeline configuration.

    Parameters:
    ---
    df : Optional[pd.DataFrame]
        Input DataFrame containing raw data. If None, data will be loaded from file or directory.
    file : Optional[str]
        Path to a single file containing data.
    dir : Optional[str]
        Path to a directory containing multiple files with data.
    loader_config : dict
        Configuration dictionary for loading data (e.g., delimiter, encoding).
    task : Literal["direction", "returns"]
        Specifies the type of task: "direction" for classification or "returns" for regression.
    features_config : dict
        Configuration dictionary for feature engineering, including indicators, lags, and time features.
    align_by_timestamps : bool
        Whether to align all tickers by timestamps.
    split : Sequence[int]
        Percentage split for train, validation, and test sets (e.g., [80, 10, 10]).
    use_cv : bool
        Whether to use cross-validation during preprocessing.
    return_raw_data : bool
        Whether to return raw data alongside processed features.
    return_features : bool
        Whether to return only the engineered features.

    Returns:
    ---
    Dataset
        A dataclass containing:
        - raw_features: DataFrame with all engineered features.
        - train: DataFrame for training data (if split is applied).
        - val: DataFrame for validation data (if split is applied).
        - test: DataFrame for test data (if split is applied).
    '''
    merged_df = pd.DataFrame()
    merged_features = pd.DataFrame()

    # Load data from single file or directory
    dfs = load_data(file, dir, loader_config)

    ts_col = features_config["base_columns"]["timestamps"]

    # Align all tickers by timestamps if needed
    if align_timestamps:
        dfs = align_by_timestamps(dfs, ts_col)

    for df in dfs:
        # Converting timestamps from string to the DateTime object and then sortin
        df[ts_col] = pd.to_datetime(df[ts_col])
        df.sort_values(by=ts_col, ignore_index=True, inplace=True)

        # Create features
        features = create_features(df, features_config=features_config)
        features[ts_col] = pd.to_datetime(df[ts_col])
        features["ticker"] = df["ticker"].astype("category")

        # Create target
        if task == "direction":
            features["target"] = create_direction_target(
                df, price_col_name="close")
            df["target"] = features["target"]
        elif task == "returns":
            features["target"] = create_price_target(
                df, price_col_name="close")
            df["target"] = features["target"]
        else:
            raise ValueError(
                "Unknown type of task. Choose one of the next tasks: 'direction' or 'returns'")

        if features_config["direction_submodel"]:
            features["direction"] = create_direction_target(
                df, price_col_name="close")

        # Drop NA values
        mask = ~features.isna().any(axis=1).values  # type: ignore
        features, df = features[mask], df[mask]

        # Merge full df containing features with all tickers df's
        df = pd.concat([df, features], axis=1)
        merged_df = pd.concat([merged_df, df], axis=0, ignore_index=True)
        merged_features = pd.concat(
            [merged_features, features], axis=0, ignore_index=True)

    # TODO: Develop saving options
    # Sorting by timestamps and ticker names, then drop timestamps cause time features already have been extracted
    merged_features = merged_features.sort_values(
        by=[ts_col, "ticker"], ignore_index=True)

    # Change type of ticker column to the category
    merged_features["ticker"] = merged_features["ticker"].astype("category")

    dataset = Dataset(raw_features=merged_features)

    # Compute train/val/test splits if split parameter is provided.
    # Here we only keep the row indices.
    if split is not None:
        train_df, val_df, test_df = train_val_test_split(merged_features, split)
        dataset.train = train_df.index.to_numpy()
        dataset.val   = val_df.index.to_numpy()
        dataset.test  = test_df.index.to_numpy()
    # If using cross-validation, precompute the splits and store them in the dataset
    elif cv_config:
        # Fetch groups and timestamps
        ts = merged_features[ts_col]
        groups = merged_features["ticker"]
        unique_groups = groups.unique()

        # Get sampler instance and then split features
        gtcv = GroupTimeSeriesSplit(
            val_folds=cv_config["val_folds"],
            test_folds=cv_config["test_folds"],
            interval=cv_config["interval"],
            window=cv_config["window"]
        )
        train_val, train_test = gtcv.split(
            X=merged_features.drop(columns=["target"]),
            y=merged_features["target"],
            groups=groups,
            timestamps=ts
        )\
        # Here can be potential error reelated with that we can don't use
        # validation or test samples at all, but `reshape()` require
        # (len(unique_groups) * cv_config["___folds"] * 2) elements in the array
        train_val  = np.reshape(train_val, (len(unique_groups), cv_config["val_folds"], 2))
        train_test = np.reshape(train_test, (len(unique_groups), cv_config["test_folds"], 2))

        train_val  = dict(zip(unique_groups, train_val))
        train_test = dict(zip(unique_groups, train_test))

        cv_splits = (train_val, train_test)
        dataset.cv_splits = cv_splits
    else:
        raise ValueError("Either `split` or `cv_config` must be provided.")

    return dataset


# TODO: Transfer training pipeline
def training_pipe(
        dataset: Dataset,
        task: Literal["direction", "returns"] = "returns",
        scaling_method: Optional[Literal["minmax", "standard", "robust"]] = "standard",
        non_scalable_cols: Set[str] = NON_SCALABLE_COLS,
        include_target: bool = True,
        winsorize_percent: Optional[int] = 1,
        cv_config: Optional[dict] = None,
        params: dict = {},
        metric: Optional[Callable] = None,
        arch: Literal["multi", "solo"] = "multi",
        ts_col: str = "timestamp"
):
    scalers = {}
    X = dataset.raw_features.drop(columns=["target"]).copy()
    y = dataset.raw_features["target"].copy()

    # FIRST CASE: scaling without CV (simple train/val/test split for time series)
    if dataset.train and dataset.val and dataset.test:
        # Prepare features and target
        X_train = X.iloc[dataset.train]
        X_val   = X.iloc[dataset.val]
        X_test  = X.iloc[dataset.test]
        y_train = y.iloc[dataset.train]
        y_val   = y.iloc[dataset.val]
        y_test  = y.iloc[dataset.test]

        # Winsorize target by quantiles on train if needed
        if winsorize_percent:
            winsorize_frac = winsorize_percent / 200
            low, high = np.quantile(y_train, [winsorize_frac, 1 - winsorize_frac])
            y_train   = pd.Series(np.clip(y_train, low, high))
            y_val     = pd.Series(np.clip(y_val, low, high))
            y_test    = pd.Series(np.clip(y_test, low, high))

        if scaling_method:
            # Select scalers
            X_scaler = SCALERS[scaling_method]()
            y_scaler = SCALERS[scaling_method]()

            # Select scalable columns
            scalable = get_scalable_cols(X.columns, non_scalable_cols)

            # Fit scaler only on X_train and transform both samples of features
            X_scaler          = SCALERS[scaling_method]().fit(X_train[scalable])
            X_train[scalable] = X_scaler.transform(X_train[scalable])
            X_val[scalable]   = X_scaler.transform(X_val[scalable])
            X_test[scalable]  = X_scaler.transform(X_test[scalable])

            # Fit scaler only on y_train and transform both samples of targets (optional)
            if include_target and task != "direction":
                y_scaler = SCALERS[scaling_method]().fit(y_train.to_numpy().reshape(-1, 1))
                y_train  = y_scaler.transform(y_train.to_numpy().reshape(-1, 1)).flatten()
                y_val    = y_scaler.transform(y_val.to_numpy().reshape(-1, 1)).flatten()
                y_test   = y_scaler.transform( y_test.to_numpy().reshape(-1, 1)).flatten()
            elif include_target:
                raise ValueError("Scaling target detected during 'direction' task. Turn of target scaling or change type of task.")

        return (None, None, None, None, None, None)
    elif dataset.cv_splits is not None:
        # Prepare features, target and groups
        train_val, train_test = dataset.cv_splits

        metrics = None
        Xs, ys = [], []
        models = []
        scalers = []
        preds = pd.DataFrame()

        for g_name, group_split in train_val.items():
            group_metrics = []
            for fold, (train_idx, val_idx) in enumerate(group_split, start=1):
                if len(val_idx) > 0:
                    sclr = []
                    # Sampling by inexies
                    X_train = X.iloc[train_idx].reset_index(drop=True)
                    X_val   = X.iloc[val_idx].reset_index(drop=True)
                    y_train = y.iloc[train_idx].reset_index(drop=True)
                    y_val   = y.iloc[val_idx].reset_index(drop=True)

                    X_train = X_train.drop(columns=[ts_col])
                    X_val = X_val.drop(columns=[ts_col])

                    # NOTE: Think about winsorize lags of log-returns too
                    # Winsorize target by quantiles on train if needed
                    if winsorize_percent:
                        winsorize_frac = winsorize_percent / 200
                        low, high = np.quantile(y_train, [winsorize_frac, 1 - winsorize_frac])
                        y_train   = pd.Series(np.clip(y_train, low, high))
                        y_val     = pd.Series(np.clip(y_val, low, high))

                    if scaling_method:
                        # Select scalable columns
                        scalable = get_scalable_cols(
                            X_train.columns, non_scalable_cols)

                        for tkr in sorted(X_train.ticker.unique()):
                            idx = X_train[X_train.ticker == tkr].index
                            idx_val = X_val[X_val.ticker == tkr].index

                            # Fit scaler only on X_train and transform both samples of features
                            feature_scaler = SCALERS[scaling_method]().fit(
                                X_train.loc[idx][scalable])
                            X_train.iloc[idx][scalable] = feature_scaler.transform(
                                X_train.iloc[idx][scalable])
                            X_val.iloc[idx_val][scalable] = feature_scaler.transform(
                                X_val.iloc[idx_val][scalable])

                            # Fit scaler only on y_train and transform both samples of targets (optional)
                            if include_target and task != "direction":
                                y_train = pd.Series(y_train)
                                y_val = pd.Series(y_val)
                                target_scaler = SCALERS[scaling_method]().fit(
                                    y_train.iloc[idx].values.reshape(-1, 1))
                                y_train.iloc[idx] = target_scaler.transform(
                                    y_train.iloc[idx].values.reshape(-1, 1)).flatten()
                                y_val.iloc[idx_val] = target_scaler.transform(
                                    y_val.iloc[idx_val].values.reshape(-1, 1)).flatten()
                                sclr.append(target_scaler)
                            elif include_target:
                                raise ValueError("Scaling target detected during 'direction' task. Turn of target scaling or change type of task.")

                    tickers_train = X_train["ticker"]
                    tickers_val = X_val["ticker"]
                    if arch == "solo":
                        X_train.drop(columns=["ticker"], inplace=True)
                        X_val.drop(columns=["ticker"], inplace=True)

                    if task == "returns":
                        model = xgb.XGBRegressor(**params)
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=0
                        )

                        y_pred_train = pd.Series(model.predict(X_train))
                        y_pred_val = pd.Series(model.predict(X_val))

                        preds = pd.concat([preds, y_pred_train, y_pred_val], axis=0, ignore_index=True)
                        Xs.append(X_val)
                        ys.append(y_val)
                        models.append(model)

                        idxs = []
                        for i, tkr in enumerate(sorted(tickers_train.unique())):
                            idx_val = X_val[tickers_val == tkr].index
                            if include_target and task != "direction":
                                y_pred_val.iloc[idx_val] = sclr[i].inverse_transform(y_pred_val.iloc[idx_val].values.reshape(-1, 1)).flatten()
                                y_val.iloc[idx_val] = sclr[i].inverse_transform(y_val.iloc[idx_val].values.reshape(-1, 1)).flatten()
                            idxs.append(idx_val)

                        mse = mean_squared_error(y_val, y_pred_val)
                        mae = np.mean(np.abs(y_val - y_pred_val))
                        mape = mean_absolute_percentage_error(y_val, y_pred_val) * 100
                        mase = mae / np.mean(np.abs(y_val))

                        # if len(y_val.shift(-1)) - y_val.shift(-1).isna().sum() > 2:
                        #    r2_ret = r2_score(((y_val - y_val.shift(-1)) / y_val.shift(-1)).dropna(), ((y_val - y_pred_val.shift(-1)) / y_pred_val.shift(-1)).dropna())
                        # else:
                        #    r2_ret = np.nan
                        # print(f"Returns R2: {r2_ret:5f}")
                        # mase = metric(y_val.values, y_pred_val.values, idxs)
                        r2 = r2_score(y_val, y_pred_val)
                        naive = np.sqrt((y_val**2).mean()) * 100
                        print(f"Naive RMSE: {naive:.5f}")
                        print(f"Fold {fold} — rows train:{len(train_idx)} val:{len(val_idx)} | RMSE: {np.sqrt(mse) * 100:.5f}, MAE: {mae:.5f}, MAPE: {mape:.2f}%, R2: {r2:.5f}, MASE: {mase:.5f}")

                        group_metrics.append({"rmse": np.sqrt(mse), "mae": mae, "r2": r2, "mape": mape, "mase": mase})
                        scalers.append(sclr)
                    else:
                        model = xgb.XGBClassifier(**params)
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=0
                        )

                        y_pred_train = pd.Series(model.predict(X_train))
                        y_pred_val = pd.Series(model.predict(X_val))

                        preds = pd.concat([preds, y_pred_train, y_pred_val], axis=0, ignore_index=True)
                        Xs.append(X_val)
                        ys.append(y_val)
                        models.append(model)

                        # For classification: predict_proba or pred labels
                        proba = model.predict_proba(X_val)[:, 1]
                        precision, recall, _ = precision_recall_curve(y_val, proba)
                        auc_score = auc(recall, precision)
                        recall    = recall_score(y_val, y_pred_val)
                        precision = precision_score(y_val, y_pred_val)
                        ll        = log_loss(y_val, proba)
                        acc       = accuracy_score(y_val, y_pred_val)

                        print(f"Fold {fold} — rows train:{len(train_idx)} val:{len(val_idx)} | LogLoss: {ll:.5f}, Accuracy: {acc:.4f}, AUC: {auc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

                        group_metrics.append({"logloss": ll, "accuracy": acc})
                        scalers.append(sclr)

            if metrics is None:
                metrics = pd.DataFrame(group_metrics)
            else:
                metrics = metrics + pd.DataFrame(group_metrics)

            group_metrics = pd.DataFrame(group_metrics).mean(axis=0).to_dict()
            if task == "returns":
                print(f"Group {g_name} | RMSE: {group_metrics['rmse']:.5f}, R2: {group_metrics['r2']:.5f}, MASE: {group_metrics['mase']:.5f}")
            else:
                print(f"Group {g_name} | LogLoss: {group_metrics['logloss']:.4f}, Accuracy: {group_metrics['accuracy']:.4f}")
            print("----------------------------------------------------------------")
        print()

        metrics = metrics / len(train_val.keys())
        for ind, (_, fold) in enumerate(metrics.iterrows(), start=1):
            if task == "returns":
                print(f"Period {ind} | RMSE: {fold['rmse']:.5f}, R2: {fold['r2']:.5f}, MASE: {fold['mase']:.5f}")
            else:
                print(f"Period {ind} | LogLoss: {fold['logloss']:.4f}, Accuracy: {fold['accuracy']:.4f}")

        return metrics, models, Xs, ys, scalers, preds



def get_good_timestamps(
        merged_df: pd.DataFrame,
        ts_col: str,
        ticker_col: str
) -> pd.Index:
    """Returns the Index of timestamps in which all tickers are present."""
    # Count how many unique tickers there are on each timestamp
    counts = (
        merged_df
        .groupby(ts_col)[ticker_col]
        .nunique()
    )
    total_tickers = merged_df[ticker_col].nunique()
    # Keep only those timestamps with full coverage
    good_ts = counts[counts == total_tickers].index
    return good_ts
