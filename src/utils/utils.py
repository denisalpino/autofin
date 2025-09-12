from ast import List
from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Set, Sequence


NON_SCALABLE_COLS = {
    "target", "ticker", "minute_sin", "minute_cos",
    "day_sin", "day_cos", "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos", "hour_sin", "hour_cos"
}

SCALERS = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
    "robust": RobustScaler
}

@dataclass
class DataScaler:
    method: Optional[Literal["minmax", "standard", "robust"]]
    non_scalable_cols: Set[str] = NON_SCALABLE_COLS
    include_target: bool = False


@dataclass
class DataManager:
    file: Optional[str] = None
    dir: Optional[str] = None
    save_features_dir: Optional[str] = None
    save_ready_data_dir: Optional[str] = None
    return_raw_data: bool = False
    return_features: bool = False


@dataclass
class DataSampler:
    split: Sequence[int] = (75, 15, 10)
    use_cv: bool = True
    k_fold: int = 5
    winsorize_percent: Optional[int] = 1


@dataclass
class FeatureExtractor:
    base_columns: Dict
    timestamps: Dict
    returns: Dict
    lags: Dict
    indicators: list


@dataclass
class RSI:
    window: int = 14


@dataclass
class BBands:
    window: int = 14
    std: float = 2.0
    kind: Literal["ema", "rma", "sma", "wma"] = "ema"
    output: list[Literal["BBP", "BBB"]] = ["BBB", "BBP"]


@dataclass
class MA:
    by: Literal["open", "high", "low", "close", "volume"] = "close"
    window: int = 14
    kind: Literal["ema", "rma", "sma", "wma"] = "ema"


@dataclass
class MACD:
    by: Literal["open", "high", "low", "close"] = "close"
    short_window: int = 14
    long_window: int = 26
    signal_window: int = 9
    output: list[Literal["MACD", "MACD_Signal", "MACD_Hist"]] = ["MACD", "MACD_Signal", "MACD_Hist"]


@dataclass
class ATR:
    window: int = 14
    kind: Literal["ema", "rma", "sma", "wma"] = "rma"


@dataclass
class ADX:
    base_window: int = 14
    signal_window: int = 14
    output: list[Literal["ADX", "+DI", "-DI"]] = ["ADX", "+DI", "-DI"]


@dataclass
class DonchainChannel:
    window: int = 20
    output: list[Literal["LowDCR", "UpDCR"]] = ["LowDCR", "UpDCR"]


@dataclass
class ERP:
    window: int = 14


@dataclass
class MI:
    short_window: int = 9
    long_window: int = 25


class Pipeline:
    ...
    def __init__(self) -> None:
        pass

    ...