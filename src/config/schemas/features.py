from pydantic import BaseModel
from typing import Optional, Dict, List


class IndicatorConfig(BaseModel):
    window: int
    price:  str = "close"

class RSIConfig(IndicatorConfig):
    pass

class MACDConfig(IndicatorConfig):
    short_window:  int
    long_window:   int
    signal_window: int

class FeatureConfig(BaseModel):
    time_features: List[str]
    indicators:    Dict[str, Dict]
    lags:          Optional[int] = None



# Конфигурация по умолчанию для данных
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
    "time_features": ["minute", "hour", "day", "day_of_week"],
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
        "DC": {
            "window": 20,
            "output": ["LowDCR", "UpDCR"]
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
    "apply_pca": False
}