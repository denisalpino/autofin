from enum import Enum
from typing import List, Literal

from pydantic import BaseModel, Field

from .features import ColumnSource


class MovingAverageType(str, Enum):
    SMA = "sma"
    EMA = "ema"
    RMA = "rma"
    WMA = "wma"

# Базовый класс для всех конфигураций индикаторов
class IndicatorConfig(BaseModel):
    name: str

class RSIConfig(IndicatorConfig):
    """Relative Strength Index - momentum oscillator"""
    name: Literal["RSI"] = "RSI"
    price: ColumnSource = Field(default=ColumnSource.CLOSE)
    window: int = Field(default=14, ge=1, le=100)

class BBandsConfig(IndicatorConfig):
    """Bollinger Bands - volatility bands around a moving average"""
    name: Literal["BBP"] = "BBP"
    window: int = Field(default=14, ge=1, le=100)
    std: float = Field(default=2.0, ge=0.5, le=4.0)
    mamode: MovingAverageType = Field(default=MovingAverageType.EMA)
    output: List[str] = Field(default_factory=lambda: ["BBB", "BBP"])

class MAConfig(IndicatorConfig):
    """Moving Average (including few kinds) - smooths price data"""
    name: Literal["MA"] = "MA"
    price: ColumnSource = Field(default=ColumnSource.CLOSE)
    window: int = Field(default=14, ge=1, le=100)
    mamode: MovingAverageType = Field(default=MovingAverageType.EMA)

class MACDConfig(IndicatorConfig):
    """Moving Average Convergence Divergence (including signal and histogram) Trend-following momentum indicator"""
    name: Literal["MACD"] = "MACD"
    price: ColumnSource = Field(default=ColumnSource.CLOSE)
    short_window: int = Field(default=12, ge=1, le=100)
    long_window: int = Field(default=26, ge=1, le=100)
    signal_window: int = Field(default=9, ge=1, le=100)
    output: List[str] = Field(default_factory=lambda: ["MACD", "MACD_Signal", "MACD_Hist"])

class ATRConfig(IndicatorConfig):
    """Average True Range - measures market volatility"""
    name: Literal["ATR"] = "ATR"
    window: int = Field(default=14, ge=1, le=100)
    mamode: MovingAverageType = Field(default=MovingAverageType.RMA)

class ADXConfig(IndicatorConfig):
    """Average Directional Index - measures trend strength"""
    name: Literal["ADX"] = "ADX"
    base_window: int = Field(default=14, ge=1, le=100)
    signal_window: int = Field(default=14, ge=1, le=100)
    mamode: MovingAverageType = Field(default=MovingAverageType.RMA)
    output: List[str] = Field(default_factory=lambda: ["ADX", "+DI", "-DI"])

class DCConfig(IndicatorConfig):
    """Donchian Channels - volatility indicator showing highest high and lowest low"""
    name: Literal["DC"] = "DC"
    window: int = Field(default=20, ge=1, le=100)
    output: List[str] = Field(default_factory=lambda: ["LowDC", "UpDC"])

class ERPConfig(IndicatorConfig):
    """Elder Ray Power - measures buying and selling pressure"""
    name: Literal["ERP"] = "ERP"
    window: int = Field(default=13, ge=1, le=100)

class MIConfig(IndicatorConfig):
    """Mass Index (MI) - identifies trend reversals by measuring volatility"""
    name: Literal["MI"] = "MI"
    short_window: int = Field(default=9, ge=1, le=100)
    long_window: int = Field(default=25, ge=1, le=100)

class VRConfig(IndicatorConfig):
    """Volatility Ratio"""
    name: Literal["VR"] = "VR"

class CSConfig(IndicatorConfig):
    """Candle Strength"""
    name: Literal["CS"] = "CS"

class BodyConfig(IndicatorConfig):
    """Candle Body"""
    name: Literal["BODY"] = "BODY"

class UpWConfig(IndicatorConfig):
    """Upper Wick"""
    name: Literal["UpW"] = "UpW"

class LowWConfig(IndicatorConfig):
    """Lower Wick"""
    name: Literal["LowW"] = "LowW"

class PtoMAConfig(IndicatorConfig):
    """Differance between price and Moving Average"""
    name: Literal["PtoMA"] = "PtoMA"
    price: ColumnSource = Field(default=ColumnSource.CLOSE)
    window: int = Field(default=20, ge=1, le=100)
    mamode: MovingAverageType = Field(default=MovingAverageType.EMA)

class MomentumConfig(IndicatorConfig):
    """Momentum (absolute differance close-to-close)"""
    name: Literal["MOM"] = "MOM"
    window: int = Field(default=10, ge=1, le=100)

class ZScoreConfig(IndicatorConfig):
    """Z-Score close - shows how many standard deviations close is from mean"""
    name: Literal["ZScore"] = "ZScore"
    window: int = Field(default=10, ge=1, le=100)

class KCConfig(IndicatorConfig):
    """Keltner Channels - volatility-based envelopes around moving average"""
    name: Literal["KC"] = "KC"
    window: int = Field(default=14, ge=1, le=100)
    scalar: float = Field(default=1.0, ge=0.5, le=100)
    output: List[str] = Field(default_factory=lambda: ["LowKC", "MidKC", "UpKC"])

class RetSkewConfig(IndicatorConfig):
    """Skew of absolute returns per rolling window - measures return distribution asymmetry"""
    name: Literal["RetSkew"] = "RetSkew"
    window: int = Field(default=14, ge=1, le=100)

class RetKurtConfig(IndicatorConfig):
    """Kurtosis of absolute returns per rolling window - measures tail heaviness of return distribution"""
    name: Literal["RetKurt"] = "RetKurt"
    window: int = Field(default=14, ge=1, le=100)

class RetStdConfig(IndicatorConfig):
    """Standart deviation of absolute returns per rolling window - measures volatility"""
    name: Literal["RetStd"] = "RetStd"
    window: int = Field(default=14, ge=1, le=100)

class RetMeanConfig(IndicatorConfig):
    """Mean of absolute returns per rolling window - measures average return"""
    name: Literal["RetMean"] = "RetMean"
    window: int = Field(default=14, ge=1, le=100)
