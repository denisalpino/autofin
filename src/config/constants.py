from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from src.config.schemas.indicators import *


BASE_INDICATORS = [
    RSIConfig(),
    MACDConfig(),
    BBandsConfig(),
    MAConfig(window=4),
    MAConfig(window=8),
    MAConfig(window=16),
    MAConfig(window=48),
    MAConfig(window=96),
    ATRConfig(),
    ADXConfig(),
    DCConfig(),
    ERPConfig(),
    MIConfig()
]

SCALERS = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
    "robust": RobustScaler
}

NON_SCALABLE_COLS = {
    "target", "ticker", "minute_sin", "minute_cos",
    "day_sin", "day_cos", "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos", "hour_sin", "hour_cos", "direction"
}