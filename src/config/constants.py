from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

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