from pandas import DataFrame, concat

from src.features.calculators.indicators import calculate_indicator
from src.features.calculators.returns import calculate_returns
from src.features.calculators.time_features import create_time_features
from src.features.calculators.lagging import get_lagging_features


class FeatureBuilder:
    def __init__(self, config):
        self.config = config

    def build_features(self, df: DataFrame) -> DataFrame:
        """
            Creates feature matrix based on configuration dictionary.

            Parameters
            ---
            df: Input DataFrame with raw financial data
            self.config: Configuration dictionary specifying which features to create
                - base_columns: Dictionary mapping standard names to actual column names
                - ohlc: Dictionary specifying which OHLC prices to include as features
                - time_features: Dictionary specifying which time-based features to create
                - returns: Dictionary configuring returns calculation
                - indicators: Dictionary configuring technical indicator calculations
                - lags: Dictionary specifying lag features to create

            Returns
            ---
            DataFrame: DataFrame with engineered features

            Raises
            ---
            KeyError: If required base_columns are missing from self.config
            TypeError: If indicators configuration is not a dictionary
            ValueError: If time_features are requested but no parameters are specified
        """
        df = df.copy()
        # Store original index to ensure proper alignment at the end
        original_index = df.index
        # Extract base column mappings with default empty dict
        base_columns = self.config.get("base_columns", {})
        feature_frames = []

        # Include OHLC features if needed - these are the raw price values
        if ohlc := self.config.get("ohlc"):
            for price, include in ohlc.items():
                if include:
                    # Get actual column name from base_columns mapping
                    col_name = base_columns[price]
                    # Add the price column as a feature
                    feature_frames.append(df[col_name].copy())

        # Time-based feature engineering - creates cyclic time features
        if tf_cfg := self.config.get("time_features"):
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
        if ret_cfg := self.config.get("returns"):
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
        if ind_cfg := self.config.get("indicators"):
            if isinstance(ind_cfg, dict):
                # Calculate all configured indicators
                ind_frames = [
                    calculate_indicator(df, name, cfg, base_columns)
                    for name, cfg in ind_cfg.items()
                ]
                # Concatenate all indicator results
                feature_frames.append(concat(ind_frames, axis=1))
            else:
                raise TypeError(f"Unsupported indicator configuration type: {type(ind_cfg)}")

        # Filter out any empty DataFrames before concatenation
        feature_frames = [frame for frame in feature_frames if not frame.empty]
        # Concatenate all features, ensuring alignment with original index
        if feature_frames:
            features = concat(feature_frames, axis=1).reindex(df.index)
        else:
            features = DataFrame(index=df.index)

        # Lagging feature engineering - creates lagged versions of features
        if lags := self.config.get("lags"):
            for col, lag in lags.items():
                lag_feats = get_lagging_features(features[col], max_lag=lag)
                features = concat([features, lag_feats], axis=1)

        return features.reindex(original_index)
