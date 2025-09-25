import os
import sys

from pandas import DataFrame, concat

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
sys.path.insert(0, project_root)

from src.features.calculators.indicators import calculate_indicator
from src.features.calculators.returns import calculate_returns
from src.features.calculators.time_features import create_time_features
from src.features.calculators.lagging import get_lagging_features
from src.config.schemas.features import FeatureConfig, TimeFeature, ColumnSource


class FeatureBuilder:
    def __init__(self, config: FeatureConfig):
        """
        config:
            Configuration model specifying which features to create
            - time_features: Field specifying which time-based features to create
            - returns: Field configuring returns calculation
            - indicators: Field configuring technical indicator calculations
            - lags: Field specifying lag features to create
            - dimred: Field specifying lag features to create
        """
        self.config = config

    def build_features(self, df: DataFrame) -> DataFrame:
        """
        Creates feature matrix based on configuration model.

        Parameters
        ---
        df : DataFrame
            Input DataFrame with raw financial data

        Returns
        ---
        DataFrame:
            DataFrame with engineered features

        Raises
        ---
        KeyError:
            If required base_columns are missing from self.config
        TypeError:
            If indicators configuration is not a dictionary
        ValueError:
            If time_features are requested but no parameters are specified
        """
        df = df.copy()
        # Store original index to ensure proper alignment at the end
        original_index = df.index
        # Extract base column mappings with default empty dict
        feature_frames = []

        # Time-based feature engineering - creates cyclic time features
        if tf_cfg := self.config.time_features:
            # Create boolean list of which time features to generate
            bool_params = [param in tf_cfg for param in TimeFeature]

            # Generate time features using the timestamp column
            time_features = create_time_features(df.timestamps, *bool_params)
            feature_frames.append(time_features)

        # Returns feature engineering - calculates price returns
        if ret_cfg := self.config.returns:
            ret = calculate_returns(
                df[ret_cfg.column],
                period=ret_cfg.period,
                method=ret_cfg.method,
            )
            # Name the returns series for easier identification
            ret.name = "returns"
            feature_frames.append(ret)

        # Indicator-based feature engineering - calculates technical indicators
        if ind_cfg := self.config.indicators:
            # Calculate all configured indicators
            indicator = [calculate_indicator(df, cfg) for cfg in ind_cfg]
            # Concatenate all indicator results
            feature_frames.append(concat(indicator, axis=1))

        # Filter out any empty DataFrames before concatenation
        feature_frames = [frame for frame in feature_frames if not frame.empty]
        # Concatenate all features, ensuring alignment with original index
        if feature_frames:
            features = concat(feature_frames, axis=1)
        else:
            features = DataFrame(index=df.index)

        # Lagging feature engineering - creates lagged versions of features
        if lags_cfgs := self.config.lags:
            for lag_cfg in lags_cfgs:
                lag_features = get_lagging_features(features, lag_cfg)
                features = concat([features, lag_features], axis=1)

        # TODO: We're working with features that doesn't include base_columns,
        # so we can't exclude some cols, only before merging main df with features.
        # It means that we need to transfer this part to DataPipeline!!!
        # Drop columns if needed
        #if exclude := self.config.exclude:
        #    print(self.config.exclude)
        #    print(features.columns)
        #    features = features.drop(columns=[col.value for col in exclude])

        return features.reindex(original_index)
