


import os
from typing import Any, Literal, Optional
import pandas as pd

class AssetKit:
    def __init__(self, name: str) -> None:
        self.name = name

    def set_model(self):
        ...

    def set_scaler(self):
        ...

    def set_data(self):
        ...


class Pipeline:
    ...
    def __init__(
            self,
            df: Optional[pd.DataFrame] = None,
            task: Literal["direction", "price"] = "price",
            distribution: Literal["id", "ood"] = "id",
            method: Literal["base", "gbm"] = "base",
            algo: Optional[list[Literal["xgboost", "lightgbm", "catboost", "prophet", "lstm"]]] = None,
            horizon: str = "15m",
            retrain_period: Optional[str] = "7d",
    ) -> None:
        """
        m - minute
        h - hour
        d - day
        M - month

        retrain_period - if None, retraining by model degrodation trigger
        """
        if task == "direction" and method == "gbm":
            raise ValueError(f"GBM method available only for price prediction task, but `{task=}`")
        if algo:
            for a in algo:
                if a not in  ["xgboost", "lightgbm", "catboost", "prophet", "lstm"]:
                    raise ValueError(f"Unknown algorithm selected `{a}`")
        else:
            if distribution == "id":
                algo = ["xgboost"]
            elif distribution == "ood":
                algo = ["lstm"]
            else:
                raise ValueError(
                    f"Bad distribution method set: {distribution}."
                    "Choose `id` (in-distribution prediction) or `ood` (out-of-distribution prediction)."
                )
        self._distribution = distribution
        self._algo = algo
        self._task = task
        self._method = method
        self._raw_data: Optional[list[pd.DataFrame]] = [df] if df is not None else None

    def preprocess(self, ):
        ...

    def optimize(self, ):
        ...

    def train(self, ):
        ...

    def retrain(self, ):
        ...

    def set_trigger(self):
        """Sets trigger for model retraining: periodic | distributional | degrodational"""
        ...

    def predict(self, ):
        ...

    def save(self, ):
        ...

    def load(
            self,
            path: str,
            base_columns: Optional[dict[str, str]] = None,
            loader_config: Optional[dict[str, Any]] = None,
            return_data: bool = False
    ) -> Optional[list[pd.DataFrame]]:
        """
        Function for loading data from single file or all directory with assigning
        files in the new column named 'ticker' by symbols before the first '_' (underline).

        Parameters
        ---
        `path` str:
            can be directory containing multiple files or single file
        `base_columns` dictionary of strings:
            must contain base columns names, including OHLCV (separately) and timestamp
        `loader_config` dictionary of strings:
            any configs for `pd.read_csv()`
        `return_data` bool, default=False:
            if needed to return raw loaded data
        """
        if base_columns is None:
            base_columns = dict(
                timestamp="timestamp",
                open="open", high="open",
                low="open", close="open"
            )
        if loader_config is None:
            loader_config = {}

        if os.path.isdir(path):
            file_names = os.listdir(path)

            dfs = []
            for file_name in file_names:
                full_path = os.path.join(path, file_name)

                df = pd.read_csv(full_path, **loader_config)

                # Assign ticker to the new column
                ticker_name = file_name.split("_")[0]
                df["ticker"] = ticker_name

                dfs.append(df)
            self._raw_data = dfs
        else:
            df = pd.read_csv(path, **loader_config)

            # Assign ticker to the new column
            ticker_name = path.split("_")[0]
            df.name = ticker_name

            self._raw_data = [df]

        if return_data:
            return pd.concat(self._raw_data, axis=0, ignore_index=True) # type: ignore
