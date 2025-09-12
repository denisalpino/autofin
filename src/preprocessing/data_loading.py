import os
import pandas as pd
from typing import Optional, List


def load_data(
        file: Optional[str] = None,
        dir: Optional[str] = None,
        loader_config: dict = {}
) -> List[pd.DataFrame]:
    """
    Function for loading data from single file or all directory with assigning
    files in the new column named 'ticker' by symbols before the first '_' (underline).
    """
    dfs = []
    if file:
        df = pd.read_csv(file, **loader_config)
        df["ticker"] = file.split("_")[0]
        dfs.append(df)
    elif dir:
        for fname in os.listdir(dir):
            df = pd.read_csv(os.path.join(dir, fname), **loader_config)
            df["ticker"] = fname.split("_")[0]
            dfs.append(df)
    else:
        raise ValueError("Specify either file or dir.")
    for df in dfs:
        df["ticker"] = pd.Categorical(df["ticker"])
    return dfs


def align_by_timestamps(
        dfs: list[pd.DataFrame],
        ts_col: str
) -> List[pd.DataFrame]:
    """Filters an arbitrary DataFrame, leaving only timestamped rows from good_ts."""
    merged = pd.concat(dfs, axis=0)
    counts = merged.groupby(ts_col)["ticker"].nunique()
    total = merged["ticker"].nunique()
    good_ts = counts[counts == total].index
    return [df[df[ts_col].isin(good_ts)].reset_index(drop=True) for df in dfs]
