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
