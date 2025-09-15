from typing import List

from pandas import DataFrame, Index, concat


def align_by_timestamps(
    dfs: List[DataFrame],
    ts_col: str
) -> List[DataFrame]:
    """Filters an arbitrary DataFrame, leaving only timestamped rows from good_ts."""
    merged = concat(dfs, axis=0)
    # Count how many unique tickers there are on each timestamp
    counts = merged.groupby(ts_col)["ticker"].nunique()
    total = merged["ticker"].nunique()
    # Keep only those timestamps with full coverage
    good_ts = counts[counts == total].index
    return [df[df[ts_col].isin(good_ts)].reset_index(drop=True) for df in dfs]


# TODO: guess we can delete it
def get_good_timestamps(
    merged_df: DataFrame,
    ts_col: str,
    ticker_col: str
) -> Index:
    """Returns the Index of timestamps in which all tickers are present."""
    # Count how many unique tickers there are on each timestamp
    counts = (
        merged_df
        .groupby(ts_col)[ticker_col]
        .nunique()
    )
    total_tickers = merged_df[ticker_col].nunique()
    # Keep only those timestamps with full coverage
    good_ts = counts[counts == total_tickers].index
    return good_ts
