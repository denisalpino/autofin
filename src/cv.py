import pandas as pd
import plotly.graph_objects as go

from typing import Literal, Optional, Tuple, List, Union


class GroupTimeSeriesSplit:
    def __init__(
            self,
            val_folds: int = 1, test_folds: int = 0,
            interval: str = "7d",
            window: Literal["expanding", "rolling"] = "expanding"
    ) -> None:
        """
        interval : str
            A string like '7d', '12h', '15m', '1M' giving the length T of each
            test window.
        val_folds : int
            Number of consecutive validation windows to generate per group. Value greater
            than 1 means using cross validation
        test_folds : int
            Number of consecutive test windows to generate per group. Value 0 means
            that we don't need to return indecies for testing samples.
        window : 'expanding' or 'rolling'
            Type of window.
        """
        self._val_folds   = val_folds
        self._test_folds  = test_folds
        self._offset      = self._parse_interval(interval)
        self._window      = window

    def _parse_interval(self, s: str):
        n, unit = int(s[:-1]), s[-1]
        if unit == 'm': return pd.Timedelta(minutes=n)
        if unit == 'h': return pd.Timedelta(hours=n)
        if unit == 'd': return pd.Timedelta(days=n)
        if unit == 'M': return pd.DateOffset(months=n)
        raise ValueError("Unsupported unit, use [m,h,d,M].")


    def get_timestamp_split(self, timestamps: pd.Series, steps: int) -> pd.Timestamp:
        """
        Separates the `interval` * `steps` of the most recent observations from the pandas.DataFrame.

        Parameters
        ---
        timestamps : pandas.Series[pandas.Timestamp]
            Series of timestamps from main pandas.DataFrame.
        interval : str
            A string of the form `7d`, `12h`, `15m`, `1M` (N + unit: m, h, d, M).
        steps : int
            Number of repetitions of the interval.

        Returns
        ---
        pandas.Timestamp

        Example:
        ---
        ```
        start = get_timestamp_split(df["timestamps"], '7d', steps=4)
        ```
        """
        # Sort and get last timestamp
        timestamps = timestamps.sort_values().reset_index(drop=True)
        t_end = timestamps.iloc[-1]

        # Calculate borders
        start_third = t_end - self._offset * steps

        return start_third


    def _get_train_test_idx(
            self,
            df: pd.DataFrame,
            start: pd.Timestamp,
            steps: int
    ):
        train_test = []

        for k in range(steps):
            sv = start + k * self._offset
            ev = sv + self._offset

            if self._window == "expanding":
                train_mask = df['ts'] < sv
            else:  # rolling
                train_mask = (df['ts'] < sv) & (df['ts'] >= sv - self._offset)

            test_mask = (df['ts'] >= sv) & (df['ts'] < ev)
            train_idx = df.loc[train_mask, '_idx'].tolist()
            test_idx  = df.loc[test_mask,   '_idx'].tolist()
            train_test.append([train_idx, test_idx])

        return train_test


    def split(
            self,
            X: Optional[pd.DataFrame], y: Optional[pd.Series],
            groups: pd.Series,
            timestamps: pd.Series
    ) -> Tuple[Union[Tuple[List[int],List[int]],
                    Tuple[List[int],List[int],List[int]]], ...]:
        """
        Time-series cross-validation splitter, similar to sklearn.TimeSeriesSplit,
        but applied separately within each group.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (not used in split logic, but included to mirror sklearn API).
        y : pd.Series
            Target vector (not used in split logic, but included to mirror sklearn API).
        groups : pd.Series
            A group label for each row of X/y/timestamps. Splits are generated
            independently for each unique group value, in order.
        timestamps : pd.Series
            A pandas Series of Timestamps, same length as X/y/groups.

        Returns (TODO)
        -------
        splits : tuple of (train_idx, val_idx)
            A tuple of length (n_groups * steps), where each element is a pair
            of lists of integer indices into X/y:

            - train_idx : all rows of the *same* group whose timestamp is
                strictly *before* the start of the k-th val window,
            - val_idx  : all rows of the same group whose timestamp falls
                into the k-th val window itself.

            The val windows for each group are the last `steps` windows of
            length `interval`, ordered from oldest to most recent, and for
            each successive fold, the training set automatically grows
            (includes all data older than the val window).
        """
        # we'll need the original positions
        idx = pd.RangeIndex(len(timestamps))
        df = pd.DataFrame({
            '_idx':     idx,
            'group':    groups.values,
            'ts':       timestamps.values
        })

        train_val = []
        train_test = []

        for _, gdf in df.groupby('group'):
            # sort this group's data by timestamp
            gdf        = gdf.sort_values('ts').reset_index(drop=True)
            start_val  = self.get_timestamp_split(gdf['ts'], self._val_folds + self._test_folds)
            start_test = self.get_timestamp_split(gdf['ts'], self._test_folds)

            group_train_val = self._get_train_test_idx(gdf, start_val, self._val_folds)
            train_val.extend(group_train_val)

            if self._test_folds > 0:
                group_train_val = self._get_train_test_idx(gdf, start_test, self._test_folds)
                train_test.extend(group_train_val)

        # return as a tuple of tuples
        return tuple(train_val), tuple(train_test)


    def plot_split(self, y: pd.Series, groups: pd.Series, group_name: str, timestamps: pd.Series, y_title):
        y_group          = y[groups == group_name]
        timestamps_group = timestamps[groups == group_name]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps_group,
            y=y_group,
            name=group_name,
            yaxis="y1",
        ))
        fig.update_layout(
            title=f"<b>{group_name}<b>",
            title_x=0.5,
            xaxis=dict(title="<b>Date & Time<b>"),
            yaxis=dict(
                title=f"<b>{y_title}<b>",
                side="left",
                showgrid=False
            ),
            height=600,
            template="plotly_dark"
        )

        # Show validation sample start
        q1 = self.get_timestamp_split(timestamps_group, steps=self._val_folds + self._test_folds)
        fig.add_vline(x=q1, line=dict(color="#2ca02c"))
        fig.add_annotation(
            x=q1, y=0.5, showarrow=False,
            yref="paper", yanchor="middle", xshift=10,
            text="validation", textangle=-90, font=dict(size=16)
        )

        # Show test sample start
        if self._test_folds:
            q2 = self.get_timestamp_split(timestamps_group, steps=self._test_folds)
            fig.add_vline(x=q2, line=dict(color="#ff7f0e"))
            fig.add_annotation(
                x=q2, y=0.5, showarrow=False,
                yref="paper", yanchor="middle", xshift=10,
                text="test", textangle=-90, font=dict(size=16)
            )

        groups = pd.Series([group_name] * len(timestamps_group))
        train_val, train_test = self.split(X=None, y=None, groups=groups, timestamps=timestamps_group)

        for ind, (train_idx, val_idx) in enumerate(train_val, start=0):
            train_last = timestamps_group.iloc[train_idx].max()
            val_last   = timestamps_group.iloc[val_idx].max()

            fig.add_vline(x=train_last, line=dict(color="#2ca02c", dash="dash"), opacity=0.2, name=group_name)
            fig.add_vline(x=val_last, line=dict(color="#2ca02c", dash="dash"), opacity=0.2, name=group_name)
            fig.add_annotation(
                x=train_last, y=1, showarrow=False,
                yref="paper", yanchor="bottom", xshift=-10,
                text=f"Train {ind % self._val_folds + 1}", textangle=-90, font=dict(size=12, color="#2ca02c"),
                name=group_name
            )
            fig.add_annotation(
                x=val_last, y=1, showarrow=False,
                yref="paper", yanchor="top", xshift=-10,
                text=f"Val {ind % self._val_folds + 1}", textangle=-90, font=dict(size=12, color="#2ca02c"),
                name=group_name
            )

        for ind, (train_idx, val_idx) in enumerate(train_test, start=0):
            train_last = timestamps_group.iloc[train_idx].max()
            val_last   = timestamps_group.iloc[val_idx].max()

            fig.add_vline(x=train_last, line=dict(color="#ff7f0e", dash="dash"), opacity=0.2, name=group_name)
            fig.add_vline(x=val_last, line=dict(color="#ff7f0e", dash="dash"), opacity=0.2, name=group_name)
            # Аннотация «прилипает» к низу графика
            fig.add_annotation(
                x=train_last, y=1, showarrow=False,
                yref="paper", yanchor="bottom", xshift=-10,
                text=f"Train {ind % self._test_folds + 1}", textangle=-90, font=dict(size=12, color="#ff7f0e"),
                name=group_name
            )
            # Аннотация «прилипает» к низу графика
            fig.add_annotation(
                x=val_last, y=1, showarrow=False,
                yref="paper", yanchor="top", xshift=-10,
                text=f"Test {ind % self._test_folds + 1}", textangle=-90, font=dict(size=12, color="#ff7f0e"),
                name=group_name
            )

        return fig