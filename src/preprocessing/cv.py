from dataclasses import dataclass
from collections.abc import Generator
from typing import Literal, Optional, Tuple, List, Iterator

import pandas as pd
import plotly.graph_objects as go


@dataclass
class SplitIndices:
    """
    Container for train/validation/test split indices.

    Attributes
    ----------
    train_idx : List[int]
        List of indices for training samples
    val_idx : Optional[List[int]]
        List of indices for validation samples (None if not applicable)
    test_idx : Optional[List[int]]
        List of indices for test samples (None if not applicable)
    """
    train_idx: List[int]
    val_idx: Optional[List[int]] = None
    test_idx: Optional[List[int]] = None

    def __iter__(self) -> Iterator[List[int]]:
        """Iterator implementation for unpacking split indices."""
        yield self.train_idx
        if self.val_idx is not None:
            yield self.val_idx
        if self.test_idx is not None:
            yield self.test_idx


class GroupTimeSeriesSplit:
    """
    Time-series cross-validation splitter with group support.

    This class implements time-series cross-validation that preserves the temporal
    order of data within each group. It supports both expanding and rolling window
    approaches for training data, with configurable validation and test periods.

    Key Features:
    - Group-aware splitting: Each group is processed independently
    - Multiple validation folds: Generate multiple consecutive validation periods
    - Test set support: Optionally reserve a fixed test period
    - Two window strategies: Expanding (all past data) or rolling (fixed window)
    - Visualizations: Built-in plotting capabilities for split analysis

    The splitter ensures that:
    1. Validation and test sets never overlap
    2. Temporal order is preserved (no future data leaks into past)
    3. All samples from a group appear in the same split

    Examples
    --------
    Basic usage with single validation fold:
    >>> cv = GroupTimeSeriesSplit(val_folds=1, interval='7d')
    >>> for train_idx, val_idx in cv.split(X, y, groups, timestamps):
    >>>     model.fit(X.iloc[train_idx], y.iloc[train_idx])
    >>>     score = model.score(X.iloc[val_idx], y.iloc[val_idx])

    Multiple validation folds with rolling window:
    >>> cv = GroupTimeSeriesSplit(val_folds=3, interval='2d',
    >>>                         train_interval='14d', window='rolling')
    >>> for train_idx, val_idx in cv.split(X, y, groups, timestamps):
    >>>     # Each fold uses 14 days for training and 2 days for validation
    >>>     model.fit(X.iloc[train_idx], y.iloc[train_idx])

    With test set reservation:
    >>> cv = GroupTimeSeriesSplit(val_folds=2, test_folds=1, interval='5d')
    >>> splits = list(cv.split(X, y, groups, timestamps))
    >>> # First two splits are validation, last is test
    """

    def __init__(
        self,
        val_folds: int = 1,
        test_folds: int = 0,
        interval: str = "7d",
        train_interval: Optional[str] = None,
        window: Literal["expanding", "rolling"] = "expanding"
    ) -> None:
        """
        Initialize the time-series cross-validation splitter.

        Parameters
        ----------
        val_folds : int, default=1
            Number of consecutive validation folds to generate per group.
            Each validation fold covers one interval period.
            Set to 0 to skip validation (only test folds will be generated).

        test_folds : int, default=0
            Number of consecutive test folds to generate per group.
            Each test fold covers one interval period.
            Set to 0 to skip test set creation (only validation folds will be generated).

        interval : str, default="7d"
            Time interval for each validation/test fold. Supported units:
            - 'm' - minutes (e.g., '15m' for 15 minutes)
            - 'h' - hours (e.g., '12h' for 12 hours)
            - 'd' - days (e.g., '7d' for 7 days)
            - 'M' - months (e.g., '1M' for 1 month)

        train_interval : str, optional
            Time interval for training data in rolling window mode.
            If None, uses the same interval as validation/test.
            Required when window='rolling'.
            Ignored when window='expanding'.

        window : {'expanding', 'rolling'}, default="expanding"
            Window type for training data:
            - 'expanding' - use all available past data for training
            - 'rolling' - use a fixed time window for training

        Raises
        ------
        ValueError
            If val_folds or test_folds are negative, or if rolling window is used
            without specifying train_interval.

        Notes
        -----
        - The total time range needed is (val_folds + test_folds) * interval
        - For rolling windows, an additional train_interval is needed
        - If the available data is insufficient, a ValueError will be raised during splitting
        """
        self._val_folds  = val_folds
        self._test_folds = test_folds
        self._offset     = self._parse_interval(interval)
        self._window     = window
        if train_interval:
            self._train_offset = self._parse_interval(train_interval)
        else:
            self._train_offset = self._offset

        # Validate parameters
        if val_folds < 0 or test_folds < 0:
            raise ValueError("val_folds and test_folds must be non-negative integers")

        if window == "rolling" and train_interval is None:
            raise ValueError("train_interval must be specified for rolling window")

    def _parse_interval(self, s: str):
        """Parse time interval string into pandas offset object."""
        n, unit = int(s[:-1]), s[-1]
        if unit == 'm': return pd.Timedelta(minutes=n)
        if unit == 'h': return pd.Timedelta(hours=n)
        if unit == 'd': return pd.Timedelta(days=n)
        if unit == 'M': return pd.DateOffset(months=n)
        raise ValueError("Unsupported unit. Use 'm', 'h', 'd', or 'M'")

    def _validate_time_range(self, timestamps: pd.Series) -> None:
        """
        Validate if the time range is sufficient for the requested splits.

        Parameters
        ----------
        timestamps : pd.Series
            Series of timestamps to validate

        Raises
        ------
        ValueError
            If the time range is insufficient for the requested splits
        """
        total_needed = self._offset * (self._val_folds + self._test_folds)
        if self._window == "rolling":
            total_needed += self._train_offset

        time_range = timestamps.max() - timestamps.min()

        if time_range < total_needed:
            raise ValueError(
                f"Time range {time_range} is insufficient for requested splits. "
                f"Needed at least {total_needed}."
            )

    def get_timestamp_split(self, timestamps: pd.Series, steps: int) -> pd.Timestamp:
        """
        Calculate split boundary timestamp for given number of steps.

        Parameters
        ----------
        timestamps : pd.Series
            Series of timestamps to split
        steps : int
            Number of intervals to offset from the end

        Returns
        -------
        pd.Timestamp
            Boundary timestamp for the split

        Examples
        --------
        >>> split_point = cv.get_timestamp_split(timestamps, steps=3)
        >>> print(f"Split at: {split_point}")
        """
        # Sort and get last timestamp
        timestamps = timestamps.sort_values().reset_index(drop=True)
        t_end = timestamps.iloc[-1]

        return t_end - self._offset * steps

    def _get_fold_indices(
    self,
        df: pd.DataFrame,
        start: pd.Timestamp,
        steps: int
    ) -> Generator[Tuple[List[int], List[int]], None, None]:
        """
        Generate indices for a single group's folds.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the group's data with '_idx' and 'ts' columns
        start : pd.Timestamp
            Starting timestamp for the first fold
        steps : int
            Number of folds to generate

        Yields
        ------
        Tuple[List[int], List[int]]
            Tuple of (train_indices, test_indices) for each fold
        """
        for k in range(steps):
            sv = start + k * self._offset
            ev = sv + self._offset

            if self._window == "expanding":
                train_mask = df['ts'] <= sv
            else:  # rolling
                train_start = sv - self._train_offset
                train_mask = (df['ts'] > train_start) & (df['ts'] <= sv)

            test_mask = (df['ts'] > sv) & (df['ts'] <= ev)

            train_idx = df.loc[train_mask, '_idx'].tolist()
            test_idx = df.loc[test_mask, '_idx'].tolist()

            yield train_idx, test_idx

    def split(
        self,
        X: Optional[pd.DataFrame],
        y: Optional[pd.Series],
        groups: pd.Series,
        timestamps: pd.Series
    ) -> Generator[SplitIndices, None, None]:
        """
        Generate time-series splits preserving group structure.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Feature matrix (not used directly, for scikit-learn compatibility)
        y : pd.Series, optional
            Target variable (not used directly, for scikit-learn compatibility)
        groups : pd.Series
            Group labels for each sample. Splits are generated independently
            for each unique group value.
        timestamps : pd.Series
            Timestamps for each sample. Must be the same length as X/y/groups.

        Yields
        ------
        SplitIndices
            Dataclass containing train/validation/test indices.
            The splits are yielded in the following order:
            1. All validation folds (if val_folds > 0)
            2. All test folds (if test_folds > 0)

            For each group, validation folds are ordered from oldest to most recent,
            followed by test folds (if any).

        Raises
        ------
        ValueError
            If groups or timestamps are not provided, or if the time range
            is insufficient for the requested splits.

        Examples
        --------
        Basic usage:
        >>> cv = GroupTimeSeriesSplit(val_folds=2, test_folds=1, interval='7d')
        >>> for split in cv.split(X, y, groups, timestamps):
        >>>     if split.val_idx is not None:
        >>>         print(f"Validation fold: {len(split.train_idx)} train, {len(split.val_idx)} validation")
        >>>     else:
        >>>         print(f"Test fold: {len(split.train_idx)} train, {len(split.test_idx)} test")

        Working with multiple groups:
        >>> cv = GroupTimeSeriesSplit(val_folds=1, interval='5d')
        >>> for split in cv.split(X, y, groups, timestamps):
        >>>     # Each split corresponds to one group
        >>>     group_id = groups.iloc[split.train_idx[0]] if split.train_idx else groups.iloc[split.val_idx[0]]
        >>>     print(f"Group {group_id}: {len(split.train_idx)} train, {len(split.val_idx)} validation")
        """
        if groups is None or timestamps is None:
            raise ValueError("groups and timestamps must be provided")

        idx = pd.RangeIndex(len(timestamps))
        df = pd.DataFrame({
            '_idx': idx,
            'group': groups.values,
            'ts': timestamps.values
        })

        for _, gdf in df.groupby('group'):
            gdf = gdf.sort_values('ts').reset_index(drop=True)

            # Generate validation folds
            if self._val_folds > 0:
                start_val = self.get_timestamp_split(gdf['ts'], self._val_folds + self._test_folds)
                for train_idx, val_idx in self._get_fold_indices(gdf, start_val, self._val_folds):
                    yield SplitIndices(
                        train_idx=train_idx,
                        val_idx=val_idx,
                        test_idx=None
                    )

            # Generate test folds
            if self._test_folds > 0:
                start_test = self.get_timestamp_split(gdf['ts'], self._test_folds)
                for train_idx, test_idx in self._get_fold_indices(gdf, start_test, self._test_folds):
                    yield SplitIndices(
                        train_idx=train_idx,
                        val_idx=None,
                        test_idx=test_idx
                    )

    def plot_split(
        self, y: pd.Series,
        groups: pd.Series,
        group_name: str,
        timestamps: pd.Series,
        y_title: str
    ):
        """
        Visualize splits for a specific group.

        Parameters
        ----------
        y : pd.Series
            Target variable for visualization. Used for plotting values.
        groups : pd.Series
            Group labels for each sample. Used to filter data for the specific group.
        group_name : str
            Specific group to visualize. Must be a value present in groups.
        timestamps : pd.Series
            Timestamps for each sample. Used for the x-axis of the plot.
        y_title : str
            Y-axis title for the plot.

        Returns
        -------
        go.Figure
            Plotly figure object with visualization of the splits.
            The plot includes:
            - Time series data for the specified group
            - Vertical lines marking validation and test period starts
            - Annotations for each fold's train and validation/test periods
            - Different colors for validation and test folds

        Examples
        --------
        >>> fig = cv.plot_split(y, groups, 'group_A', timestamps, 'Sales')
        >>> fig.show()

        Saving to file:
        >>> fig = cv.plot_split(y, groups, 'group_B', timestamps, 'Revenue')
        >>> fig.write_image('split_visualization.png')
        """
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
