import warnings
from dataclasses import dataclass
from collections.abc import Generator
from typing import Literal, Optional, Dict, List, Iterator, Union

import pandas as pd
import plotly.express as px
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
    group : str
        Group identifier for this split
    """
    train_idx: List[int]
    val_idx: Optional[List[int]] = None
    test_idx: Optional[List[int]] = None
    group: Optional[str] = None

    def __iter__(self) -> Iterator[List[int]]:
        """Iterator implementation for unpacking split indices."""
        yield self.train_idx
        if self.val_idx is not None:
            yield self.val_idx
        if self.test_idx is not None:
            yield self.test_idx


@dataclass
class SplitResult:
    """
    Container for all splits for a group.

    Attributes
    ----------
    group : str
        Group identifier
    train_test_split : Optional[SplitIndices]
        Full train-test split (if test_interval is specified)
    validation_splits : List[SplitIndices]
        List of validation splits
    """
    group: str
    train_test_split: Optional[SplitIndices] = None
    validation_splits: List[SplitIndices] = None

    def __post_init__(self):
        if self.validation_splits is None:
            self.validation_splits = []


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
    >>> results = cv.split(X, y, groups, timestamps)
    >>> for group_result in results.values():
    >>>     for split in group_result.validation_splits:
    >>>         model.fit(X.iloc[split.train_idx], y.iloc[split.train_idx])
    >>>         score = model.score(X.iloc[split.val_idx], y.iloc[split.val_idx])

    With test set reservation:
    >>> cv = GroupTimeSeriesSplit(val_folds=2, test_interval='5d', interval='5d')
    >>> results = cv.split(X, y, groups, timestamps)
    >>> for group_name, group_result in results.items():
    >>>     if group_result.train_test_split:
    >>>         # Use full train-test split
    >>>         train_idx = group_result.train_test_split.train_idx
    >>>         test_idx = group_result.train_test_split.test_idx
    >>>         model.fit(X.iloc[train_idx], y.iloc[train_idx])
    >>>         score = model.score(X.iloc[test_idx], y.iloc[test_idx])
    """

    def __init__(
        self,
        val_folds: int = 1,
        test_interval: Optional[str] = None,
        interval: str = "7d",
        train_interval: Optional[str] = None,
        window: Literal["expanding", "rolling"] = "expanding",
        min_train_samples: int = 1
    ) -> None:
        """
        Initialize the time-series cross-validation splitter.

        Parameters
        ----------
        val_folds : int, default=1
            Number of consecutive validation folds to generate per group.
            Each validation fold covers one interval period.
            Set to 0 to skip validation (only test splits will be generated).

        test_interval : str, optional
            Time interval for test data. If provided, reserves this period
            at the end of each group's timeline for testing.
            Supported units: 'm' (minutes), 'h' (hours), 'd' (days), 'M' (months)

        interval : str, default="7d"
            Time interval for each validation fold. Supported units:
            - 'm' - minutes (e.g., '15m' for 15 minutes)
            - 'h' - hours (e.g., '12h' for 12 hours)
            - 'd' - days (e.g., '7d' for 7 days)
            - 'M' - months (e.g., '1M' for 1 month)

        train_interval : str, optional
            Time interval for training data in rolling window mode.
            If None, uses the same interval as validation.
            Required when window='rolling'.
            Ignored when window='expanding'.

        window : {'expanding', 'rolling'}, default="expanding"
            Window type for training data:
            - 'expanding' - use all available past data for training
            - 'rolling' - use a fixed time window for training

        min_train_samples : int, default=1
            Minimum number of training samples required for a valid split.
            Splits with fewer training samples will be skipped.

        Raises
        ------
        ValueError
            If val_folds is negative, or if rolling window is used
            without specifying train_interval.
        """
        self._val_folds = val_folds
        self._test_interval = test_interval
        self._offset = self._parse_interval(interval)
        self._window = window
        self._min_train_samples = min_train_samples

        if train_interval:
            self._train_offset = self._parse_interval(train_interval)
        else:
            self._train_offset = self._offset

        # Parse test interval if provided
        if test_interval:
            self._test_offset = self._parse_interval(test_interval)
        else:
            self._test_offset = None

        # Validate parameters
        if val_folds < 0:
            raise ValueError("val_folds must be a non-negative integer")

        if window == "rolling" and train_interval is None:
            raise ValueError("train_interval must be specified for rolling window")

    def _parse_interval(self, s: str) -> Union[pd.Timedelta, pd.DateOffset]:
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
        total_needed = self._offset * self._val_folds

        if self._test_offset:
            total_needed += self._test_offset

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
        """
        timestamps = timestamps.sort_values().reset_index(drop=True)
        t_end = timestamps.iloc[-1]
        return t_end - self._offset * steps

    def _get_fold_indices(
        self,
        df: pd.DataFrame,
        start: pd.Timestamp,
        steps: int,
        group_name: str
    ) -> Generator[SplitIndices, None, None]:
        """Generate indices for a single group's folds."""
        for k in range(steps):
            sv = start + k * self._offset
            ev = sv + self._offset

            if self._window == "expanding":
                train_mask = df['ts'] <= sv
            else:  # rolling
                train_start = sv - self._train_offset
                train_mask = (df['ts'] >= train_start) & (df['ts'] <= sv)

            val_mask = (df['ts'] > sv) & (df['ts'] <= ev)

            train_idx = df.loc[train_mask, '_idx'].tolist()
            val_idx = df.loc[val_mask, '_idx'].tolist()

            # Skip folds with insufficient training data
            if len(train_idx) < self._min_train_samples:
                warnings.warn(
                    f"Skipping fold {k + 1} for group {group_name}: "
                    f"Only {len(train_idx)} training samples (minimum {self._min_train_samples} required)"
                )
                continue

            # Skip folds where training set is smaller than validation set
            if len(train_idx) < len(val_idx):
                warnings.warn(
                    f"Skipping fold {k + 1} for group {group_name}: "
                    f"Training samples ({len(train_idx)}) < validation samples ({len(val_idx)})."
                )
                continue

            yield SplitIndices(
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=None,
                group=group_name
            )


    def split(
        self,
        X: Optional[pd.DataFrame],
        y: Optional[pd.Series],
        groups: pd.Series,
        timestamps: pd.Series
    ) -> Dict[str, SplitResult]:
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

        Returns
        -------
        Dict[str, SplitResult]
            Dictionary where keys are group names and values are SplitResult objects
            containing train-test split and validation splits for each group.

        Raises
        ------
        ValueError
            If groups or timestamps are not provided, or if the time range
            is insufficient for the requested splits.
        """
        if groups is None or timestamps is None:
            raise ValueError("groups and timestamps must be provided")

        idx = pd.RangeIndex(len(timestamps))
        df = pd.DataFrame({
            '_idx': idx,
            'group': groups.values,
            'ts': timestamps.values
        })

        result = {}

        for group_name, gdf in df.groupby('group'):
            gdf = gdf.sort_values('ts').reset_index(drop=True)
            group_result = SplitResult(group=group_name)

            # Handle test data if test_interval is specified
            train_val_df = gdf
            if self._test_offset:
                test_start = gdf['ts'].iloc[-1] - self._test_offset
                test_mask = gdf['ts'] > test_start
                test_df = gdf[test_mask]
                train_val_df = gdf[~test_mask]

                # Add test split
                if len(test_df) > 0:
                    group_result.train_test_split = SplitIndices(
                        train_idx=train_val_df['_idx'].tolist(),
                        val_idx=None,
                        test_idx=test_df['_idx'].tolist(),
                        group=group_name
                    )

            # Validate time range for this group
            self._validate_time_range(gdf['ts'])

            # Generate validation folds
            if self._val_folds > 0 and len(train_val_df) > 0:
                # Calculate start point for validation folds
                # We need to ensure we have enough data for all folds
                total_val_range = self._offset * self._val_folds
                start_val = train_val_df['ts'].iloc[-1] - total_val_range

                # Generate folds
                validation_splits = list(self._get_fold_indices(
                    train_val_df, start_val, self._val_folds, group_name
                ))
                group_result.validation_splits = validation_splits

            result[group_name] = group_result

        return result

    def plot_split(
        self,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
        group_name: Optional[str] = None,
        timestamps: Optional[pd.Series] = None,
        y_title: str = "Value",
        title: Optional[str] = None,
        theme: Literal["dark", "light"] = "dark",
        height: int = 700,
        width: int = 1200
    ) -> go.Figure:
        """
        Visualize time series cross-validation splits for different groups.

        This method generates an interactive plot showing train, validation, and test
        splits for time series cross-validation. It supports multiple groups and provides
        a dropdown menu to switch between them.

        Parameters
        ----------
        y : pd.Series, optional
            Target variable values. If not provided, sequential values will be used.
        groups : pd.Series
            Group labels for each sample. Must be the same length as timestamps.
        group_name : str, optional
            Specific group to display initially. If not provided, the first group is used.
        timestamps : pd.Series
            Timestamps for each sample. Must be the same length as groups.
        y_title : str, default="Value"
            Title for the Y-axis.
        title : str, optional
            Plot title. If not provided, a default title will be generated.
        theme : {"dark", "light"}, default="dark"
            Color theme for the plot.
        height : int, default=700
            Plot height in pixels.
        width : int, default=1200
            Plot width in pixels.

        Returns
        -------
        go.Figure
            Plotly Figure object with the visualization.

        Examples
        --------
        >>> cv = GroupTimeSeriesSplit(val_folds=3, test_interval='30d')
        >>> fig = cv.plot_split(
        ...     y=target_series,
        ...     groups=group_series,
        ...     timestamps=timestamp_series,
        ...     group_name='AAPL',
        ...     title='Stock Price CV Splits',
        ...     theme='dark'
        ... )
        >>> fig.show()
        """
        # Set theme colors (dark theme by default)
        if theme == "dark":
            bg_color = '#121212'
            text_color = 'white'
            grid_color = 'rgba(255, 255, 255, 0.1)'
            train_color = '#1f77b4'
            val_color = '#ff7f0e'
            test_color = '#d62728'
            line_color = '#ffffff'
            dropdown_bg = '#2c2c2c'
            dropdown_text = 'white'
            active_bg = '#404040'
            hover_bg = '#3a5bb8'
            divider_color = 'rgba(255, 255, 255, 0.3)'
            crosshair_color = 'rgba(255, 255, 255, 0.5)'
        else:
            bg_color = 'white'
            text_color = 'black'
            grid_color = 'rgba(0, 0, 0, 0.1)'
            train_color = '#1f77b4'
            val_color = '#ff7f0e'
            test_color = '#d62728'
            line_color = '#000000'
            dropdown_bg = '#f0f0f0'
            dropdown_text = 'black'
            active_bg = '#d0d0d0'
            hover_bg = '#3a5bb8'
            divider_color = 'rgba(0, 0, 0, 0.3)'
            crosshair_color = 'rgba(0, 0, 0, 0.5)'

        # Convert timestamps to Series if it's a DatetimeIndex
        if isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.Series(timestamps)

        # Validate inputs
        if groups is None or timestamps is None:
            raise ValueError("groups and timestamps must be provided")

        # Get splits for all groups
        splits_dict = self.split(X=None, y=None, groups=groups, timestamps=timestamps)

        # Calculate global y range if y is provided
        if y is not None:
            global_y_min = y.min()
            global_y_max = y.max()
            global_y_range = global_y_max - global_y_min
        else:
            # Use index-based values if y is not provided
            global_y_min = 0
            global_y_max = len(timestamps) - 1
            global_y_range = global_y_max - global_y_min

        # Determine which group to display initially
        if group_name is None:
            group_name = list(splits_dict.keys())[0]

        if group_name not in splits_dict:
            raise ValueError(f"Group '{group_name}' not found in the split results")

        # Create a mapping from global index to group-specific index
        global_to_local_idx = {}
        group_y_ranges = {}
        for group in splits_dict.keys():
            group_mask = groups == group
            group_indices = groups[group_mask].index
            global_to_local_idx[group] = {global_idx: local_idx for local_idx, global_idx in enumerate(group_indices)}

            # Calculate group-specific y range
            if y is not None:
                group_y = y[group_mask]
                group_y_min = group_y.min()
                group_y_max = group_y.max()
                group_y_range = group_y_max - group_y_min
            else:
                group_y_min = 0
                group_y_max = len(group_indices) - 1
                group_y_range = group_y_max - group_y_min

            group_y_ranges[group] = (group_y_min, group_y_max, group_y_range)

        # Create plot
        fig = go.Figure()

        # Add traces for all groups but make them invisible initially
        all_groups = list(splits_dict.keys())
        colors = px.colors.qualitative.Plotly

        for i, group in enumerate(all_groups):
            group_mask = groups == group
            group_timestamps = timestamps[group_mask]

            if y is not None:
                group_y = y[group_mask]
            else:
                group_y = pd.Series(range(len(group_timestamps)), index=group_timestamps.index)

            # Sort by timestamp
            sorted_idx = group_timestamps.argsort()
            group_timestamps = group_timestamps.iloc[sorted_idx]
            group_y = group_y.iloc[sorted_idx]

            # Get group-specific y range
            group_y_min, group_y_max, group_y_range = group_y_ranges[group]

            # Add main trace for this group
            fig.add_trace(go.Scatter(
                x=group_timestamps,
                y=group_y,
                mode='lines+markers',
                name=group,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4),
                visible=(group == group_name),
                hovertemplate=(
                    '<b>Group</b>: %{text}<br>'
                    '<b>Date</b>: %{x}<br>'
                    '<b>Value</b>: %{y:.2f}<br>'
                    '<extra></extra>'
                ),
                text=[group] * len(group_timestamps)
            ))

            # Get split results for this group
            group_result = splits_dict[group]

            # Calculate the minimum time interval between data points
            if len(group_timestamps) > 1:
                time_diffs = group_timestamps.sort_values().diff().dropna()
                min_interval = time_diffs.min()
            else:
                min_interval = pd.Timedelta(days=1)

            # Calculate total number of folds
            total_val_folds = len(group_result.validation_splits)
            has_test = group_result.train_test_split and group_result.train_test_split.test_idx

            # Add test split if exists
            if has_test:
                test_idx = group_result.train_test_split.test_idx
                # Convert global indices to group-specific indices
                test_idx_local = [global_to_local_idx[group][idx] for idx in test_idx if idx in global_to_local_idx[group]]
                test_timestamps = group_timestamps.iloc[test_idx_local]
                test_start = test_timestamps.min()
                test_end = test_timestamps.max() + min_interval

                # Add test rectangle
                fig.add_trace(go.Scatter(
                    x=[test_start, test_start, test_end, test_end, test_start],
                    y=[group_y_min, group_y_max, group_y_max, group_y_min, group_y_min],
                    fill="toself",
                    fillcolor=test_color,
                    opacity=0.2,
                    line=dict(color=test_color, width=2),
                    mode="lines",
                    showlegend=False,
                    hoverinfo="skip",
                    visible=(group == group_name)
                ))

                # Add rotated test annotation
                test_center_x = test_start + (test_end - test_start) / 2
                test_center_y = group_y_min + group_y_range / 2

                # Calculate font size based on test duration
                test_duration = (test_end - test_start).total_seconds()
                font_size = min(36, max(16, int(test_duration / 3600)))

                fig.add_trace(go.Scatter(
                    x=[test_center_x],
                    y=[test_center_y],
                    mode="text",
                    text=["TEST"],
                    textfont=dict(size=font_size, color=test_color, family="Arial Black"),
                    showlegend=False,
                    hoverinfo="skip",
                    visible=(group == group_name)
                ))

                # Add dividing lines around test rectangle
                fig.add_trace(go.Scatter(
                    x=[test_start, test_start],
                    y=[group_y_min, group_y_max],
                    mode="lines",
                    line=dict(color=divider_color, width=1, dash="solid"),
                    showlegend=False,
                    hoverinfo="skip",
                    visible=(group == group_name)
                ))
                fig.add_trace(go.Scatter(
                    x=[test_end, test_end],
                    y=[group_y_min, group_y_max],
                    mode="lines",
                    line=dict(color=divider_color, width=1, dash="solid"),
                    showlegend=False,
                    hoverinfo="skip",
                    visible=(group == group_name)
                ))

            # Add validation and training folds
            if total_val_folds > 0:
                fold_height = group_y_range / total_val_folds

                for i, split in enumerate(group_result.validation_splits):
                    if split.val_idx:
                        # Convert global indices to group-specific indices
                        val_idx_local = [global_to_local_idx[group][idx] for idx in split.val_idx if idx in global_to_local_idx[group]]
                        val_timestamps = group_timestamps.iloc[val_idx_local]
                        val_start = val_timestamps.min()
                        val_end = val_timestamps.max() + min_interval

                        # Calculate vertical position
                        fold_y_min = group_y_min + i * fold_height
                        fold_y_max = fold_y_min + fold_height
                        fold_center_y = fold_y_min + fold_height / 2

                        # Add training rectangle
                        if split.train_idx:
                            # Convert global indices to group-specific indices
                            train_idx_local = [global_to_local_idx[group][idx] for idx in split.train_idx if idx in global_to_local_idx[group]]
                            train_timestamps = group_timestamps.iloc[train_idx_local]
                            train_start = train_timestamps.min()
                            train_end = train_timestamps.max() + min_interval

                            fig.add_trace(go.Scatter(
                                x=[train_start, train_start, train_end, train_end, train_start],
                                y=[fold_y_min, fold_y_max, fold_y_max, fold_y_min, fold_y_min],
                                fill="toself",
                                fillcolor=train_color,
                                opacity=0.2,
                                line=dict(color=train_color, width=2),
                                mode="lines",
                                showlegend=False,
                                hoverinfo="skip",
                                visible=(group == group_name)
                            ))

                            # Add training annotation
                            train_center_x = train_start + (train_end - train_start) / 2
                            font_size = min(24, max(14, int(fold_height / 10)))

                            fig.add_trace(go.Scatter(
                                x=[train_center_x],
                                y=[fold_center_y],
                                mode="text",
                                text=[f"TRAIN {i+1}"],
                                textfont=dict(size=font_size, color=train_color, family="Arial Black"),
                                showlegend=False,
                                hoverinfo="skip",
                                visible=(group == group_name)
                            ))

                            # Add dividing lines around training rectangle
                            fig.add_trace(go.Scatter(
                                x=[train_start, train_start],
                                y=[fold_y_min, fold_y_max],
                                mode="lines",
                                line=dict(color=divider_color, width=1, dash="solid"),
                                showlegend=False,
                                hoverinfo="skip",
                                visible=(group == group_name)
                            ))
                            fig.add_trace(go.Scatter(
                                x=[train_end, train_end],
                                y=[fold_y_min, fold_y_max],
                                mode="lines",
                                line=dict(color=divider_color, width=1, dash="solid"),
                                showlegend=False,
                                hoverinfo="skip",
                                visible=(group == group_name)
                            ))

                        # Add validation rectangle
                        fig.add_trace(go.Scatter(
                            x=[val_start, val_start, val_end, val_end, val_start],
                            y=[fold_y_min, fold_y_max, fold_y_max, fold_y_min, fold_y_min],
                            fill="toself",
                            fillcolor=val_color,
                            opacity=0.2,
                            line=dict(color=val_color, width=2),
                            mode="lines",
                            showlegend=False,
                            hoverinfo="skip",
                            visible=(group == group_name)
                        ))

                        # Add validation annotation
                        val_center_x = val_start + (val_end - val_start) / 2
                        font_size = min(24, max(14, int(fold_height / 10)))

                        fig.add_trace(go.Scatter(
                            x=[val_center_x],
                            y=[fold_center_y],
                            mode="text",
                            text=[f"VAL {i+1}"],
                            textfont=dict(size=font_size, color=val_color, family="Arial Black"),
                            showlegend=False,
                            hoverinfo="skip",
                            visible=(group == group_name)
                        ))

                        # Add dividing lines around validation rectangle
                        fig.add_trace(go.Scatter(
                            x=[val_start, val_start],
                            y=[fold_y_min, fold_y_max],
                            mode="lines",
                            line=dict(color=divider_color, width=1, dash="solid"),
                            showlegend=False,
                            hoverinfo="skip",
                            visible=(group == group_name)
                        ))
                        fig.add_trace(go.Scatter(
                            x=[val_end, val_end],
                            y=[fold_y_min, fold_y_max],
                            mode="lines",
                            line=dict(color=divider_color, width=1, dash="solid"),
                            showlegend=False,
                            hoverinfo="skip",
                            visible=(group == group_name)
                        ))

        # Set title
        if title is None:
            title = f"<b>Time Series Cross-Validation Split</b><br><span style='font-size:14px'>Group: {group_name}</span>"

        # Create dropdown menu
        dropdown_buttons = []
        for group in all_groups:
            # Create visibility list for this group
            visibility = [False] * len(fig.data)

            # Find all traces that belong to this group
            for i, trace in enumerate(fig.data):
                if trace.name == group:
                    visibility[i] = True
                elif hasattr(trace, 'visible') and trace.visible == (group == group_name):
                    visibility[i] = True

            # Create button for this group
            dropdown_buttons.append(
                dict(
                    label=group,
                    method="update",
                    args=[
                        {"visible": visibility},
                        {
                            "title": f"<b>Time Series Cross-Validation Split</b><br><span style='font-size:14px'>Group: {group}</span>",
                            "yaxis.range": [
                                group_y_ranges[group][0] - group_y_ranges[group][2] * 0.05,
                                group_y_ranges[group][1] + group_y_ranges[group][2] * 0.05
                            ]
                        }
                    ]
                )
            )

        # Update layout with ROC-AUC inspired styling
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=20, color=text_color)
            ),
            xaxis=dict(
                title="<b>Date & Time</b>",
                gridcolor=grid_color,
                title_font=dict(size=16, color=text_color),
                tickfont=dict(color=text_color),
                showline=False,
                zeroline=False,
                showspikes=True,
                spikecolor=crosshair_color,
                spikethickness=1,
                spikedash="dot",
                spikemode="across"
            ),
            yaxis=dict(
                title=f"<b>{y_title}</b>",
                gridcolor=grid_color,
                title_font=dict(size=16, color=text_color),
                tickfont=dict(color=text_color),
                range=[
                    group_y_ranges[group_name][0] - group_y_ranges[group_name][2] * 0.05,
                    group_y_ranges[group_name][1] + group_y_ranges[group_name][2] * 0.05
                ],
                showline=True,
                linecolor=grid_color,
                zeroline=False,
                showspikes=True,
                spikecolor=crosshair_color,
                spikethickness=1,
                spikedash="dot",
                spikemode="across"
            ),
            hovermode='x unified',
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            height=height,
            width=width,
            margin=dict(l=80, r=50, t=100, b=80),
            font=dict(family="Arial", color=text_color),
            showlegend=False,
            updatemenus=[
                dict(
                    buttons=dropdown_buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.02,
                    xanchor="left",
                    y=0.98,
                    yanchor="top",
                    bgcolor=dropdown_bg,
                    bordercolor=text_color,
                    borderwidth=1,
                    font=dict(color=dropdown_text, size=12),
                    active=all_groups.index(group_name)
                )
            ]
        )

        # Update layout with ROC-AUC inspired styling
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=20, color=text_color)
            ),
            xaxis=dict(
                title="<b>Date & Time</b>",
                gridcolor=grid_color,
                title_font=dict(size=16, color=text_color),
                tickfont=dict(color=text_color),
                showline=False,
                zeroline=False,
                showspikes=True,
                spikecolor=crosshair_color,
                spikethickness=1,
                spikedash="dot",
                spikemode="across"
            ),
            yaxis=dict(
                title=f"<b>{y_title}</b>",
                gridcolor=grid_color,
                title_font=dict(size=16, color=text_color),
                tickfont=dict(color=text_color),
                range=[
                    group_y_ranges[group_name][0] - group_y_ranges[group_name][2] * 0.05,
                    group_y_ranges[group_name][1] + group_y_ranges[group_name][2] * 0.05
                ],
                showline=True,
                linecolor=grid_color,
                zeroline=False,
                showspikes=True,
                spikecolor=crosshair_color,
                spikethickness=1,
                spikedash="dot",
                spikemode="across"
            ),
            hovermode='x unified',
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            height=height,
            width=width,
            margin=dict(l=80, r=50, t=100, b=80),
            font=dict(family="Arial", color=text_color),
            showlegend=False,
            updatemenus=[
                dict(
                    buttons=dropdown_buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.02,
                    xanchor="left",
                    y=0.98,
                    yanchor="top",
                    bgcolor=dropdown_bg,
                    bordercolor=text_color,
                    borderwidth=1,
                    font=dict(color=dropdown_text, size=12),
                    active=all_groups.index(group_name)
                )
            ]
        )

        return fig
