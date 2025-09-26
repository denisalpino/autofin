from collections.abc import Generator
from typing import Dict, Literal, Optional, Union
import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ...config.schemas.splitting import SplitIndices, SplitResult


class GroupTimeSeriesSplit:
    """
    Time-series cross-validation splitter with group support.

    This class implements time-series cross-validation that preserves the temporal
    order of data within each group. It supports both expanding and rolling window
    approaches for training data, with configurable validation and test periods,
    and padding between validation samples.

    Key Features:
    - Group-aware splitting: Each group is processed independently
    - Multiple validation folds: Generate multiple consecutive validation periods
    - Test set support: Optionally reserve a fixed test period
    - Two window strategies: Expanding (all past data) or rolling (fixed window)
    - Padding: Configurable gaps between validation samples
    - Visualizations: Built-in plotting capabilities for split analysis

    The splitter ensures that:
    1. Validation and test sets never overlap
    2. Temporal order is preserved (no future data leaks into past)
    3. All samples from a group appear in the same split

    Examples
    --------
    Basic usage with padding between validation folds:
    >>> cv = GroupTimeSeriesSplit(k_folds=3, val_interval='7d', padding='2d', window="rolling)
    >>> results = cv.split(X, y, groups, timestamps)
    """

    def __init__(
        self,
        k_folds: int = 1,
        test_interval: Optional[str] = None,
        val_interval: str = "7d",
        train_interval: Optional[str] = None,
        window: Literal["expanding", "rolling"] = "expanding",
        min_train_samples: int = 1,
        padding: Optional[str] = None
    ) -> None:
        """
        Initialize the time-series cross-validation splitter.

        Parameters
        ----------
        k_folds : int, default=1
            Number of consecutive validation folds to generate per group.
            Each validation fold covers one interval period.
            Set to 0 to skip validation (only test splits will be generated).

        test_interval : str, optional
            Time interval for test data. If provided, reserves this period
            at the end of each group's timeline for testing.
            Supported units: 'm' (minutes), 'h' (hours), 'd' (days), 'M' (months)

        val_interval : str, default="7d"
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

        padding : str, optional
            Time interval for padding between validation folds. This creates gaps
            between consecutive validation periods. Supported units same as val_interval.
            Useful for creating independent validation sets with temporal separation.

        Raises
        ------
        ValueError
            If k_folds is negative, or if rolling window is used
            without specifying train_interval.
        """
        self._k_folds = k_folds
        self._test_interval = test_interval
        self._offset = self._parse_interval(val_interval)
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

        # Parse padding interval if provided
        if padding:
            self._padding = self._parse_interval(padding)
        else:
            self._padding = None

        # Validate parameters
        if k_folds < 0:
            raise ValueError("k_folds must be a non-negative integer")

        if window == "rolling" and train_interval is None:
            raise ValueError("train_interval must be specified for rolling window")

    def _parse_interval(self, s: str) -> pd.Timedelta:
        """Parse time interval string into pandas offset object."""
        n, unit = int(s[:-1]), s[-1]
        if unit == 'm': return pd.Timedelta(minutes=n)
        if unit == 'h': return pd.Timedelta(hours=n)
        if unit == 'd': return pd.Timedelta(days=n)
        if unit == 'M': return pd.Timedelta(days=30*n)
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
        total_needed = self._offset * self._k_folds

        # Add padding between folds if specified
        if self._padding and self._k_folds > 1:
            total_needed += self._padding * (self._k_folds - 1)

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

        # Calculate total offset including padding
        if self._padding:
            total_offset = steps * self._offset + (steps - 1) * self._padding
        else:
            total_offset = steps * self._offset

        return t_end - total_offset

    def _get_fold_indices(
        self,
        df: pd.DataFrame,
        start: pd.Timestamp,
        steps: int,
        group_name: str
    ) -> Generator[SplitIndices, None, None]:
        """Generate indices for a single group's folds."""
        for k in range(steps):
            # Calculate validation start time with padding between folds
            if self._padding:
                sv = start + k * (self._offset + self._padding)
            else:
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
                train_indices=train_idx,
                validation_indices=val_idx,
                test_indices=None,
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
                        train_indices=train_val_df['_idx'].tolist(),
                        validation_indices=None,
                        test_indices=test_df['_idx'].tolist(),
                        group=group_name
                    )

            # Validate time range for this group
            self._validate_time_range(gdf['ts'])

            # Generate validation folds
            if self._k_folds > 0 and len(train_val_df) > 0:
                # Calculate start point for validation folds including padding
                if self._padding:
                    total_val_range = self._k_folds * self._offset + (self._k_folds - 1) * self._padding
                else:
                    total_val_range = self._k_folds * self._offset

                start_val = train_val_df['ts'].iloc[-1] - total_val_range

                # Generate folds
                validation_splits = list(self._get_fold_indices(
                    train_val_df, start_val, self._k_folds, group_name
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
        """
        # Theme configuration (более контрастные цвета)
        if theme == "dark":
            bg_color = '#121212'
            text_color = 'white'
            grid_color = 'rgba(255, 255, 255, 0.1)'
            train_color = '#3366CC'
            val_color = '#FF9933'
            test_color = '#FF3333'
            line_color = 'rgba(255, 255, 255, 0.8)'
            dropdown_bg = '#2c2c2c'
            dropdown_text = 'white'
            active_bg = '#404040'
            hover_bg = '#3a5bb8'
            crosshair_color = 'rgba(255, 255, 255, 0.5)'
            significant_color = '#FFC107'
            legend_bg = 'rgba(30, 30, 30, 0.9)'
            legend_border = '#FFC107'
            padding_color = 'rgba(200, 200, 200, 0.3)'  # New color for padding areas
        else:
            bg_color = 'white'
            text_color = 'black'
            grid_color = 'rgba(0, 0, 0, 0.1)'
            train_color = '#3366CC'
            val_color = '#FF9933'
            test_color = '#FF3333'
            line_color = 'rgba(0, 0, 0, 0.8)'
            dropdown_bg = '#f0f0f0'
            dropdown_text = 'black'
            active_bg = '#d0d0d0'
            hover_bg = '#3a5bb8'
            crosshair_color = 'rgba(0, 0, 0, 0.5)'
            significant_color = '#FF8F00'
            legend_bg = 'rgba(240, 240, 240, 0.9)'
            legend_border = '#FF8F00'
            padding_color = 'rgba(100, 100, 100, 0.2)'  # New color for padding areas

        # Convert timestamps to Series if it's a DatetimeIndex
        if isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.Series(timestamps)

        # Validate inputs
        if groups is None or timestamps is None:
            raise ValueError("groups and timestamps must be provided")

        # Get splits for all groups
        splits_dict = self.split(X=None, y=None, groups=groups, timestamps=timestamps)

        # Determine which group to display initially
        if group_name is None:
            group_name = list(splits_dict.keys())[0]

        if group_name not in splits_dict:
            raise ValueError(f"Group '{group_name}' not found in the split results")

        # Create a mapping from global index to group-specific index
        global_to_local_idx = {}
        group_y_ranges = {}
        group_data = {}

        for group in splits_dict.keys():
            group_mask = groups == group
            group_indices = groups[group_mask].index
            global_to_local_idx[group] = {global_idx: local_idx for local_idx, global_idx in enumerate(group_indices)}

            # Get group data
            group_timestamps = timestamps[group_mask]
            if y is not None:
                group_y = y[group_mask]
            else:
                group_y = pd.Series(range(len(group_timestamps)), index=group_timestamps.index)

            # Sort by timestamp
            sorted_idx = group_timestamps.argsort()
            group_timestamps = group_timestamps.iloc[sorted_idx]
            group_y = group_y.iloc[sorted_idx]

            # Calculate group-specific y range
            if y is not None:
                group_y_min = group_y.min()
                group_y_max = group_y.max()
                group_y_range = group_y_max - group_y_min
            else:
                group_y_min = 0
                group_y_max = len(group_indices) - 1
                group_y_range = group_y_max - group_y_min

            group_y_ranges[group] = (group_y_min, group_y_max, group_y_range)
            group_data[group] = (group_timestamps, group_y)

        # Create plot
        fig = go.Figure()

        # Store trace indices for each group
        group_trace_indices = {}

        for group_idx, group in enumerate(splits_dict.keys()):
            group_timestamps, group_y = group_data[group]
            group_y_min, group_y_max, group_y_range = group_y_ranges[group]

            # Store starting index for this group's traces
            group_trace_indices[group] = len(fig.data)

            # Add main trace for this group
            fig.add_trace(go.Scatter(
                x=group_timestamps,
                y=group_y,
                mode='lines',
                name=group,
                line=dict(color=line_color, width=2),
                marker=dict(size=4),
                visible=(group == group_name),
                hovertemplate=(
                    '<b>Group</b>: ' + group + '<br>' +
                    '<b>Date</b>: %{x|%Y-%m-%d %H:%M:%S}<br>' +
                    '<b>Value</b>: %{y:.4f}<extra></extra>'
                ),
                showlegend=False
            ))

            # Get split results for this group
            group_result = splits_dict[group]

            # Calculate the minimum time interval between data points
            if len(group_timestamps) > 1:
                time_diffs = group_timestamps.sort_values().diff().dropna()
                min_interval = time_diffs.min()
            else:
                min_interval = pd.Timedelta(days=1)

            # Calculate total number of validation folds
            total_k_folds = len(group_result.validation_splits)
            has_test = group_result.train_test_split and group_result.train_test_split.test_indices

            # Add test split if exists
            if has_test:
                test_idx = group_result.train_test_split.test_indices
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
                    opacity=0.4,
                    line=dict(color=test_color, width=2),
                    mode="lines",
                    name="Test",
                    legendgroup="test",
                    showlegend=(group == group_name),
                    hoverinfo="skip",
                    visible=(group == group_name)
                ))

            # Add validation and training folds with vertical spacing
            if total_k_folds > 0:
                # Add vertical spacing between folds (5% of total range)
                spacing = group_y_range * 0.05
                available_height = group_y_range - spacing * (total_k_folds - 1)
                fold_height = available_height / total_k_folds

                for i, split in enumerate(group_result.validation_splits):
                    if split.validation_indices:
                        # Calculate vertical position with spacing
                        fold_y_min = group_y_min + i * (fold_height + spacing)
                        fold_y_max = fold_y_min + fold_height
                        fold_center_y = fold_y_min + fold_height / 2

                        # Convert global indices to group-specific indices
                        val_idx_local = [global_to_local_idx[group][idx] for idx in split.validation_indices if idx in global_to_local_idx[group]]
                        val_timestamps = group_timestamps.iloc[val_idx_local]
                        val_start = val_timestamps.min()
                        val_end = val_timestamps.max() + min_interval

                        # Add padding visualization if padding is used
                        if self._padding and i > 0:
                            # Calculate padding area between current and previous fold
                            prev_split = group_result.validation_splits[i-1]
                            prev_val_idx_local = [global_to_local_idx[group][idx] for idx in prev_split.validation_indices if idx in global_to_local_idx[group]]
                            prev_val_timestamps = group_timestamps.iloc[prev_val_idx_local]
                            prev_val_end = prev_val_timestamps.max() + min_interval

                            padding_start = prev_val_end
                            padding_end = val_start

                            # Only add padding visualization if there's a significant gap
                            if padding_end > padding_start:
                                fig.add_trace(go.Scatter(
                                    x=[padding_start, padding_start, padding_end, padding_end, padding_start],
                                    y=[fold_y_min, fold_y_max, fold_y_max, fold_y_min, fold_y_min],
                                    fill="toself",
                                    fillcolor=padding_color,
                                    opacity=0.3,
                                    line=dict(color=padding_color, width=1, dash='dot'),
                                    mode="lines",
                                    name="Padding",
                                    legendgroup="padding",
                                    showlegend=(group == group_name and i == 1),  # Show only once per group
                                    hoverinfo="skip",
                                    visible=(group == group_name)
                                ))

                        # Add training rectangle
                        if split.train_indices:
                            train_idx_local = [global_to_local_idx[group][idx] for idx in split.train_indices if idx in global_to_local_idx[group]]
                            train_timestamps = group_timestamps.iloc[train_idx_local]
                            train_start = train_timestamps.min()
                            train_end = train_timestamps.max() + min_interval

                            fig.add_trace(go.Scatter(
                                x=[train_start, train_start, train_end, train_end, train_start],
                                y=[fold_y_min, fold_y_max, fold_y_max, fold_y_min, fold_y_min],
                                fill="toself",
                                fillcolor=train_color,
                                opacity=0.4,
                                line=dict(color=train_color, width=2),
                                mode="lines",
                                name="Train",
                                legendgroup="train",
                                showlegend=(group == group_name and i == 0),
                                hoverinfo="skip",
                                visible=(group == group_name)
                            ))

                        # Add validation rectangle
                        fig.add_trace(go.Scatter(
                            x=[val_start, val_start, val_end, val_end, val_start],
                            y=[fold_y_min, fold_y_max, fold_y_max, fold_y_min, fold_y_min],
                            fill="toself",
                            fillcolor=val_color,
                            opacity=0.4,
                            line=dict(color=val_color, width=2),
                            mode="lines",
                            name="Validation",
                            legendgroup="validation",
                            showlegend=(group == group_name and i == 0),
                            hoverinfo="skip",
                            visible=(group == group_name)
                        ))

                        # Add fold number annotation as a separate trace
                        time_range = group_timestamps.max() - group_timestamps.min()
                        text_x = group_timestamps.min() - time_range * 0.01
                        text_y = max(group_y_min, min(group_y_max, fold_center_y))

                        fig.add_trace(go.Scatter(
                            x=[text_x],
                            y=[text_y],
                            mode="text",
                            text=[f"Fold {i+1}"],
                            textfont=dict(size=12, color=text_color, family="Arial"),
                            textposition="middle center",
                            showlegend=False,
                            hoverinfo="skip",
                            visible=(group == group_name)
                        ))

        # Create dropdown menu
        dropdown_buttons = []

        for group in splits_dict.keys():
            # Create visibility list for this group
            visibility = [False] * len(fig.data)

            # Get the trace indices for this group
            start_idx = group_trace_indices[group]
            next_group_idx = None

            # Find the starting index of the next group
            group_list = list(splits_dict.keys())
            current_index = group_list.index(group)
            if current_index + 1 < len(group_list):
                next_group = group_list[current_index + 1]
                next_group_idx = group_trace_indices[next_group]
            else:
                next_group_idx = len(fig.data)

            # Set visibility for this group's traces
            for i in range(start_idx, next_group_idx):
                visibility[i] = True

            # Update legend visibility for all legend groups
            legend_visibility = {'train': False, 'validation': False, 'test': False, 'padding': False}
            for i in range(start_idx, min(next_group_idx, len(fig.data))):
                trace = fig.data[i]
                if hasattr(trace, 'legendgroup'):
                    if trace.legendgroup == 'train' and not legend_visibility['train']:
                        trace.showlegend = True
                        legend_visibility['train'] = True
                    elif trace.legendgroup == 'validation' and not legend_visibility['validation']:
                        trace.showlegend = True
                        legend_visibility['validation'] = True
                    elif trace.legendgroup == 'test' and not legend_visibility['test']:
                        trace.showlegend = True
                        legend_visibility['test'] = True
                    elif trace.legendgroup == 'padding' and not legend_visibility['padding']:
                        trace.showlegend = True
                        legend_visibility['padding'] = True

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
                            ],
                            "xaxis.range": [
                                group_data[group][0].min() - (group_data[group][0].max() - group_data[group][0].min()) * 0.02,
                                group_data[group][0].max() + (group_data[group][0].max() - group_data[group][0].min()) * 0.02
                            ]
                        }
                    ]
                )
            )

        # Set title
        if title is None:
            padding_info = f" (padding: {self._padding})" if self._padding else ""
            title = f"<b>Time Series Cross-Validation Split{padding_info}</b><br><span style='font-size:14px'>Group: {group_name}</span>"

        # Calculate initial ranges
        initial_timestamps, initial_y = group_data[group_name]
        initial_y_min, initial_y_max, initial_y_range = group_y_ranges[group_name]

        x_range = [
            initial_timestamps.min() - (initial_timestamps.max() - initial_timestamps.min()) * 0.02,
            initial_timestamps.max() + (initial_timestamps.max() - initial_timestamps.min()) * 0.02
        ]

        y_range = [
            initial_y_min - initial_y_range * 0.05,
            initial_y_max + initial_y_range * 0.05
        ]

        # Update layout with consistent styling
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=20, color=text_color),
                y=0.97
            ),
            xaxis=dict(
                title="<b>Date & Time</b>",
                gridcolor=grid_color,
                title_font=dict(size=14, color=text_color),
                tickfont=dict(size=12, color=text_color),
                showline=True,
                linecolor=grid_color,
                zeroline=False,
                showspikes=True,
                spikecolor=crosshair_color,
                spikethickness=1,
                spikedash="dot",
                spikemode="across",
                range=x_range
            ),
            yaxis=dict(
                title=f"<b>{y_title}</b>",
                gridcolor=grid_color,
                title_font=dict(size=14, color=text_color),
                tickfont=dict(size=12, color=text_color),
                range=y_range,
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
            font=dict(family="Arial, sans-serif", color=text_color),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor=legend_bg,
                bordercolor=legend_border,
                borderwidth=1,
                font=dict(size=12, color=text_color),
                itemsizing='constant'
            ),
            hoverlabel=dict(
                bgcolor="rgba(30,30,30,0.9)" if theme == 'dark' else "rgba(255,255,255,0.9)",
                font=dict(color=text_color, size=12),
                bordercolor=significant_color if theme == 'dark' else train_color,
                namelength=-1
            ),
            updatemenus=[
                dict(
                    buttons=dropdown_buttons,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.02,
                    xanchor="left",
                    y=1.08,
                    yanchor="top",
                    bgcolor=dropdown_bg,
                    bordercolor=text_color,
                    borderwidth=1,
                    font=dict(color=dropdown_text, size=12),
                    active=list(splits_dict.keys()).index(group_name)
                )
            ]
        )

        return fig
