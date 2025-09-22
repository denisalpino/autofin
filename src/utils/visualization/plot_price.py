import math
from typing import Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_price(
    df: pd.DataFrame,
    display_mode: Literal["subplots", "dropdown"] = "subplots",
    chart_type: Literal["candlestick", "line"] = "candlestick",
    subplots_grid: Optional[Tuple[int, int]] = None,
    title: str = "",
    theme: Literal["dark", "light"] = "dark",
    height: int = 700,
    width: int = 1200
) -> go.Figure:
    # --- minimal validation ---
    required_cols = ['open', 'high', 'low', 'close', 'timestamps']
    if not all(col in df.columns for col in required_cols):
        missing = set(required_cols) - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # --- theme colors & grid appearance ---
    if theme == "dark":
        bg_color = '#121212'
        card_bg_color = '#1E1E1E'
        text_color = 'white'
        grid_color = 'rgba(255, 255, 255, 0.1)'
        increasing_color = '#4CAF50'
        decreasing_color = '#F44336'
        price_line_color = '#2962FF'
        dropdown_bg = '#2c2c2c'
        dropdown_text = 'white'
        crosshair_color = 'rgba(255,255,255,0.5)'
        volume_increasing_color = 'rgba(76,175,80,0.35)'
        volume_decreasing_color = 'rgba(244,67,54,0.28)'
    else:
        bg_color = 'white'
        card_bg_color = '#F5F5F5'
        text_color = 'black'
        grid_color = 'rgba(0,0,0,0.1)'
        increasing_color = '#2E7D32'
        decreasing_color = '#D32F2F'
        price_line_color = '#1565C0'
        dropdown_bg = '#f0f0f0'
        dropdown_text = 'black'
        crosshair_color = 'rgba(0,0,0,0.5)'
        volume_increasing_color = 'rgba(46,125,50,0.25)'
        volume_decreasing_color = 'rgba(198,40,40,0.2)'

    # --- prepare dataframe ---
    df = df.copy()
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)

    ticker_col = "ticker"
    has_volume = 'volume' in df.columns
    has_multiple_tickers = (ticker_col in df.columns) and (df[ticker_col].nunique() > 1)

    # normalize display mode
    if has_multiple_tickers and display_mode == "subplots":
        mode = "subplots"
    elif has_multiple_tickers and display_mode == "dropdown":
        mode = "dropdown"
    else:
        mode = "single"

    # collect tickers
    if has_multiple_tickers:
        tickers = list(df[ticker_col].unique())
    else:
        tickers = [df[ticker_col].iloc[0]] if ticker_col in df.columns else [None]

    # --- grid sizing ---
    if mode == "subplots":
        if subplots_grid is None:
            n = len(tickers)
            rows = math.ceil(math.sqrt(n))
            cols = math.ceil(n / rows)
            subplots_grid = (rows, cols)
        else:
            rows, cols = subplots_grid
    else:
        rows, cols = (1, 1)

    # row heights
    if mode == "subplots":
        row_heights = [1.0 / rows] * rows
    else:
        row_heights = [0.68, 0.32] if has_volume else [1.0]

    hspace = 0.06 if mode == "subplots" else 0.08
    margin_l = 120
    margin_r = 40
    margin_t = 110
    margin_b = 80

    # Prepare subplot titles with price change for subplots mode
    subplot_titles = []
    if mode == "subplots":
        for ticker in tickers[:rows*cols]:
            ticker_df = df[df[ticker_col] == ticker] if has_multiple_tickers else df
            if not ticker_df.empty:
                last_close = float(ticker_df['close'].iloc[-1])  # pyright: ignore[reportAttributeAccessIssue]
                first_close = float(ticker_df['close'].iloc[0])  # pyright: ignore[reportAttributeAccessIssue]
                price_change = last_close - first_close
                percent_change = (price_change / first_close) * 100 if first_close != 0 else 0.0
                change_symbol = "▲" if price_change >= 0 else "▼"
                change_color = increasing_color if price_change >= 0 else decreasing_color

                # For subplots, include price change in the title
                title_text = f"{ticker} {change_symbol} {percent_change:+.2f}%"
                subplot_titles.append(title_text)
            else:
                subplot_titles.append(str(ticker))
    else:
        subplot_titles = tickers[:rows*cols] if has_multiple_tickers else [None]

    # --- make subplots ---
    if mode == "subplots":
        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes=True,
            vertical_spacing=0.06,
            horizontal_spacing=hspace,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": has_volume} for _ in range(cols)] for _ in range(rows)],
            row_heights=row_heights
        )
    else:
        nrows_plot = 2 if has_volume else 1
        specs = [[{"secondary_y": False}] for _ in range(nrows_plot)]
        fig = make_subplots(
            rows=nrows_plot,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            specs=specs,
            row_heights=row_heights
        )

    # mapping ticker -> trace indices for dropdown
    ticker_trace_map: Dict[object, List[int]] = {t: [] for t in tickers}
    subplot_ticker_map: Dict[int, object] = {}
    price_axis_for_subplot: Dict[int, str] = {}

    def add_trace_and_map(trace, ticker=None, row=1, col=1, secondary_y=False):
        fig.add_trace(trace, row=row, col=col, secondary_y=secondary_y)
        idx = len(fig.data) - 1
        if ticker is not None:
            ticker_trace_map.setdefault(ticker, []).append(idx)
        return idx

    # --- add traces per subplot or single/dropdown (no indicators) ---
    if mode == "subplots":
        n_subplots = rows * cols
        for i, ticker in enumerate(tickers[:n_subplots]):
            ticker_df = df[df[ticker_col] == ticker] if has_multiple_tickers else df
            if ticker_df.empty:
                continue
            row = i // cols + 1
            col = i % cols + 1
            subplot_index = i

            if chart_type == "candlestick":
                price_trace = go.Candlestick(
                    x=ticker_df['timestamps'],
                    open=ticker_df['open'],
                    high=ticker_df['high'],
                    low=ticker_df['low'],
                    close=ticker_df['close'],
                    name=str(ticker),
                    increasing_line_color=increasing_color,
                    decreasing_line_color=decreasing_color,
                    showlegend=False,
                    customdata=ticker_df[['open', 'high', 'low', 'close']].values
                )
                idx_price = add_trace_and_map(price_trace, ticker=ticker, row=row, col=col, secondary_y=False)
            else:
                price_trace = go.Scatter(
                    x=ticker_df['timestamps'],
                    y=ticker_df['close'],
                    mode='lines',
                    name=str(ticker),
                    line=dict(color=price_line_color, width=2),
                    showlegend=False,
                    hovertemplate=(
                        'Open: %{customdata[0]}<br>'
                        'High: %{customdata[1]}<br>'
                        'Low: %{customdata[2]}<br>'
                        'Close: %{customdata[3]}<extra></extra>'
                    ),
                    customdata=ticker_df[['open', 'high', 'low', 'close']].values
                )
                idx_price = add_trace_and_map(price_trace, ticker=ticker, row=row, col=col, secondary_y=False)

            # record mapping: subplot_index -> ticker
            subplot_ticker_map[subplot_index] = ticker
            try:
                yaxis_ref = getattr(fig.data[idx_price], 'yaxis', None) or 'y'
            except Exception:
                yaxis_ref = 'y'
            axis_key = 'yaxis' + (yaxis_ref[1:] if len(yaxis_ref) > 1 else '')
            price_axis_for_subplot[subplot_index] = axis_key

            if has_volume:
                volume_colors = [
                    volume_increasing_color if c >= o else volume_decreasing_color
                    for c, o in zip(ticker_df['close'], ticker_df['open'])
                ]
                vol_trace = go.Bar(
                    x=ticker_df['timestamps'],
                    y=ticker_df['volume'],
                    name=f"{ticker} Volume",
                    marker_color=volume_colors,
                    marker_line_width=0,
                    marker_line_color='rgba(0,0,0,0)',
                    showlegend=False,
                    hovertemplate='Volume: %{y}<extra></extra>'
                )
                add_trace_and_map(vol_trace, ticker=ticker, row=row, col=col, secondary_y=True)

    else:
        # single / dropdown mode
        for idx_ticker, ticker in enumerate(tickers):
            ticker_df = df[df[ticker_col] == ticker] if has_multiple_tickers else df
            if ticker_df.empty:
                continue
            visible = True if idx_ticker == 0 else False

            if chart_type == "candlestick":
                price_trace = go.Candlestick(
                    x=ticker_df['timestamps'],
                    open=ticker_df['open'],
                    high=ticker_df['high'],
                    low=ticker_df['low'],
                    close=ticker_df['close'],
                    name=str(ticker) if has_multiple_tickers else "Price",
                    increasing_line_color=increasing_color,
                    decreasing_line_color=decreasing_color,
                    visible=visible,
                    showlegend=not has_multiple_tickers,
                    customdata=ticker_df[['open', 'high', 'low', 'close']].values
                )
                idx_price = add_trace_and_map(price_trace, ticker=ticker, row=1, col=1, secondary_y=False)
            else:
                price_trace = go.Scatter(
                    x=ticker_df['timestamps'],
                    y=ticker_df['close'],
                    mode='lines',
                    name=str(ticker) if has_multiple_tickers else "Price",
                    line=dict(color=price_line_color, width=2),
                    visible=visible,
                    showlegend=not has_multiple_tickers,
                    hovertemplate=(
                        'Open: %{customdata[0]}<br>'
                        'High: %{customdata[1]}<br>'
                        'Low: %{customdata[2]}<br>'
                        'Close: %{customdata[3]}<extra></extra>'
                    ),
                    customdata=ticker_df[['open', 'high', 'low', 'close']].values
                )
                idx_price = add_trace_and_map(price_trace, ticker=ticker, row=1, col=1, secondary_y=False)

            # map axis
            if 0 not in price_axis_for_subplot:
                try:
                    yaxis_ref = getattr(fig.data[idx_price], 'yaxis', None) or 'y'
                except Exception:
                    yaxis_ref = 'y'
                axis_key = 'yaxis' + (yaxis_ref[1:] if len(yaxis_ref) > 1 else '')
                price_axis_for_subplot[0] = axis_key
                subplot_ticker_map[0] = ticker

            if has_volume:
                volume_colors = [
                    volume_increasing_color if c >= o else volume_decreasing_color
                    for c, o in zip(ticker_df['close'], ticker_df['open'])
                ]
                vol_trace = go.Bar(
                    x=ticker_df['timestamps'],
                    y=ticker_df['volume'],
                    name=f"{ticker} Volume" if has_multiple_tickers else "Volume",
                    marker_color=volume_colors,
                    marker_line_width=0,
                    marker_line_color='rgba(0,0,0,0)',
                    visible=visible,
                    showlegend=False,
                    hovertemplate='Volume: %{y}<extra></extra>'
                )
                add_trace_and_map(vol_trace, ticker=ticker, row=2 if has_volume else 1, col=1, secondary_y=False)

    # --- Get x-axis range for all buttons ---
    x_min = df['timestamps'].min()
    x_max = df['timestamps'].max()
    all_range = [x_min, x_max]

    # --- compute_y_ranges and helpers (unchanged logic) ---
    def compute_y_ranges(start_ts, end_ts):
        relayout = {}
        for subplot_idx, axis_key in price_axis_for_subplot.items():
            ticker = subplot_ticker_map.get(subplot_idx)
            ticker_df = df[df[ticker_col] == ticker] if has_multiple_tickers else df
            mask = (ticker_df['timestamps'] >= start_ts) & (ticker_df['timestamps'] <= end_ts)
            if mask.any():
                low_v = float(ticker_df.loc[mask, 'low'].min())
                high_v = float(ticker_df.loc[mask, 'high'].max())
            else:
                low_v = float(ticker_df['low'].min()) if not ticker_df.empty else 0.0
                high_v = float(ticker_df['high'].max()) if not ticker_df.empty else 1.0

            span = max(high_v - low_v, 1e-8)
            pad = span * 0.02
            y_min = low_v - pad
            y_max = high_v + pad
            relayout[f'{axis_key}.range'] = [y_min, y_max]
            relayout[f'{axis_key}.autorange'] = False
        return relayout

    def compute_y_range_for_ticker_in_xrange(ticker: str, x_range: Union[List, Tuple, None] = None) -> Dict:
        if mode != "dropdown":
            return {}
        if x_range is None or len(x_range) != 2:
            x_range = all_range
        start_ts, end_ts = pd.to_datetime(x_range[0]), pd.to_datetime(x_range[1])
        ticker_df = df[df[ticker_col] == ticker] if has_multiple_tickers else df
        if ticker_df.empty:
            return {}
        mask = (ticker_df['timestamps'] >= start_ts) & (ticker_df['timestamps'] <= end_ts)
        if mask.any():
            low_v = float(ticker_df.loc[mask, 'low'].min())
            high_v = float(ticker_df.loc[mask, 'high'].max())
        else:
            low_v = float(ticker_df['low'].min()) if not ticker_df.empty else 0.0
            high_v = float(ticker_df['high'].max()) if not ticker_df.empty else 1.0
        span = max(high_v - low_v, 1e-8)
        pad = span * 0.02
        y_min = low_v - pad
        y_max = high_v + pad
        price_axis_key = price_axis_for_subplot.get(0, 'y')
        return {
            f'{price_axis_key}.range': [y_min, y_max],
            f'{price_axis_key}.autorange': False
        }

    def create_range_buttons_for_ticker(ticker_for_buttons: str) -> List[Dict]:
        periods = [
            ("1d", pd.DateOffset(days=1)),
            ("1w", pd.DateOffset(weeks=1)),
            ("1m", pd.DateOffset(months=1)),
            ("6m", pd.DateOffset(months=6)),
            ("1y", pd.DateOffset(years=1)),
        ]
        buttons = []
        for label, offset in periods:
            start_ts = x_max - offset
            new_x_range = [start_ts, x_max]
            if mode == "dropdown":
                y_axis_update = compute_y_range_for_ticker_in_xrange(ticker_for_buttons, new_x_range)
                relayout_payload = {'xaxis.range': new_x_range}
                if y_axis_update:
                    relayout_payload.update(y_axis_update)
                else:
                    price_axis_key = price_axis_for_subplot.get(0, 'y')
                    relayout_payload[f'{price_axis_key}.autorange'] = True
            else:
                y_relayout = compute_y_ranges(start_ts, x_max)
                relayout_payload = {**y_relayout, 'xaxis.range': [start_ts, x_max]}
            buttons.append(dict(
                label=label,
                method='relayout',
                args=[relayout_payload]
            ))
        return buttons

    # --- INITIAL range buttons (for first ticker) ---
    initial_range_buttons = create_range_buttons_for_ticker(tickers[0] if tickers else None)

    # --- Initialize y-ranges / autorange depending on mode ---
    if mode == "subplots":
        if price_axis_for_subplot:
            init_relayout = compute_y_ranges(x_min, x_max)
            fig.update_layout(**init_relayout)
    else:
        if 0 in price_axis_for_subplot and tickers:
            first_ticker = tickers[0]
            initial_y_layout = compute_y_range_for_ticker_in_xrange(first_ticker, all_range)
            if initial_y_layout:
                fig.update_layout(**initial_y_layout)
            else:
                price_axis_key = price_axis_for_subplot[0]
                fig.update_layout(**{f"{price_axis_key}.autorange": True})

    # --- attach rangeslider=False on primary x-axis; initial view = all_range ---
    fig.update_layout(
        xaxis=dict(
            gridcolor=grid_color,
            gridwidth=1,
            tickfont=dict(color=text_color, size=12),
            rangeslider=dict(visible=False),
            automargin=False,
            range=all_range
        )
    )
    try:
        fig.update_xaxes(matches='x')
    except Exception:
        pass

    # --- axis styling: unify with ROC look & crosshair for single ---
    if mode == "subplots":
        fig.update_xaxes(
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            showspikes=True,
            spikecolor=crosshair_color,
            spikethickness=1,
            spikedash="dot",
            spikemode="across",
            automargin=False,
            tickfont=dict(color=text_color, size=12),
            range=all_range
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            showspikes=True,
            spikecolor=crosshair_color,
            spikethickness=1,
            spikedash="dot",
            spikemode="across",
            automargin=False,
            tickfont=dict(color=text_color, size=12)
        )
    else:
        fig.update_xaxes(
            showgrid=False,
            showspikes=True,
            spikecolor=crosshair_color,
            spikethickness=1,
            spikedash="dot",
            spikemode="across",
            automargin=True,
            tickfont=dict(color=text_color, size=12),
            range=all_range
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            showspikes=True,
            spikecolor=crosshair_color,
            spikethickness=1,
            spikedash="dot",
            spikemode="across",
            automargin=True,
            tickfont=dict(color=text_color, size=12)
        )

    # --- final layout tweaks (unified title + meta area like ROC) ---
    title_text = f"<b>{title}</b>"

    # Установка начального заголовка с информацией для первого тикера в dropdown режиме
    if mode == "dropdown" and tickers:
        first_ticker = tickers[0]
        ticker_df = df[df[ticker_col] == first_ticker] if has_multiple_tickers else df
        if not ticker_df.empty:
            last_close = float(ticker_df['close'].iloc[-1])
            first_close = float(ticker_df['close'].iloc[0])
            price_change = last_close - first_close
            percent_change = (price_change / first_close) * 100 if first_close != 0 else 0.0
            change_symbol = "▲" if price_change >= 0 else "▼"
            subtitle = f"Ticker: {first_ticker} | {last_close:.4f} {change_symbol} {percent_change:+.2f}%"
            fig.update_layout(
                title=dict(
                    text=f"{title_text}<br><span style='font-size:14px'>{subtitle}</span>",
                    x=0.5,
                    xanchor='center',
                    font=dict(size=20, color=text_color),
                    y=0.95,
                    yanchor="top"
                )
            )
    else:
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor='center',
                font=dict(size=20, color=text_color),
                y=0.95,
                yanchor="top"
            )
        )

    # --- Определение позиции кнопок в зависимости от режима ---
    if mode == "subplots":
        range_menu_x = 0
        range_menu_xanchor = "left"
        range_menu_y = 1.15  # Ниже, чем в dropdown режиме
    else:
        range_menu_x = 0.1
        range_menu_xanchor = "left"
        range_menu_y = 1.05

    # RANGE MENU
    range_menu = dict(
        type='buttons',
        buttons=initial_range_buttons,
        direction='left',
        showactive=False,
        x=range_menu_x,
        xanchor=range_menu_xanchor,
        y=range_menu_y,
        yanchor="top",
        bgcolor=card_bg_color,
        bordercolor=text_color,
        borderwidth=1,
        font=dict(color=text_color, size=11),
    )
    ticker_menu = None

    # --- dropdown controls (tickers) now populated with left placement and switching logic updates range buttons at updatemenus[1] ---
    if mode == "dropdown":
        buttons = []
        total_traces = len(fig.data)
        base_visibility = [False] * total_traces

        for i, ticker in enumerate(tickers):
            vis = base_visibility.copy()
            for idx in ticker_trace_map.get(ticker, []):
                if idx < total_traces:
                    vis[idx] = True

            ticker_df = df[df[ticker_col] == ticker] if has_multiple_tickers else df
            if ticker_df.empty:
                continue
            first_close = float(ticker_df['close'].iloc[0])
            last_close = float(ticker_df['close'].iloc[-1])
            price_change = last_close - first_close
            percent_change = (price_change / first_close) * 100 if first_close != 0 else 0.0
            change_symbol = "▲" if price_change >= 0 else "▼"

            # Формируем подзаголовок с информацией
            subtitle = f"Ticker: {ticker} | {last_close:.4f} {change_symbol} {percent_change:+.2f}%"

            # get current x range
            try:
                current_x_range = fig.layout.xaxis.range
            except (AttributeError, KeyError):
                current_x_range = all_range

            y_axis_update_switch = compute_y_range_for_ticker_in_xrange(ticker, current_x_range)

            layout_update = {
                "title": f"{title_text}<br><span style='font-size:16px'>{subtitle}</span>"
            }
            if y_axis_update_switch:
                layout_update.update(y_axis_update_switch)
            else:
                if 0 in price_axis_for_subplot:
                    price_axis_key = price_axis_for_subplot[0]
                    layout_update[f'{price_axis_key}.autorange'] = True

            # Create new range buttons for the switched-to ticker
            new_range_buttons_for_this_ticker = create_range_buttons_for_ticker(ticker)

            # IMPORTANT: update the second updatemenu (range buttons)
            layout_update[f"updatemenus[1].buttons"] = new_range_buttons_for_this_ticker
            layout_update[f"updatemenus[1].bgcolor"] = card_bg_color
            layout_update[f"updatemenus[1].font.color"] = text_color
            layout_update[f"updatemenus[1].showactive"] = False
            layout_update[f"updatemenus[1].direction"] = "left"
            layout_update[f"updatemenus[1].x"] = range_menu_x
            layout_update[f"updatemenus[1].xanchor"] = range_menu_xanchor
            layout_update[f"updatemenus[1].y"] = range_menu_y
            layout_update[f"updatemenus[1].yanchor"] = "top"

            buttons.append(dict(
                label=str(ticker),
                method="update",
                args=[{"visible": vis}, layout_update]
            ))

        ticker_menu = dict(
            buttons=buttons,
            direction="down",
            showactive=False,
            x=0.01,
            xanchor="left",
            y=range_menu_y,
            yanchor="top",
            bgcolor=dropdown_bg,
            bordercolor=text_color,
            borderwidth=1,
            font=dict(color=dropdown_text, size=12),
        )

    # Combine updatemenus: ticker_menu (left) + range_menu (to its right)
    updatemenus = []
    if ticker_menu is not None:
        updatemenus.append(ticker_menu)
    updatemenus.append(range_menu)
    fig.update_layout(updatemenus=updatemenus)

    # --- final per-subplot y titles for readability ---
    if mode == "subplots":
        plotted = min(len(tickers), rows * cols)
        for i in range(plotted):
            axis_index = i + 1
            yaxis_name = 'yaxis' + (str(axis_index) if axis_index > 1 else '')
            try:
                fig.layout[yaxis_name].update(title=f"Price", titlefont=dict(size=12, color=text_color))
            except Exception:
                pass

    # --- final layout tweaks (unified title + meta area like ROC) ---
    fig.update_layout(
        hovermode='x unified',
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        height=height,
        width=width,
        margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
        font=dict(family="Arial", color=text_color),
        showlegend=False
    )

    return fig
