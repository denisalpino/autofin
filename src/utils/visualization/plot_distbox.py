from typing import Literal, Tuple

import numpy as np
from pandas import DataFrame
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde


def plot_distbox(
    data: DataFrame,
    width: int = 1400,
    height: int = 1200,
    theme: Literal["dark", "light"] = "dark",
    subplots_grid: Tuple[int, int] = (2, 2),
    bins: int = 50,
    show_kde: bool = True,
    show_hist: bool = True,
    show_stats: bool = True
) -> go.Figure:

    tickers = list(data['ticker'].unique())
    n_tickers = len(tickers)
    rows, cols = subplots_grid

    # Цветовые переменные как HEX (без словарных имен, чтобы hex_to_rgb всегда работал)
    if theme == 'dark':
        bg_color = '#121212'
        grid_color = 'rgba(255, 255, 255, 0.08)'
        text_color = '#FFFFFF'
        kde_color = '#ff7f0e'
        legend_bgcolor = 'rgba(0, 0, 0, 0.8)'
        legend_bordercolor = '#FFFFFF'
    else:
        bg_color = '#FFFFFF'
        grid_color = 'rgba(0, 0, 0, 0.08)'
        text_color = '#000000'
        kde_color = '#f58518'
        legend_bgcolor = 'rgba(255, 255, 255, 0.9)'
        legend_bordercolor = '#000000'

    # Заголовки субплотов — учитываем, что у нас двойная сетка (гист+бокс)
    subplot_titles = []
    total_rows = rows * 2
    for r in range(1, total_rows + 1):
        for c in range(1, cols + 1):
            if r % 2 == 1:
                ticker_index = ((r - 1) // 2) * cols + (c - 1)
                if ticker_index < n_tickers:
                    subplot_titles.append(f"<b>{tickers[ticker_index]}</b>")
                else:
                    subplot_titles.append("")
            else:
                subplot_titles.append("")

    fig = make_subplots(
        rows=total_rows,
        cols=cols,
        row_heights=[2, 0.6] * rows,
        vertical_spacing=0.045,
        horizontal_spacing=0.06,
        subplot_titles=subplot_titles
    )

    colors = pc.qualitative.Set3

    for i, ticker in enumerate(tickers):
        ticker_data = data[data['ticker'] == ticker]['returns'].dropna()  # pyright: ignore[reportAttributeAccessIssue]
        if len(ticker_data) == 0:
            continue

        mean_val = float(ticker_data.mean())
        variance_val = float(ticker_data.var())  # pandas default ddof=1
        skewness_val = float(ticker_data.skew())
        kurtosis_val = float(ticker_data.kurtosis())

        row_asset = i // cols
        col_asset = i % cols

        hist_row = row_asset * 2 + 1
        box_row = row_asset * 2 + 2

        hist, bin_edges = np.histogram(ticker_data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        t_min = float(ticker_data.min())
        t_max = float(ticker_data.max())
        t_range = t_max - t_min
        if t_range == 0 or np.isnan(t_range):
            pad = abs(t_min) * 0.05 if t_min != 0 else 1e-3
        else:
            pad = 0.05 * t_range
        x_min = t_min - pad
        x_max = t_max + pad

        # KDE (безопасно — если данные вырожденные, gaussian_kde может бросить ошибку)
        try:
            kde = gaussian_kde(ticker_data)
            x_kde = np.linspace(x_min, x_max, 1000)
            y_kde = kde(x_kde)
        except Exception:
            x_kde = np.linspace(x_min, x_max, 100)
            y_kde = np.zeros_like(x_kde)

        y_k_max = float(np.max(y_kde)) if len(y_kde) > 0 else 0.0
        y_k_limit = y_k_max * 1.1 if y_k_max > 0 else 1.0

        bar_color = colors[i % len(colors)]
        box_color = colors[i % len(colors)]

        # Гистограмма
        if show_hist:
            fig.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=hist,
                    name='Histogram' if i == 0 else None,
                    marker=dict(color=bar_color, line=dict(width=0)),
                    opacity=0.6,
                    showlegend=(i == 0),
                    legendgroup='histogram',
                    hovertemplate='Return: %{x:.6f}<br>Density: %{y:.6f}<extra></extra>'
                ),
                row=hist_row, col=col_asset + 1
            )

        # KDE (с прозрачной заливкой)
        if show_kde:
            fillcolor = f'rgba{pc.hex_to_rgb(kde_color) + (0.25,)}'
            fig.add_trace(
                go.Scatter(
                    x=x_kde,
                    y=y_kde,
                    mode='lines',
                    name='KDE' if i == 0 else None,
                    line=dict(color=kde_color, width=2.5),
                    fill='tozeroy' if show_hist else 'tonexty',
                    fillcolor=fillcolor,
                    showlegend=(i == 0),
                    legendgroup='kde',
                    hovertemplate='Return: %{x:.6f}<br>Density: %{y:.6f}<extra></extra>'
                ),
                row=hist_row, col=col_asset + 1
            )

        # Форматирование статистик
        def format_stat_value(val):
            if np.isnan(val):
                return "NaN"
            if abs(val) < 1e-6:
                return f"{val:.2e}"
            elif abs(val) < 0.0001:
                return f"{val:.2e}"
            elif abs(val) < 0.01:
                return f"{val:.6f}"
            elif abs(val) < 1:
                return f"{val:.4f}"
            else:
                return f"{val:.2f}"

        if show_stats:
            stats_text = (
                f"<b>{ticker}</b><br>"
                f"Mean: {format_stat_value(mean_val)}<br>"
                f"Var: {format_stat_value(variance_val)}<br>"
                f"Skew: {format_stat_value(skewness_val)}<br>"
                f"Kurt: {format_stat_value(kurtosis_val)}"
            )

            # Номер x/y-оси для данного гіст-плота (используется для ссылок на оси)
            hist_xaxis_num = (hist_row - 1) * cols + (col_asset + 1)

            # Добавляем аккуратную аннотацию (маленькое окошко) в правый верхний угол каждого гіст-субплота.
            fig.add_annotation(
                xref=f"x{hist_xaxis_num}",
                yref=f"y{hist_xaxis_num}",
                x= x_max - (x_max - x_min) * 0.02,  # близко к правому краю данных
                y= y_k_limit * 0.98 if y_k_limit > 0 else 0.98,   # близко к верхнему краю плотности
                xanchor='right',
                yanchor='top',
                text=stats_text,
                showarrow=False,
                bgcolor=f'rgba{pc.hex_to_rgb(bg_color) + (0.85,)}',
                bordercolor=text_color,
                borderwidth=1,
                borderpad=6,
                font=dict(size=10, color=text_color, family="Arial"),
                row=hist_row, col=col_asset + 1
            )

        # Боксплот горизонтальный под гистограммой
        fig.add_trace(
            go.Box(
                x=ticker_data,
                name=ticker,
                boxpoints='outliers',
                notched=True,
                marker=dict(color=box_color, size=4, opacity=0.7, line=dict(width=0)),
                line=dict(color=box_color, width=1.0),
                showlegend=False,
                orientation='h',
                hovertemplate=(
                    'Value: %{x:.6f}<br>'
                    '<extra></extra>'
                )
            ),
            row=box_row, col=col_asset + 1
        )

        hist_xaxis_num = (hist_row - 1) * cols + (col_asset + 1)

        # Синхронизация диапазонов x у графика и бокса
        fig.update_xaxes(range=[x_min, x_max], row=hist_row, col=col_asset + 1)
        fig.update_xaxes(range=[x_min, x_max], row=box_row, col=col_asset + 1)
        fig.update_xaxes(matches=f"x{hist_xaxis_num}", row=box_row, col=col_asset + 1)

        # Убираем вертикальные линии сетки для боксплота
        fig.update_xaxes(showgrid=False, row=box_row, col=col_asset + 1)

        # Настройка осей Y
        fig.update_yaxes(range=[0, y_k_limit], title_text="Density" if (hist_row == 1 and col_asset + 1 == 1) else "", gridcolor=grid_color, row=hist_row, col=col_asset + 1)
        fig.update_yaxes(showticklabels=False, row=box_row, col=col_asset + 1)

    # Легенда располагаем над заголовком — центрируем и поднимаем чуть выше.
    fig.update_layout(
        title=dict(text="<b>Cryptocurrency Returns Distribution Analysis</b>", x=0.5, xanchor='center', y=0.975, font=dict(size=24, color=text_color, family="Arial Black")),
        width=width,
        height=height,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color, family="Arial"),
        margin=dict(l=50, r=50, t=160, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,          # поднята над заголовком
            xanchor="center",
            x=0.5,
            bgcolor=legend_bgcolor,
            bordercolor=legend_bordercolor,
            borderwidth=1,
            itemclick="toggleothers",
            itemdoubleclick="toggle",
            font=dict(size=12, color=text_color)
        ),
        hovermode='closest',
        dragmode='zoom'
    )

    return fig
