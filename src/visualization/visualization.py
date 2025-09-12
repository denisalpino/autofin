import plotly.graph_objects as go


def show_price_line(
    df,
    col: str = "close",
    dt_col: str = "timestamp"
):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[dt_col], y=df[col],
        name="AAVE/USDT",
        yaxis="y1"
    ))
    fig.update_layout(
        title=f"<b>Price AAVE/USDT ({col})<b>",
        title_x=0.5,
        xaxis=dict(title="<b>Date & Time<b>"),
        yaxis=dict(
            title="<b>US Dollars, $<b>",
            side="left",
            showgrid=False
        ),
        height=600,
        template="plotly_dark"
    )

    fig.show()

    return fig

def show_candlestick(
    df,
    OHLC_cols: list[str] = ["open", "high", "low", "close"],
    dt_col: str = "timestamp"
):
    open, high, low, close = OHLC_cols
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df[dt_col],
                open=df[open],
                high=df[high],
                low=df[low],
                close=df[close],
                name="Candlestick"
            )
        ]
    )

    # Настройка осей и заголовков
    fig.update_layout(
        title="<b>Candlestick of AAVE/USDT<b>",
        title_x=0.5,
        xaxis_title="<b>Date & Time<b>",
        yaxis_title="<b>Price, $<b>",
        xaxis_rangeslider_visible=False,  # Скрыть ползунок масштаба
        template="plotly_dark",
        width=1500,  # Ширина графика
        height=800  # Увеличенная высота
    )

    # Отображение графика
    fig.show()