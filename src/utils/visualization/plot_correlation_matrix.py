from typing import Literal, Optional
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.preprocessing import LabelEncoder


def plot_correlation_matrix(
    data: pd.DataFrame,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    title: str = "Correlation Matrix",
    theme: Literal["dark", "light"] = "dark",
    significance_threshold: float = 0.01,
    mask_upper: bool = False,
    annotate: bool = True,
    handle_categorical: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> go.Figure:
    """
    Creates beautiful, concise and informative correlation matrix
    using reliable statistical libraries for calculations.
    """

    # Create data copy for processing
    df = data.copy()

    # Handle categorical variables
    if handle_categorical:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns

        if len(categorical_columns) > 0:
            warnings.warn(f"Categorical variables detected: {list(categorical_columns)}. "
                         "Encoding with LabelEncoder.")

            le = LabelEncoder()
            for col in categorical_columns:
                try:
                    if pd.api.types.is_categorical_dtype(df[col]):
                        if 'missing' not in df[col].cat.categories:
                            df[col] = df[col].cat.add_categories('missing')
                        df[col] = df[col].fillna('missing')
                        df[col] = le.fit_transform(df[col].astype(str))
                    else:
                        df[col] = df[col].fillna('missing')
                        df[col] = le.fit_transform(df[col].astype(str))
                except Exception as e:
                    warnings.warn(f"Failed to encode column {col}: {e}")
                    df = df.drop(columns=[col])

    # Remove remaining non-numeric columns
    df = df.select_dtypes(include=[np.number])

    if len(df.columns) < 2:
        raise ValueError("Less than 2 numeric columns remaining after processing")

    # Calculate correlation matrix
    corr_matrix = df.corr(method=method)
    n = len(df)

    # Calculate p-values using scipy
    p_values = pd.DataFrame(np.ones((len(df.columns), len(df.columns))),
                           index=df.columns, columns=df.columns)

    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            if i <= j:
                data1 = df[col1].dropna()
                data2 = df[col2].dropna()

                common_idx = data1.index.intersection(data2.index)
                if len(common_idx) > 1:
                    data1_aligned = data1.loc[common_idx]
                    data2_aligned = data2.loc[common_idx]

                    try:
                        if method == "pearson":
                            corr_val, p_val = pearsonr(data1_aligned, data2_aligned)
                        elif method == "spearman":
                            corr_val, p_val = spearmanr(data1_aligned, data2_aligned)
                        elif method == "kendall":
                            corr_val, p_val = kendalltau(data1_aligned, data2_aligned)

                        p_values.loc[col1, col2] = p_val
                        p_values.loc[col2, col1] = p_val
                    except:
                        pass

    # Mask upper triangle if needed
    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_matrix_masked = corr_matrix.mask(mask)
        p_values_masked = p_values.mask(mask)
    else:
        corr_matrix_masked = corr_matrix
        p_values_masked = p_values

    # Theme configuration
    if theme == 'dark':
        bg_color = '#121212'
        text_color = 'white'
        grid_color = 'rgba(255, 255, 255, 0.1)'
        significant_color = '#FFC107'
        non_significant_color = 'rgba(255, 255, 255, 0.6)'
        color_scale = [
            [0.0, '#B71C1C'],
            [0.45, '#121212'],
            [0.55, '#121212'],
            [1.0, '#1B5E20']
        ]
    else:
        bg_color = 'white'
        text_color = 'black'
        grid_color = 'rgba(0, 0, 0, 0.1)'
        significant_color = '#FF8F00'
        non_significant_color = 'rgba(0, 0, 0, 0.6)'
        color_scale = [
            [0.0, '#D32F2F'],
            [0.45, '#FFFFFF'],
            [0.55, '#FFFFFF'],
            [1.0, '#2E7D32']
        ]

    # Determine figure size
    if width is None:
        width = 600 + len(df.columns) * 15
    if height is None:
        height = 600 + len(df.columns) * 15

    # Create figure
    fig = go.Figure()

    # Calculate statistics for subtitle
    corr_no_diag = corr_matrix.values.copy()
    np.fill_diagonal(corr_no_diag, np.nan)
    corr_no_diag = pd.DataFrame(corr_no_diag, index=corr_matrix.index, columns=corr_matrix.columns)

    strong_pos = (corr_no_diag > 0.7).sum().sum()
    strong_neg = (corr_no_diag < -0.7).sum().sum()

    # Count significant correlations
    significant_count = 0
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            if (not pd.isna(corr_matrix_masked.iloc[i, j]) and
                p_values.iloc[i, j] < significance_threshold and
                i != j and
                (not mask_upper or i >= j)):
                significant_count += 1

    # Prepare annotation text (without colors)
    annotation_text = []
    for i, row_name in enumerate(corr_matrix.index):
        text_row = []
        for j, col_name in enumerate(corr_matrix.columns):
            if pd.isna(corr_matrix_masked.iloc[i, j]):
                text_row.append("")
            else:
                corr_val = corr_matrix.iloc[i, j]
                text_row.append(f"{corr_val:.2f}")
        annotation_text.append(text_row)

    # Prepare hover text
    hover_text = []
    for i, row_name in enumerate(corr_matrix.index):
        hover_row = []
        for j, col_name in enumerate(corr_matrix.columns):
            if pd.isna(corr_matrix_masked.iloc[i, j]):
                hover_row.append("")
            else:
                corr_val = corr_matrix.iloc[i, j]
                p_val = p_values.iloc[i, j]
                significance = "✓" if p_val < significance_threshold and i != j else "✗"
                stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

                hover_text_str = (
                    f"<b>{row_name} vs {col_name}</b><br>"
                    f"Correlation: {corr_val:.3f}<br>"
                    f"P-value: {p_val:.4f} {stars}<br>"
                    f"Significance: {significance} (p < {significance_threshold})<br>"
                    f"Method: {method}<br>"
                    f"Observations: {n}"
                )
                hover_row.append(hover_text_str)
        hover_text.append(hover_row)

    # Main heatmap (without text colors)
    heatmap = go.Heatmap(
        z=corr_matrix_masked.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=color_scale,
        zmin=-1,
        zmax=1,
        hoverinfo="text",
        text=annotation_text if annotate else None,
        texttemplate="%{text}" if annotate else None,
        textfont=dict(size=12, color=text_color),  # Единый цвет для всего текста
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_text,
        colorbar=dict(
            title=dict(text="Correlation", font=dict(color=text_color, size=14)),
            tickfont=dict(color=text_color, size=12),
            ticks="outside",
            len=0.75
        )
    )

    fig.add_trace(heatmap)

    # Add colored annotations for significant correlations
    if annotate:
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                if (not pd.isna(corr_matrix_masked.iloc[i, j]) and
                    i != j and  # Не диагональ
                    p_values.iloc[i, j] < significance_threshold):

                    # Определяем цвет текста в зависимости от значимости
                    font_color = significant_color if p_values.iloc[i, j] < significance_threshold else non_significant_color

                    # Добавляем аннотацию с правильным цветом
                    fig.add_annotation(
                        x=j,
                        y=i,
                        text=annotation_text[i][j],
                        showarrow=False,
                        font=dict(size=12, color=font_color),
                        xref="x",
                        yref="y"
                    )

    # Enhanced subtitle with all statistics
    extended_subtitle = (
        f"Method: {method} | Variables: {len(df.columns)} | "
        f"Observations: {n} | "
        f"Strong correlations (>0.7 | <-0.7): {strong_pos + strong_neg} | "
        f"Significant (p<{significance_threshold}): {significant_count}"
    )

    # Layout configuration
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title}</b><br>"
                f"<span style='font-size:14px; color: {significant_color}'>"
                f"{extended_subtitle}"
                f"</span>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=20, color=text_color),
            y=0.97
        ),
        xaxis=dict(
            tickangle=45,
            gridcolor=grid_color,
            tickfont=dict(size=12, color=text_color),
            side="bottom",
            tickvals=list(range(len(corr_matrix.columns))),
            ticktext=corr_matrix.columns.tolist()
        ),
        yaxis=dict(
            gridcolor=grid_color,
            tickfont=dict(size=12, color=text_color),
            tickvals=list(range(len(corr_matrix.index))),
            ticktext=corr_matrix.index.tolist()
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        height=height,
        width=width,
        margin=dict(l=100, r=50, t=120, b=80),
        font=dict(family="Arial, sans-serif", color=text_color),
        hoverlabel=dict(
            bgcolor="rgba(30,30,30,0.9)" if theme == 'dark' else "rgba(255,255,255,0.9)",
            font=dict(color=text_color, size=16),
            bordercolor=significant_color
        )
    )

    return fig
