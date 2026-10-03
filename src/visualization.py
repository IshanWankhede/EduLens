"""Streamlit-free Plotly and Matplotlib figure builders for exploratory analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from src import config

COLORBLIND_SAFE_PALETTE = (
    "#6366F1",
    "#22D3EE",
    "#FBBF24",
    "#D946EF",
    "#34D399",
    "#F97316",
)
MARKER_SYMBOLS = ("circle", "square", "diamond", "triangle-up", "cross", "x")
_TITLE_FONT = "Inter, system-ui, sans-serif"
_TEXT_COLOR = "#E8ECF8"
_MUTED_COLOR = "#A3ADC8"
_GRID_COLOR = "rgba(163, 173, 200, 0.18)"
_MATPLOTLIB_GRID_COLOR = (163 / 255, 173 / 255, 200 / 255, 0.18)


class VisualizationInputError(ValueError):
    """Raised when a chart cannot be built from the supplied data."""


def apply_plotly_theme(
    figure: go.Figure,
    *,
    title: str,
    x_label: str,
    y_label: str,
    caption: str,
) -> go.Figure:
    """Apply the EduLens dark chart style and attach a visible, plain-English caption.

    Colors follow the proposed DESIGN.md midnight/indigo/cyan/amber palette. Plot traces should
    pair colors with markers, category labels, or patterns so a distinction is not conveyed by
    color alone.
    """
    if not title.strip() or not x_label.strip() or not y_label.strip() or not caption.strip():
        raise VisualizationInputError("Every chart requires a title, axis labels, and a caption.")
    annotations = list(figure.layout.annotations or ())
    annotations.append(
        {
            "text": f"<b>How to read this:</b> {caption}",
            "xref": "paper",
            "yref": "paper",
            "x": 0,
            "y": -0.36,
            "xanchor": "left",
            "yanchor": "top",
            "showarrow": False,
            "align": "left",
            "font": {"size": 12, "color": _MUTED_COLOR},
        }
    )
    figure.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"family": _TITLE_FONT}},
        font={"family": _TITLE_FONT, "color": _TEXT_COLOR},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 60, "r": 28, "t": 72, "b": 108},
        legend={"title": {"text": ""}, "orientation": "h", "y": -0.24},
        xaxis={
            "title": {"text": x_label},
            "gridcolor": _GRID_COLOR,
            "zerolinecolor": _GRID_COLOR,
            "color": _MUTED_COLOR,
        },
        yaxis={
            "title": {"text": y_label},
            "gridcolor": _GRID_COLOR,
            "zerolinecolor": _GRID_COLOR,
            "color": _MUTED_COLOR,
        },
        annotations=annotations,
        meta={"caption": caption},
    )
    return figure


def apply_matplotlib_theme(
    figure: plt.Figure,
    axis: plt.Axes,
    *,
    title: str,
    x_label: str,
    y_label: str,
    caption: str,
) -> None:
    """Apply the shared EduLens palette and accessible labels to a static Matplotlib figure."""
    if not title.strip() or not x_label.strip() or not y_label.strip() or not caption.strip():
        raise VisualizationInputError("Every chart requires a title, axis labels, and a caption.")
    axis.set_facecolor("#11172E")
    axis.set_title(title, color=_TEXT_COLOR)
    axis.set_xlabel(x_label, color=_MUTED_COLOR)
    axis.set_ylabel(y_label, color=_MUTED_COLOR)
    axis.tick_params(colors=_MUTED_COLOR)
    axis.grid(color=_MATPLOTLIB_GRID_COLOR, alpha=0.65)
    figure.text(
        0.01,
        -0.04,
        f"How to read this: {caption}",
        color=_MUTED_COLOR,
        fontsize=9,
        wrap=True,
    )


def _require_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    """Validate that a non-empty DataFrame contains all requested columns."""
    if data.empty:
        raise VisualizationInputError("Cannot create a chart from an empty dataset.")
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise VisualizationInputError(f"Chart column(s) missing: {', '.join(missing)}.")


def _numeric_values(data: pd.DataFrame, column: str) -> pd.Series:
    """Return valid finite numeric observations or raise a friendly chart-input exception."""
    _require_columns(data, [column])
    series = data[column]
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise VisualizationInputError(f"Chart column '{column}' must be numeric.")
    values = series.dropna()
    if values.empty:
        raise VisualizationInputError(f"Chart column '{column}' has no valid observations.")
    values = values[np.isfinite(values)]
    if values.empty:
        raise VisualizationInputError(f"Chart column '{column}' has no finite observations.")
    return values


def _display_name(column: str) -> str:
    """Return a readable chart label for a column name."""
    return column.replace("_", " ").strip()


def histogram_plot(
    data: pd.DataFrame,
    column: str,
    *,
    bins: int | None = None,
    title: str | None = None,
    caption: str | None = None,
) -> go.Figure:
    """Build an interactive histogram for numeric data, retaining repeated discrete values."""
    values = _numeric_values(data, column)
    if bins is not None and bins < 1:
        raise VisualizationInputError("bins must be a positive integer.")
    figure = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=bins,
            marker={"color": COLORBLIND_SAFE_PALETTE[0], "pattern": {"shape": "/"}},
            name=_display_name(column),
        )
    )
    return apply_plotly_theme(
        figure,
        title=title or f"Distribution of {_display_name(column)}",
        x_label=_display_name(column),
        y_label="Count",
        caption=caption
        or "Each bar counts observations within a value interval; repeated values contribute to the same bin.",
    )


def kde_plot(
    data: pd.DataFrame,
    column: str,
    *,
    title: str | None = None,
    caption: str | None = None,
) -> go.Figure:
    """Build a kernel-density estimate curve for continuous, non-constant numeric data.

    A KDE smooths observations into an estimated density; it can be misleading for small discrete
    ordinal scales, so known EduLens ordinal columns are rejected instead of presenting a smooth
    curve as if their codes were continuous measurements.
    """
    values = _numeric_values(data, column)
    if column in config.ORDINAL_COLUMNS:
        raise VisualizationInputError(
            f"KDE is not recommended for ordinal column '{column}'; use a count chart instead."
        )
    if values.nunique() < 2:
        raise VisualizationInputError(f"KDE requires variation; column '{column}' is constant.")
    try:
        density = stats.gaussian_kde(values.to_numpy(dtype=float))
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise VisualizationInputError(
            f"KDE could not be estimated for column '{column}'; check its variation and sample size."
        ) from exc
    lower = float(values.min())
    upper = float(values.max())
    padding = (upper - lower) * 0.08
    x_values = np.linspace(lower - padding, upper + padding, 256)
    figure = go.Figure(
        go.Scatter(
            x=x_values,
            y=density(x_values),
            mode="lines",
            line={"color": COLORBLIND_SAFE_PALETTE[1], "width": 3, "dash": "solid"},
            name="Kernel density estimate",
        )
    )
    return apply_plotly_theme(
        figure,
        title=title or f"Estimated distribution of {_display_name(column)}",
        x_label=_display_name(column),
        y_label="Estimated density",
        caption=caption
        or "The curve is a smoothed estimate of relative density, not a count or a probability at one exact value.",
    )


def box_plot(
    data: pd.DataFrame,
    value_col: str,
    group_col: str | None = None,
    *,
    title: str | None = None,
    caption: str | None = None,
) -> go.Figure:
    """Build a box-and-point chart, especially suited to discrete or ordinal groupings.

    For ordinal or categorical x groups, separate labeled boxes show medians and quartiles without
    implying equal numeric distances between codes. Individual observations use jittered markers
    to expose ties and group sizes.
    """
    values = _numeric_values(data, value_col)
    if group_col is not None:
        _require_columns(data, [group_col])
        valid = data.loc[data[value_col].notna() & data[group_col].notna(), [value_col, group_col]]
        if valid.empty:
            raise VisualizationInputError(
                "No complete observations are available for the box plot."
            )
        categories = valid[group_col].drop_duplicates().tolist()
        figure = go.Figure()
        for index, category in enumerate(categories):
            group_values = valid.loc[valid[group_col] == category, value_col]
            figure.add_trace(
                go.Box(
                    y=group_values,
                    name=str(category),
                    boxpoints="all",
                    jitter=0.32,
                    pointpos=0,
                    marker={
                        "color": COLORBLIND_SAFE_PALETTE[index % len(COLORBLIND_SAFE_PALETTE)],
                        "symbol": MARKER_SYMBOLS[index % len(MARKER_SYMBOLS)],
                        "size": 5,
                        "opacity": 0.65,
                    },
                    line={"color": COLORBLIND_SAFE_PALETTE[index % len(COLORBLIND_SAFE_PALETTE)]},
                )
            )
        return apply_plotly_theme(
            figure,
            title=title or f"{_display_name(value_col)} by {_display_name(group_col)}",
            x_label=_display_name(group_col),
            y_label=_display_name(value_col),
            caption=caption
            or "Each box shows the median and interquartile range; jittered symbols show individual observations.",
        )

    figure = go.Figure(
        go.Box(
            y=values,
            name=_display_name(value_col),
            boxpoints="all",
            jitter=0.3,
            marker={"color": COLORBLIND_SAFE_PALETTE[0], "symbol": "circle-open", "size": 5},
        )
    )
    return apply_plotly_theme(
        figure,
        title=title or f"Spread of {_display_name(value_col)}",
        x_label=_display_name(value_col),
        y_label=_display_name(value_col),
        caption=caption
        or "The box shows the median and interquartile range; jittered symbols show individual observations.",
    )


def violin_plot(
    data: pd.DataFrame,
    value_col: str,
    group_col: str | None = None,
    *,
    title: str | None = None,
    caption: str | None = None,
) -> go.Figure:
    """Build an interactive violin distribution with points and labeled discrete groups.

    Discrete/ordinal x values are displayed as categories with separate violin shapes rather than
    as a continuous scatter axis. The points make repeated observations visible in addition to the
    smoothed violin outline.
    """
    _numeric_values(data, value_col)
    if group_col is not None:
        _require_columns(data, [group_col])
        valid = data.loc[data[value_col].notna() & data[group_col].notna(), [value_col, group_col]]
        if valid.empty:
            raise VisualizationInputError(
                "No complete observations are available for the violin plot."
            )
        categories = valid[group_col].drop_duplicates().tolist()
        figure = go.Figure()
        for index, category in enumerate(categories):
            group_values = valid.loc[valid[group_col] == category, value_col]
            if group_values.nunique() < 2:
                raise VisualizationInputError(
                    f"Violin density cannot be estimated for constant group '{category}'."
                )
            color = COLORBLIND_SAFE_PALETTE[index % len(COLORBLIND_SAFE_PALETTE)]
            figure.add_trace(
                go.Violin(
                    y=group_values,
                    name=str(category),
                    points="all",
                    jitter=0.25,
                    pointpos=0,
                    line={"color": color},
                    fillcolor=color,
                    opacity=0.65,
                    marker={"symbol": MARKER_SYMBOLS[index % len(MARKER_SYMBOLS)]},
                    box_visible=True,
                    meanline_visible=True,
                )
            )
        x_label = _display_name(group_col)
    else:
        values = _numeric_values(data, value_col)
        if values.nunique() < 2:
            raise VisualizationInputError(
                f"Violin density cannot be estimated for constant column '{value_col}'."
            )
        figure = go.Figure(
            go.Violin(
                y=values,
                name=_display_name(value_col),
                points="all",
                line={"color": COLORBLIND_SAFE_PALETTE[0]},
                fillcolor=COLORBLIND_SAFE_PALETTE[0],
                opacity=0.65,
                marker={"symbol": "circle-open"},
                box_visible=True,
                meanline_visible=True,
            )
        )
        x_label = _display_name(value_col)
    return apply_plotly_theme(
        figure,
        title=title or f"Distribution of {_display_name(value_col)}",
        x_label=x_label,
        y_label=_display_name(value_col),
        caption=caption
        or "Violin width summarizes estimated density; the embedded box and points show quartiles and observations.",
    )


def bar_count_plot(
    data: pd.DataFrame,
    column: str,
    *,
    probability: bool = False,
    title: str | None = None,
    caption: str | None = None,
) -> go.Figure:
    """Build a category count or empirical-percent bar chart with a truthful zero baseline.

    With `probability=True`, each observed category's share is shown on a 0–100 percent axis;
    percentages are computed from non-missing observations.
    """
    _require_columns(data, [column])
    values = data[column].dropna()
    if values.empty:
        raise VisualizationInputError(f"Chart column '{column}' has no valid observations.")
    counts = values.value_counts(sort=False, dropna=True)
    if isinstance(values.dtype, pd.CategoricalDtype) and values.dtype.ordered:
        counts = counts.reindex(values.cat.categories, fill_value=0)
    elif column in config.ORDINAL_COLUMNS and pd.api.types.is_numeric_dtype(values.dtype):
        counts = counts.sort_index()
    plotted = counts / len(values) * 100 if probability else counts
    y_label = "Share of valid observations (%)" if probability else "Count"
    if probability:
        caption = caption or (
            "Each bar is the percentage of non-missing observations in that category; bars total 100%."
        )
    else:
        caption = caption or "Each bar counts non-missing observations in its labeled category."
    figure = go.Figure(
        go.Bar(
            x=[str(value) for value in plotted.index],
            y=plotted.to_numpy(),
            marker={
                "color": [
                    COLORBLIND_SAFE_PALETTE[index % len(COLORBLIND_SAFE_PALETTE)]
                    for index in range(len(plotted))
                ],
                "pattern": {"shape": ["/", "\\", "x", ".", "+", "-"] * (len(plotted) // 6 + 1)},
            },
            name="Category share" if probability else "Category count",
        )
    )
    figure.update_yaxes(range=[0, 100] if probability else [0, None], rangemode="tozero")
    return apply_plotly_theme(
        figure,
        title=title
        or ("Category proportions" if probability else f"Counts of {_display_name(column)}"),
        x_label=_display_name(column),
        y_label=y_label,
        caption=caption,
    )


def probability_bar_plot(
    probabilities: Mapping[str, float],
    *,
    title: str,
    caption: str,
) -> go.Figure:
    """Build a labeled probability bar chart with a fixed 0–100% vertical scale.

    Values must be a non-empty probability distribution. A fixed percentage scale makes
    differently predicted profiles directly comparable without visual autoscaling.
    """
    if not probabilities:
        raise VisualizationInputError("A probability chart requires at least one class.")
    labels = list(probabilities)
    values = np.asarray(list(probabilities.values()), dtype=float)
    if any(not label.strip() for label in labels):
        raise VisualizationInputError("Every probability category requires a non-empty label.")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise VisualizationInputError("Probability values must be finite and between 0 and 1.")
    if not np.isclose(values.sum(), 1.0, rtol=1e-9, atol=1e-9):
        raise VisualizationInputError("Class probabilities must sum to 1.")
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values * 100,
            marker={
                "color": COLORBLIND_SAFE_PALETTE[0],
                "pattern": {"shape": ["/", "x", "."] * (len(labels) // 3 + 1)},
            },
            name="Estimated probability",
        )
    )
    figure.update_yaxes(range=[0, 100], rangemode="tozero")
    return apply_plotly_theme(
        figure,
        title=title,
        x_label="Performance category",
        y_label="Estimated probability (%)",
        caption=caption,
    )


def scatter_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    trend_line: bool = False,
    group_col: str | None = None,
    title: str | None = None,
    caption: str | None = None,
) -> go.Figure:
    """Build a scatter plot or an ordinal-x grouped point/box display.

    Known EduLens ordinal and binary predictors, ordered categoricals, and non-numeric x columns
    use separate category positions with jittered points rather than overplotted numeric dots.
    Optional trend lines are limited to non-ordinal numeric x values and are labeled explicitly.
    """
    _require_columns(data, [x_col, y_col] + ([group_col] if group_col else []))
    _numeric_values(data, y_col)
    x_series = data[x_col]
    if pd.api.types.is_numeric_dtype(x_series.dtype) and not pd.api.types.is_bool_dtype(
        x_series.dtype
    ):
        x_numeric = x_series
        valid = data.loc[x_numeric.notna() & data[y_col].notna()].copy()
    else:
        valid = data.loc[data[x_col].notna() & data[y_col].notna()].copy()
        x_numeric = None
    valid = valid.loc[np.isfinite(pd.to_numeric(valid[y_col], errors="coerce"))]
    if valid.empty:
        raise VisualizationInputError(
            "No complete finite observations are available for the scatter plot."
        )

    ordinal_x = (
        x_col in config.ORDINAL_COLUMNS
        or x_col in config.BINARY_COLUMNS
        or isinstance(x_series.dtype, pd.CategoricalDtype)
        or x_numeric is None
    )
    figure = go.Figure()
    if ordinal_x:
        categories = (
            list(x_series.cat.categories)
            if isinstance(x_series.dtype, pd.CategoricalDtype) and x_series.dtype.ordered
            else valid[x_col].drop_duplicates().tolist()
        )
        for index, category in enumerate(categories):
            group = valid.loc[valid[x_col] == category]
            if group.empty:
                continue
            color = COLORBLIND_SAFE_PALETTE[index % len(COLORBLIND_SAFE_PALETTE)]
            jitter_seed = index + 1
            jitter = np.random.default_rng(jitter_seed).uniform(-0.12, 0.12, len(group))
            figure.add_trace(
                go.Scatter(
                    x=np.full(len(group), index, dtype=float) + jitter,
                    y=group[y_col],
                    mode="markers",
                    marker={
                        "color": color,
                        "symbol": MARKER_SYMBOLS[index % len(MARKER_SYMBOLS)],
                        "size": 8,
                        "opacity": 0.7,
                    },
                    name=str(category),
                    showlegend=False,
                    hovertemplate=f"{_display_name(x_col)}={category}<br>{_display_name(y_col)}=%{{y}}<extra></extra>",
                )
            )
        figure.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(categories))),
            ticktext=[str(category) for category in categories],
        )
        actual_caption = caption or (
            f"Each jittered point is one observation. {_display_name(x_col)} is shown as categories "
            "because it is ordinal or categorical; spacing does not imply measured numeric distance."
        )
    else:
        x_values = pd.to_numeric(valid[x_col], errors="coerce")
        y_clean = pd.to_numeric(valid[y_col], errors="coerce")
        valid = valid.loc[np.isfinite(x_values) & np.isfinite(y_clean)].copy()
        if valid.empty:
            raise VisualizationInputError(
                "No complete finite observations are available for the scatter plot."
            )
        x_values = pd.to_numeric(valid[x_col], errors="coerce")
        y_clean = pd.to_numeric(valid[y_col], errors="coerce")
        if group_col:
            categories = valid[group_col].drop_duplicates().tolist()
            for index, category in enumerate(categories):
                mask = valid[group_col].eq(category)
                figure.add_trace(
                    go.Scatter(
                        x=x_values[mask],
                        y=y_clean[mask],
                        mode="markers",
                        name=str(category),
                        marker={
                            "color": COLORBLIND_SAFE_PALETTE[index % len(COLORBLIND_SAFE_PALETTE)],
                            "symbol": MARKER_SYMBOLS[index % len(MARKER_SYMBOLS)],
                            "size": 8,
                            "opacity": 0.72,
                        },
                    )
                )
        else:
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_clean,
                    mode="markers",
                    name="Observations",
                    marker={
                        "color": COLORBLIND_SAFE_PALETTE[0],
                        "symbol": "circle-open",
                        "size": 8,
                        "opacity": 0.72,
                    },
                )
            )
        if trend_line:
            if len(valid) < 2 or x_values.nunique() < 2:
                raise VisualizationInputError(
                    "A trend line requires at least two distinct x values."
                )
            slope, intercept, *_ = stats.linregress(x_values, y_clean)
            ordered_x = np.linspace(float(x_values.min()), float(x_values.max()), 100)
            figure.add_trace(
                go.Scatter(
                    x=ordered_x,
                    y=slope * ordered_x + intercept,
                    mode="lines",
                    name="Linear trend (association)",
                    line={"color": COLORBLIND_SAFE_PALETTE[2], "width": 2, "dash": "dash"},
                )
            )
        actual_caption = caption or (
            "Each marker is an observation. "
            + (
                "The dashed line summarizes a linear association and is not a causal effect."
                if trend_line
                else "The plot displays observed values without implying causation."
            )
        )

    return apply_plotly_theme(
        figure,
        title=title or f"{_display_name(y_col)} by {_display_name(x_col)}",
        x_label=_display_name(x_col),
        y_label=_display_name(y_col),
        caption=actual_caption,
    )


def correlation_heatmap(
    correlations: pd.DataFrame,
    *,
    title: str = "Correlation matrix",
    caption: str = "Each cell shows the supplied pairwise correlation; the colorbar maps color to coefficient value.",
) -> go.Figure:
    """Build an interactive correlation heatmap with a labeled, fixed −1 to 1 color scale."""
    if correlations.empty:
        raise VisualizationInputError("Cannot create a correlation heatmap from an empty matrix.")
    if correlations.shape[0] != correlations.shape[1]:
        raise VisualizationInputError("A correlation heatmap requires a square matrix.")
    numeric = correlations.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise VisualizationInputError("The correlation matrix must contain only numeric values.")
    if not np.isfinite(numeric.to_numpy()).all():
        raise VisualizationInputError("The correlation matrix contains non-finite values.")
    if numeric.shape[0] != numeric.shape[1] or not np.allclose(
        numeric.to_numpy(), numeric.to_numpy().T, equal_nan=True
    ):
        raise VisualizationInputError("The correlation matrix must be symmetric.")
    figure = go.Figure(
        go.Heatmap(
            z=numeric.to_numpy(),
            x=[str(value) for value in numeric.columns],
            y=[str(value) for value in numeric.index],
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdBu",
            colorbar={"title": {"text": "Correlation coefficient"}},
            hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    return apply_plotly_theme(
        figure,
        title=title,
        x_label="Variable",
        y_label="Variable",
        caption=caption,
    )


def grouped_comparison(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
    *,
    statistic: Literal["mean", "median"] = "median",
    title: str | None = None,
    caption: str | None = None,
) -> go.Figure:
    """Compare a numeric outcome across labeled discrete groups with distributions and point symbols.

    This builder uses category axes even when the group codes are integers; for ordinal predictors,
    their code values represent ordered categories rather than assured equal intervals.
    """
    if statistic not in {"mean", "median"}:
        raise VisualizationInputError("statistic must be 'mean' or 'median'.")
    _numeric_values(data, value_col)
    _require_columns(data, [group_col, value_col])
    valid = data.loc[data[group_col].notna() & data[value_col].notna(), [group_col, value_col]]
    if valid.empty:
        raise VisualizationInputError(
            "No complete observations are available for grouped comparison."
        )
    categories = valid[group_col].drop_duplicates().tolist()
    figure = go.Figure()
    for index, category in enumerate(categories):
        group = valid.loc[valid[group_col] == category, value_col]
        color = COLORBLIND_SAFE_PALETTE[index % len(COLORBLIND_SAFE_PALETTE)]
        figure.add_trace(
            go.Box(
                y=group,
                name=str(category),
                boxpoints="all",
                jitter=0.32,
                pointpos=0,
                marker={
                    "color": color,
                    "symbol": MARKER_SYMBOLS[index % len(MARKER_SYMBOLS)],
                    "size": 5,
                    "opacity": 0.65,
                },
                line={"color": color},
                boxmean=statistic == "mean",
            )
        )
    return apply_plotly_theme(
        figure,
        title=title or f"{_display_name(value_col)} by {_display_name(group_col)}",
        x_label=_display_name(group_col),
        y_label=_display_name(value_col),
        caption=caption
        or f"Each category is a separate group; boxes show median and quartiles, points show observations, and the {'mean' if statistic == 'mean' else 'median'} is the selected comparison summary.",
    )


def qq_plot(
    values: pd.Series | Sequence[float],
    *,
    title: str = "Normal Q–Q plot",
    caption: str = "Points near the reference line are more consistent with a normal distribution; systematic departures indicate differences.",
) -> plt.Figure:
    """Build a static Matplotlib normal Q–Q plot for residual diagnostics and distribution checks."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise VisualizationInputError("A Q–Q plot requires a non-empty one-dimensional sample.")
    array = array[np.isfinite(array)]
    if array.size < 2:
        raise VisualizationInputError("A Q–Q plot requires at least two finite observations.")
    if np.unique(array).size < 2:
        raise VisualizationInputError("A Q–Q plot requires variation; all observations are equal.")

    with plt.style.context("dark_background"):
        figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        figure.patch.set_alpha(0)
        stats.probplot(array, dist="norm", plot=axis)
        apply_matplotlib_theme(
            figure,
            axis,
            title=title,
            x_label="Theoretical normal quantiles",
            y_label="Ordered sample values",
            caption=caption,
        )
        axis.get_lines()[0].set_markerfacecolor(COLORBLIND_SAFE_PALETTE[0])
        axis.get_lines()[0].set_markeredgecolor(COLORBLIND_SAFE_PALETTE[0])
        axis.get_lines()[1].set_color(COLORBLIND_SAFE_PALETTE[2])
    return figure


def studytime_vs_g3(data: pd.DataFrame) -> go.Figure:
    """Compare final-grade distributions across the ordinal weekly study-time categories."""
    return grouped_comparison(
        data,
        "studytime",
        "G3",
        title="Final grade by weekly study-time category",
        caption="Study time is ordinal; separate boxes compare G3 distributions within each labeled category.",
    )


def absences_vs_g3(data: pd.DataFrame) -> go.Figure:
    """Show the observed association pattern between absences and final grade."""
    return scatter_plot(
        data,
        "absences",
        "G3",
        trend_line=True,
        title="Final grade and school absences",
        caption="Markers show observed students; the dashed linear trend summarizes association only.",
    )


def failures_vs_g3(data: pd.DataFrame) -> go.Figure:
    """Compare final-grade distributions across the recorded previous-failure categories."""
    return grouped_comparison(
        data,
        "failures",
        "G3",
        title="Final grade by previous class-failure category",
        caption="Failure count is treated as an ordered category; boxes compare observed G3 distributions.",
    )


def g1_vs_g3(data: pd.DataFrame) -> go.Figure:
    """Show the association between first-period and final grades."""
    return scatter_plot(
        data,
        "G1",
        "G3",
        trend_line=True,
        title="Final grade and first-period grade (G1)",
        caption="Each marker is a student; the dashed trend summarizes a linear association, not a causal effect.",
    )


def g2_vs_g3(data: pd.DataFrame) -> go.Figure:
    """Show the association between second-period and final grades."""
    return scatter_plot(
        data,
        "G2",
        "G3",
        trend_line=True,
        title="Final grade and second-period grade (G2)",
        caption="Each marker is a student; the dashed trend summarizes a linear association, not a causal effect.",
    )


def famsup_vs_g3(data: pd.DataFrame) -> go.Figure:
    """Compare final-grade distributions by reported family educational support."""
    return grouped_comparison(
        data,
        "famsup",
        "G3",
        title="Final grade by reported family educational support",
        caption="Separate labeled groups compare the observed G3 distributions for each support response.",
    )


def parental_education_vs_g3(data: pd.DataFrame) -> go.Figure:
    """Compare G3 against mother's and father's ordinal education categories in two panels."""
    _require_columns(data, ["Medu", "Fedu", "G3"])
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Mother's education (Medu)", "Father's education (Fedu)"),
        shared_yaxes=True,
        horizontal_spacing=0.1,
    )
    for column_index, column in enumerate(("Medu", "Fedu"), start=1):
        subset = data.loc[data[column].notna() & data["G3"].notna(), [column, "G3"]]
        if subset.empty:
            raise VisualizationInputError(f"No complete observations are available for '{column}'.")
        for category_index, category in enumerate(sorted(subset[column].unique())):
            grades = subset.loc[subset[column] == category, "G3"]
            color = COLORBLIND_SAFE_PALETTE[category_index % len(COLORBLIND_SAFE_PALETTE)]
            figure.add_trace(
                go.Box(
                    y=grades,
                    name=str(category),
                    legendgroup=f"{column}-{category}",
                    showlegend=False,
                    boxpoints="all",
                    jitter=0.3,
                    marker={
                        "color": color,
                        "symbol": MARKER_SYMBOLS[category_index % len(MARKER_SYMBOLS)],
                        "size": 4,
                        "opacity": 0.6,
                    },
                    line={"color": color},
                ),
                row=1,
                col=column_index,
            )
    apply_plotly_theme(
        figure,
        title="Final grade by parental education",
        x_label="Ordinal education category (0–4)",
        y_label="Final grade (G3)",
        caption="Each panel compares G3 distributions across one parent's ordered education categories; category codes are not assumed equally spaced.",
    )
    figure.update_xaxes(title_text="Ordinal education category (0–4)")
    figure.update_yaxes(title_text="Final grade (G3)")
    return figure


def health_vs_g3(data: pd.DataFrame) -> go.Figure:
    """Compare final-grade distributions across the ordinal self-reported health scale."""
    return grouped_comparison(
        data,
        "health",
        "G3",
        title="Final grade by self-reported health category",
        caption="Health is ordinal; separate boxes compare observed G3 distributions by response category.",
    )


def freetime_vs_g3(data: pd.DataFrame) -> go.Figure:
    """Compare final-grade distributions across the ordinal after-school free-time scale."""
    return grouped_comparison(
        data,
        "freetime",
        "G3",
        title="Final grade by after-school free-time category",
        caption="Free time is ordinal; separate boxes compare observed G3 distributions by response category.",
    )
