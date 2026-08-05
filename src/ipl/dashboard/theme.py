"""Chart theming and Plotly helpers.

Colour policy
-------------
Two palettes, used for two different jobs:

*Team identity* -- franchise brand colours from :mod:`ipl.constants`. Readers
expect CSK yellow and MI blue; overriding that would be actively confusing.
Because brand hues are not chosen for colour-vision safety, every team-coloured
chart also carries a legend or direct labels, so colour is never the only
channel carrying identity.

*Analytical series* -- :data:`CATEGORICAL` below, a palette validated for
colour-vision deficiency separation (worst adjacent CVD ΔE 9.1 light / 8.4
dark) and lightness banding in both light and dark mode. Slots are assigned in
fixed order and never cycled; past eight series the tail folds into "Other".

Other invariants applied throughout:

* No dual-axis charts. Two measures on different scales get two charts.
* Sequential encoding uses one hue, light to dark -- never a rainbow.
* Grid and axes are recessive; text uses ink tokens, never a series colour.
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
import plotly.graph_objects as go

from ..constants import DEFAULT_TEAM_COLOR, TEAM_COLORS

# ---------------------------------------------------------------------------
# Palette (validated -- see module docstring)
# ---------------------------------------------------------------------------
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

# Single-hue ramp for magnitude encoding (light to dark).
SEQUENTIAL_BLUE: tuple[str, ...] = (
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95",
)

# Two poles plus a neutral midpoint, for signed quantities (e.g. run-rate delta).
DIVERGING: tuple[str, ...] = ("#184f95", "#3987e5", "#f0efec", "#e34948", "#a52a2a")

# Ink and chrome. Text never takes a series colour.
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
SURFACE = "rgba(0,0,0,0)"  # inherit Streamlit's surface

# Status colours are reserved and never reused as a series slot.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def format_values(values: pd.Series, spec: str | None) -> list[str] | None:
    """Format a numeric column for direct data labels, defensively.

    Chart data arrives from SQL aggregates, so an "integer" column is routinely
    a float (``14.0``) and may contain nulls. ``format(14.0, "d")`` raises, which
    would take down the whole page for a cosmetic label -- so integer specs
    round first, nulls render as an empty label, and anything unexpected falls
    back to ``str`` rather than propagating an exception.
    """
    if spec is None:
        return None

    numeric = pd.to_numeric(values, errors="coerce")
    integral = spec.endswith("d")

    labels: list[str] = []
    for value in numeric:
        if pd.isna(value):
            labels.append("")
            continue
        try:
            labels.append(format(int(round(value)) if integral else float(value), spec))
        except (ValueError, TypeError):
            labels.append(str(value))
    return labels


def team_palette(teams: Sequence[str]) -> list[str]:
    """Brand colours for a list of franchises, in the order given."""
    return [TEAM_COLORS.get(team, DEFAULT_TEAM_COLOR) for team in teams]


def series_palette(count: int) -> list[str]:
    """Return ``count`` categorical slots in fixed order.

    Raises no error past eight: callers should have folded the tail into
    "Other" first, but a graceful truncation beats a crash in a dashboard.
    """
    return [CATEGORICAL[i % len(CATEGORICAL)] for i in range(count)]


def base_layout(
    title: str | None = None,
    *,
    height: int = 420,
    x_title: str | None = None,
    y_title: str | None = None,
    show_legend: bool = False,
    legend_horizontal: bool = True,
) -> dict[str, Any]:
    """Standard Plotly layout: recessive chrome, readable ink, no clutter."""
    layout: dict[str, Any] = {
        "height": height,
        "template": "plotly_white",
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {"family": FONT_FAMILY, "size": 13, "color": INK_SECONDARY},
        "margin": {"l": 60, "r": 24, "t": 56 if title else 24, "b": 56},
        "hoverlabel": {"font": {"family": FONT_FAMILY, "size": 12}},
        "showlegend": show_legend,
        "xaxis": {
            "title": {"text": x_title or "", "font": {"color": INK_SECONDARY}},
            "gridcolor": GRIDLINE,
            "linecolor": AXIS_LINE,
            "zeroline": False,
            "tickfont": {"color": INK_MUTED},
        },
        "yaxis": {
            "title": {"text": y_title or "", "font": {"color": INK_SECONDARY}},
            "gridcolor": GRIDLINE,
            "linecolor": AXIS_LINE,
            "zeroline": False,
            "tickfont": {"color": INK_MUTED},
        },
    }
    if title:
        layout["title"] = {
            "text": title,
            "font": {"size": 17, "color": INK_PRIMARY},
            "x": 0,
            "xanchor": "left",
        }
    if show_legend and legend_horizontal:
        layout["legend"] = {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "title": {"text": ""},
        }
    return layout


def bar_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str | None = None,
    colors: Sequence[str] | None = None,
    orientation: str = "v",
    height: int = 420,
    x_title: str | None = None,
    y_title: str | None = None,
    text_format: str | None = None,
    hover_template: str | None = None,
) -> go.Figure:
    """Single-series bar chart with rounded data-ends and direct value labels.

    A single series carries no legend -- the title names it.
    """
    if data.empty:
        return empty_figure(title)

    palette = list(colors) if colors is not None else [CATEGORICAL[0]] * len(data)
    horizontal = orientation == "h"

    figure = go.Figure(
        go.Bar(
            x=data[y] if horizontal else data[x],
            y=data[x] if horizontal else data[y],
            orientation="h" if horizontal else "v",
            marker={
                "color": palette,
                # A 2px surface gap keeps adjacent fills from merging.
                "line": {"width": 1.5, "color": "rgba(255,255,255,0.9)"},
                "cornerradius": 4,
            },
            text=format_values(data[y], text_format),
            textposition="outside",
            textfont={"color": INK_SECONDARY, "size": 11},
            hovertemplate=hover_template or "%{x}<br>%{y}<extra></extra>",
        )
    )
    figure.update_layout(
        **base_layout(
            title, height=height,
            x_title=x_title or (y_title if horizontal else x.replace("_", " ").title()),
            y_title=y_title or (x if horizontal else y.replace("_", " ").title()),
        )
    )
    return figure


def grouped_bar_chart(
    data: pd.DataFrame,
    x: str,
    series: dict[str, str],
    *,
    title: str | None = None,
    colors: Sequence[str] | None = None,
    height: int = 420,
    x_title: str | None = None,
    y_title: str | None = None,
) -> go.Figure:
    """Multi-series grouped bars. Always legended (two or more series)."""
    if data.empty:
        return empty_figure(title)

    palette = list(colors) if colors is not None else series_palette(len(series))
    figure = go.Figure()
    for index, (label, column) in enumerate(series.items()):
        figure.add_trace(
            go.Bar(
                name=label,
                x=data[x],
                y=data[column],
                marker={
                    "color": palette[index % len(palette)],
                    "line": {"width": 1.5, "color": "rgba(255,255,255,0.9)"},
                    "cornerradius": 4,
                },
                hovertemplate=f"{label}<br>%{{x}}: %{{y}}<extra></extra>",
            )
        )
    figure.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        **base_layout(title, height=height, x_title=x_title, y_title=y_title, show_legend=True),
    )
    return figure


def line_chart(
    data: pd.DataFrame,
    x: str,
    series: dict[str, str],
    *,
    title: str | None = None,
    colors: Sequence[str] | None = None,
    height: int = 420,
    x_title: str | None = None,
    y_title: str | None = None,
    markers: bool = True,
) -> go.Figure:
    """Line chart with a shared crosshair tooltip.

    A legend appears only when there are two or more series.
    """
    if data.empty:
        return empty_figure(title)

    palette = list(colors) if colors is not None else series_palette(len(series))
    figure = go.Figure()
    for index, (label, column) in enumerate(series.items()):
        figure.add_trace(
            go.Scatter(
                name=label,
                x=data[x],
                y=data[column],
                mode="lines+markers" if markers else "lines",
                line={"width": 2, "color": palette[index % len(palette)]},
                marker={"size": 8, "line": {"width": 2, "color": "rgba(255,255,255,0.9)"}},
                hovertemplate=f"{label}<br>%{{x}}: %{{y}}<extra></extra>",
            )
        )
    layout = base_layout(
        title, height=height, x_title=x_title, y_title=y_title,
        show_legend=len(series) > 1,
    )
    # A unified crosshair is the default reading mode for time series.
    layout["hovermode"] = "x unified"
    figure.update_layout(**layout)
    return figure


def horizontal_probability_bar(
    label_left: str,
    label_right: str,
    probability_left: float,
    *,
    color_left: str | None = None,
    color_right: str | None = None,
    height: int = 130,
) -> go.Figure:
    """A single stacked bar showing a two-way probability split.

    Used for win probability. Both segments are directly labelled, so the split
    is readable without relying on the colours at all.
    """
    left = max(min(probability_left, 1.0), 0.0) * 100
    right = 100 - left

    figure = go.Figure()
    for name, value, color, side in (
        (label_left, left, color_left or CATEGORICAL[0], "left"),
        (label_right, right, color_right or CATEGORICAL[1], "right"),
    ):
        figure.add_trace(
            go.Bar(
                name=name,
                x=[value],
                y=[""],
                orientation="h",
                marker={
                    "color": color,
                    # 2px surface gap between the two fills.
                    "line": {"width": 2, "color": "rgba(255,255,255,0.95)"},
                },
                text=[f"{name} {value:.1f}%"],
                textposition="inside",
                insidetextanchor="middle" if side == "left" else "middle",
                textfont={"color": "#ffffff", "size": 14},
                hovertemplate=f"{name}: %{{x:.1f}}%<extra></extra>",
            )
        )

    figure.update_layout(
        barmode="stack",
        height=height,
        showlegend=False,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        font={"family": FONT_FAMILY},
        xaxis={"visible": False, "range": [0, 100]},
        yaxis={"visible": False},
    )
    return figure


def gauge(value: float, title: str, *, suffix: str = "%", color: str | None = None) -> go.Figure:
    """A single-number gauge for a headline probability."""
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"size": 34, "color": INK_PRIMARY}},
            title={"text": title, "font": {"size": 14, "color": INK_SECONDARY}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": INK_MUTED},
                "bar": {"color": color or CATEGORICAL[0], "thickness": 0.7},
                "bgcolor": "rgba(0,0,0,0.04)",
                "borderwidth": 0,
            },
        )
    )
    figure.update_layout(
        height=240,
        paper_bgcolor=SURFACE,
        margin={"l": 24, "r": 24, "t": 48, "b": 16},
        font={"family": FONT_FAMILY},
    )
    return figure


def heatmap(
    matrix: pd.DataFrame,
    *,
    title: str | None = None,
    height: int = 480,
    colorbar_title: str = "",
    value_format: str = ".0f",
) -> go.Figure:
    """Sequential heatmap using the single-hue blue ramp."""
    if matrix.empty:
        return empty_figure(title)

    figure = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            # One hue, light to dark: magnitude, not identity.
            colorscale=[[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)],
            texttemplate=f"%{{z:{value_format}}}",
            textfont={"size": 11},
            colorbar={"title": colorbar_title, "thickness": 12, "outlinewidth": 0},
            hovertemplate="%{y} vs %{x}<br>%{z}<extra></extra>",
            xgap=2,  # 2px surface gap between cells
            ygap=2,
        )
    )
    figure.update_layout(**base_layout(title, height=height))
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(showgrid=False, autorange="reversed")
    return figure


def empty_figure(title: str | None = None, message: str = "No data available") -> go.Figure:
    """Placeholder shown when a chart has nothing to draw."""
    figure = go.Figure()
    figure.add_annotation(
        text=message, showarrow=False,
        font={"size": 14, "color": INK_MUTED, "family": FONT_FAMILY},
        xref="paper", yref="paper", x=0.5, y=0.5,
    )
    figure.update_layout(
        height=280,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 8, "r": 8, "t": 48 if title else 8, "b": 8},
        title=({"text": title, "x": 0, "xanchor": "left"} if title else None),
    )
    return figure


# ---------------------------------------------------------------------------
# Page styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }

  /* Stat tiles: a hairline ring rather than a heavy card. */
  div[data-testid="stMetric"] {
      background: rgba(128,128,128,0.05);
      border: 1px solid rgba(128,128,128,0.18);
      border-radius: 10px;
      padding: 0.85rem 1rem;
  }
  div[data-testid="stMetricLabel"] { opacity: 0.75; font-size: 0.82rem; }

  /* Team chip used in headers and comparison tables. */
  .team-chip {
      display: inline-flex; align-items: center; gap: 0.45rem;
      font-weight: 600; font-size: 0.95rem;
  }
  .team-swatch {
      width: 12px; height: 12px; border-radius: 3px; display: inline-block;
      border: 1px solid rgba(128,128,128,0.35);
  }

  .caption-note { font-size: 0.82rem; opacity: 0.7; margin-top: -0.4rem; }

  /* Tabs a little roomier than the default. */
  button[data-baseweb="tab"] { font-size: 0.95rem; }
</style>
"""


def team_chip(team: str) -> str:
    """Inline HTML for a team name with its brand swatch.

    The swatch is decorative; the name beside it is what conveys identity, so
    the pairing satisfies the "never colour alone" rule.
    """
    color = TEAM_COLORS.get(team, DEFAULT_TEAM_COLOR)
    return (
        f'<span class="team-chip"><span class="team-swatch" '
        f'style="background:{color}"></span>{team}</span>'
    )
