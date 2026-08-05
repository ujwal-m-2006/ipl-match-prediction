"""Helpers shared by the dashboard pages."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

from ..data import has_data


def require_data() -> bool:
    """Show a friendly prompt and return ``False`` when the warehouse is empty."""
    if has_data():
        return True

    st.title("No data loaded yet")
    st.markdown(
        """
        The database is empty. Load IPL data first:

        ```bash
        python scripts/ingest.py
        ```

        Or open the **Admin** page and press *Refresh data*.
        """
    )
    return False


def require_model(service, task: str, label: str) -> bool:
    """Show a prompt and return ``False`` when a model has not been trained."""
    if service is not None and service.has_model(task):
        return True

    st.info(
        f"The **{label}** model has not been trained yet. Run "
        "`python scripts/train_models.py`, or use the **Admin** page."
    )
    return False


def page_header(title: str, subtitle: str | None = None) -> None:
    """Consistent page title block."""
    st.title(title)
    if subtitle:
        st.markdown(f'<p class="caption-note">{subtitle}</p>', unsafe_allow_html=True)
    st.write("")


def season_filter(
    seasons: Iterable[int], *, key: str, label: str = "Season", allow_all: bool = True
) -> int | None:
    """Season selectbox. Returns ``None`` when "All seasons" is picked."""
    options: list = list(seasons)
    if allow_all:
        options = ["All seasons"] + options
    if not options:
        return None
    choice = st.selectbox(label, options, key=key)
    return None if choice == "All seasons" else int(choice)


def show_table(
    frame: pd.DataFrame,
    *,
    height: int | None = None,
    hide_index: bool = True,
    column_config: dict | None = None,
    caption: str | None = None,
) -> None:
    """Render a DataFrame, or a friendly placeholder when it is empty.

    Every chart on a page is accompanied by its underlying table somewhere, so
    a reader who cannot distinguish two colours can still read the numbers.
    """
    if frame is None or frame.empty:
        st.caption("No rows to display.")
        return

    # Streamlit rejects an explicit height=None, so the argument is only passed
    # when the caller actually asked for a fixed height.
    kwargs: dict = {
        "width": "stretch",
        "hide_index": hide_index,
        "column_config": column_config or {},
    }
    if height is not None:
        kwargs["height"] = height

    st.dataframe(frame, **kwargs)
    if caption:
        st.caption(caption)


def metric_row(metrics: list[tuple[str, object, str | None]]) -> None:
    """Render a row of stat tiles: ``(label, value, delta)``."""
    columns = st.columns(len(metrics))
    for column, (label, value, delta) in zip(columns, metrics):
        with column:
            st.metric(label, value, delta)


def stat_table(rows: list[tuple[str, object]]) -> pd.DataFrame:
    """Build a two-column metric/value table with values rendered as text.

    A column mixing ints, floats and NaN lands in pandas as ``object`` dtype,
    which Arrow cannot serialise -- Streamlit recovers, but it logs a noisy
    traceback to the console. Formatting to strings up front avoids that and
    also renders missing values as an em dash instead of ``NaN``.
    """
    formatted = []
    for label, value in rows:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            text = "—"
        elif isinstance(value, float):
            # Whole-valued floats (counts read back from SQL) lose the ".0".
            text = f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
        elif isinstance(value, (int, np.integer)):
            text = f"{value:,}"
        else:
            text = str(value)
        formatted.append({"Metric": label, "Value": text})
    return pd.DataFrame(formatted)


def format_percent(value: float | None, decimals: int = 1) -> str:
    """Format a 0-100 percentage, tolerating ``None``/``NaN``."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{decimals}f}%"


def format_number(value: float | None, decimals: int = 0) -> str:
    """Format a number, tolerating ``None``/``NaN``."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}"
