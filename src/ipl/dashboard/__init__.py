"""Streamlit dashboard package."""

__all__ = ["run"]


def run() -> None:
    """Launch the dashboard. Imported lazily so Streamlit is optional."""
    from .app import main

    main()
