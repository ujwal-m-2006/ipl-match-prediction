"""Logging setup shared by every entry point.

One call to :func:`setup_logging` at process start configures a console handler
plus a rotating file handler under ``logs/``. Library modules should only ever
call :func:`get_logger` and never touch handlers themselves.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from .config import LOGS_DIR, get_settings

_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | None = None, *, log_file: str = "ipl.log") -> None:
    """Configure root logging exactly once per process.

    Args:
        level: Override the configured log level (e.g. ``"DEBUG"``).
        log_file: File name inside ``logs/`` for the rotating handler.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    resolved = (level or settings.log_level).upper()

    root = logging.getLogger()
    root.setLevel(resolved)
    # Drop pre-existing handlers so repeated Streamlit reruns don't duplicate lines.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOGS_DIR / log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # A read-only filesystem (some PaaS sandboxes) must not kill the app.
        root.warning("Could not open log file; continuing with console logging only.")

    # These libraries are chatty at INFO and add nothing for our purposes.
    for noisy in ("urllib3", "matplotlib", "PIL", "asyncio", "watchdog"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring logging on first use."""
    setup_logging()
    return logging.getLogger(name)
