"""Central configuration.

Every tunable in the project funnels through the :class:`Settings` object so
that behaviour can be changed with environment variables (or a ``.env`` file)
without touching code. Import the shared instance via :func:`get_settings`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
# config.py lives at <root>/src/ipl/config.py, so the project root is 3 levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"           # Untouched feed payloads (HTTP cache)
INTERIM_DIR = DATA_DIR / "interim"   # Normalised-but-not-yet-loaded frames
PROCESSED_DIR = DATA_DIR / "processed"  # Model-ready feature tables
EXTERNAL_DIR = DATA_DIR / "external"    # Third-party archives (Cricsheet zip)

MODELS_DIR = PROJECT_ROOT / "models" / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

_ALL_DIRS = (
    DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, EXTERNAL_DIR,
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR, LOGS_DIR,
)

# Load .env once, at import time, before Settings reads os.environ.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_bool(key: str, default: bool) -> bool:
    """Read a boolean env var, accepting the usual truthy spellings."""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, "").strip() or default)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, "").strip() or default)
    except ValueError:
        return default


class Settings:
    """Runtime configuration, resolved from environment variables.

    Deliberately a plain class rather than a pydantic model: it keeps the
    import graph light (the Streamlit app imports this on every rerun) and
    every field here is a simple scalar with an obvious default.
    """

    def __init__(self) -> None:
        # --- Database ---
        # Default to a file-based SQLite DB so a fresh clone runs with no setup.
        # Relative SQLite paths are resolved against PROJECT_ROOT, not the CWD,
        # so the dashboard and CLI scripts always agree on which file to open.
        raw_url = os.getenv("IPL_DATABASE_URL", "sqlite:///data/ipl.db").strip()
        self.database_url: str = self._resolve_sqlite_path(raw_url)
        self.db_echo: bool = _env_bool("IPL_DB_ECHO", False)

        # --- Data collection ---
        self.request_delay: float = _env_float("IPL_REQUEST_DELAY", 0.6)
        self.request_timeout: int = _env_int("IPL_REQUEST_TIMEOUT", 45)
        self.max_retries: int = _env_int("IPL_MAX_RETRIES", 4)
        self.use_http_cache: bool = _env_bool("IPL_USE_HTTP_CACHE", True)
        self.http_cache_ttl_hours: int = _env_int("IPL_HTTP_CACHE_TTL_HOURS", 12)
        self.enable_cricsheet: bool = _env_bool("IPL_ENABLE_CRICSHEET", True)
        self.ingest_deliveries: bool = _env_bool("IPL_INGEST_DELIVERIES", True)

        # --- Modelling ---
        # Hold out the three most recent seasons (~215 matches). Two seasons
        # leaves only ~144 test matches, where the standard error on ROC-AUC is
        # ~0.05 -- too wide to distinguish the models from each other.
        self.test_season_from: int = _env_int("IPL_TEST_SEASON_FROM", 2024)
        self.random_state: int = _env_int("IPL_RANDOM_STATE", 42)

        # --- Serving ---
        self.api_host: str = os.getenv("IPL_API_HOST", "0.0.0.0")
        self.api_port: int = _env_int("IPL_API_PORT", 8000)
        self.admin_password: str = os.getenv("IPL_ADMIN_PASSWORD", "change-me")

        # --- Logging ---
        self.log_level: str = os.getenv("IPL_LOG_LEVEL", "INFO").upper()

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _resolve_sqlite_path(url: str) -> str:
        """Rewrite a relative ``sqlite:///`` URL to an absolute one.

        Without this, running ``streamlit run`` from a different directory than
        ``python scripts/ingest.py`` would silently create two separate
        databases -- a classic and very confusing failure mode.
        """
        prefix = "sqlite:///"
        if not url.startswith(prefix):
            return url
        path_part = url[len(prefix):]
        if not path_part or path_part.startswith(":memory:"):
            return url
        path = Path(path_part)
        if path.is_absolute():
            return url
        return prefix + (PROJECT_ROOT / path).as_posix()

    @property
    def dialect(self) -> str:
        """``sqlite`` | ``postgresql`` | ``mysql`` - the backend in use."""
        return self.database_url.split("://", 1)[0].split("+", 1)[0]

    @property
    def is_sqlite(self) -> bool:
        return self.dialect == "sqlite"

    def ensure_directories(self) -> None:
        """Create every directory the pipeline writes to."""
        for directory in _ALL_DIRS:
            directory.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Settings(dialect={self.dialect!r}, "
            f"test_season_from={self.test_season_from}, "
            f"deliveries={self.ingest_deliveries})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    settings = Settings()
    settings.ensure_directories()
    return settings
