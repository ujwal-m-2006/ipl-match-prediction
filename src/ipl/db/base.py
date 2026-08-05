"""Engine and session management.

A single lazily-created :class:`~sqlalchemy.engine.Engine` is shared by the
whole process. The same code path serves SQLite, PostgreSQL and MySQL; the
only backend-specific handling is the SQLite pragma tuning below.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ..config import get_settings
from ..logging_utils import get_logger

logger = get_logger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


class Base(DeclarativeBase):
    """Declarative base for every ORM model in the project."""


def _configure_sqlite(engine: Engine) -> None:
    """Apply pragmas that make SQLite behave sanely for this workload.

    ``WAL`` lets the Streamlit dashboard read while an ingestion run writes,
    and ``foreign_keys=ON`` is off by default in SQLite -- without it the
    relationships declared on our models would not actually be enforced.
    """

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, _connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


def get_engine(*, force_new: bool = False) -> Engine:
    """Return the shared engine, creating it on first call."""
    global _engine, _SessionFactory
    if _engine is not None and not force_new:
        return _engine

    settings = get_settings()
    kwargs: dict = {"echo": settings.db_echo, "future": True}

    if settings.is_sqlite:
        # check_same_thread=False is required because Streamlit serves each
        # session on a different thread while sharing our module-level engine.
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    else:
        # Server-backed databases benefit from a real pool with liveness checks.
        kwargs.update(pool_size=5, max_overflow=10, pool_pre_ping=True, pool_recycle=1800)

    engine = create_engine(settings.database_url, **kwargs)
    if settings.is_sqlite:
        _configure_sqlite(engine)

    _engine = engine
    _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    logger.debug("Database engine created for dialect %s", settings.dialect)
    return engine


def get_session() -> Session:
    """Return a new ORM session. Caller is responsible for closing it."""
    if _SessionFactory is None:
        get_engine()
    assert _SessionFactory is not None  # narrowed by get_engine()
    return _SessionFactory()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commits on success, rolls back on any exception."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(*, drop_existing: bool = False) -> None:
    """Create every table declared on :class:`Base`.

    Args:
        drop_existing: Drop all project tables first. Destructive -- only used
            by ``scripts/init_db.py --drop`` and the test fixtures.
    """
    # Import for the side effect of registering models on Base.metadata.
    from . import models  # noqa: F401

    engine = get_engine()
    if drop_existing:
        logger.warning("Dropping all existing IPL tables")
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    logger.info("Database schema ready (%d tables)", len(Base.metadata.tables))
