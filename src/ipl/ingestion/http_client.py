"""A polite, cached, retrying HTTP client.

The IPL feed host is a free public endpoint that serves the official website.
Hammering it is both rude and counter-productive (it starts resetting
connections after a burst -- observed at ~200 concurrent requests). This client
therefore enforces:

* a minimum delay between consecutive requests (serialised, single-threaded),
* exponential backoff with jitter on transient failures,
* an on-disk cache so a re-run of the pipeline downloads nothing at all,
* a descriptive User-Agent identifying the project.

The cache is keyed on the URL and lives under ``data/raw/``, which is
git-ignored.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import requests

from ..config import RAW_DIR, get_settings
from ..logging_utils import get_logger

logger = get_logger(__name__)

USER_AGENT = (
    "ipl-analytics/1.0 (educational portfolio project; "
    "+https://github.com/your-username/ipl-analytics)"
)

# The feeds are JSONP: `MatchSchedule({...})`, `onScoring({...})`, ...
_JSONP_RE = re.compile(r"^\s*[A-Za-z_$][\w$.]*\s*\(", re.MULTILINE)

# Status codes worth retrying: transient server/proxy/rate-limit conditions.
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class FeedNotFound(Exception):
    """Raised when a feed URL returns 404 -- an expected, non-fatal outcome.

    Many optional feeds (e.g. a squad list for an abandoned match) simply do
    not exist; callers treat this as "no data" rather than an error.
    """


class HttpClient:
    """Serialised, cached HTTP GET client for the public cricket feeds."""

    def __init__(
        self,
        *,
        delay: float | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        use_cache: bool | None = None,
        cache_dir: Path | None = None,
        cache_ttl_hours: int | None = None,
    ) -> None:
        settings = get_settings()
        self.delay = settings.request_delay if delay is None else delay
        self.timeout = settings.request_timeout if timeout is None else timeout
        self.max_retries = settings.max_retries if max_retries is None else max_retries
        self.use_cache = settings.use_http_cache if use_cache is None else use_cache
        self.cache_dir = cache_dir or (RAW_DIR / "feeds")
        self.cache_ttl_seconds = (
            settings.http_cache_ttl_hours if cache_ttl_hours is None else cache_ttl_hours
        ) * 3600

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self._last_request_at = 0.0
        self.stats = {"hits": 0, "misses": 0, "errors": 0, "not_found": 0}

    # -- cache --------------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        """Deterministic cache filename: readable tail + hash for uniqueness."""
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        tail = re.sub(r"[^A-Za-z0-9._-]", "_", url.rsplit("/", 1)[-1])[:60]
        return self.cache_dir / f"{tail}.{digest}.cache"

    def _read_cache(self, path: Path, *, allow_stale: bool = False) -> str | None:
        if not self.use_cache or not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self.cache_ttl_seconds and not allow_stale:
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    def _write_cache(self, path: Path, payload: str) -> None:
        if not self.use_cache:
            return
        try:
            path.write_text(payload, encoding="utf-8")
        except OSError as exc:  # pragma: no cover - disk full / permissions
            logger.debug("Could not write cache %s: %s", path, exc)

    # -- rate limiting ------------------------------------------------------
    def _throttle(self) -> None:
        """Sleep just long enough to honour the configured inter-request delay."""
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_at = time.monotonic()

    # -- fetching -----------------------------------------------------------
    def get_text(self, url: str, *, force_refresh: bool = False) -> str:
        """GET a URL and return the decoded body, using the disk cache.

        Raises:
            FeedNotFound: the server returned 404.
            requests.RequestException: every retry was exhausted.
        """
        path = self._cache_path(url)
        if not force_refresh:
            cached = self._read_cache(path)
            if cached is not None:
                self.stats["hits"] += 1
                logger.debug("cache hit  %s", url)
                return cached

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                response = self.session.get(url, timeout=self.timeout)

                if response.status_code == 404:
                    self.stats["not_found"] += 1
                    raise FeedNotFound(url)

                if response.status_code in _RETRY_STATUS:
                    raise requests.HTTPError(
                        f"HTTP {response.status_code}", response=response
                    )

                response.raise_for_status()
                # utf-8-sig strips the BOM these feeds are served with.
                payload = response.content.decode("utf-8-sig", errors="replace")
                self.stats["misses"] += 1
                self._write_cache(path, payload)
                logger.debug("fetched    %s (%d bytes)", url, len(payload))
                return payload

            except FeedNotFound:
                raise
            except (requests.RequestException, OSError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                # Exponential backoff with jitter, so parallel workers de-sync.
                backoff = min(2 ** attempt, 30) + random.uniform(0, 0.75)
                logger.warning(
                    "Request failed (%s), attempt %d/%d - retrying in %.1fs: %s",
                    type(exc).__name__, attempt, self.max_retries, backoff, url,
                )
                time.sleep(backoff)

        # Every retry failed. A stale cache entry beats no data at all.
        stale = self._read_cache(path, allow_stale=True)
        if stale is not None:
            logger.warning("Serving STALE cache after failures: %s", url)
            self.stats["hits"] += 1
            return stale

        self.stats["errors"] += 1
        raise requests.RequestException(f"Failed after {self.max_retries} attempts: {url}") from last_error

    def get_jsonp(self, url: str, *, force_refresh: bool = False) -> dict[str, Any]:
        """GET a JSONP feed and return the unwrapped JSON object.

        The IPL feeds wrap their payload in a callback, e.g.
        ``onScoring({"Innings1": ...})``; this strips the wrapper before
        parsing. Plain JSON bodies are also accepted.
        """
        payload = self.get_text(url, force_refresh=force_refresh)
        return unwrap_jsonp(payload, url=url)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def unwrap_jsonp(payload: str, *, url: str = "") -> dict[str, Any]:
    """Strip a JSONP callback wrapper and parse the enclosed JSON object."""
    text = payload.strip()
    if not text:
        raise ValueError(f"Empty payload from {url}")

    if _JSONP_RE.match(text):
        start = text.find("(")
        end = text.rfind(")")
        if start != -1 and end > start:
            text = text[start + 1 : end]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON from {url}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object from {url}, got {type(parsed).__name__}")
    return parsed
