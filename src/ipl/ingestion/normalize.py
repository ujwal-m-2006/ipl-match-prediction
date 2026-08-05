"""Parsing and canonicalisation helpers.

Both data sources describe the same events with different spellings and free
text ("Royal Challengers Bengaluru Won by 6 Runs (Winners)"). Everything that
turns raw feed strings into typed, canonical values lives here so the two
client modules stay thin and the rules are unit-testable in isolation.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

from ..constants import (
    BALLS_PER_OVER,
    PLAYOFF_KEYWORDS,
    TEAM_ALIASES,
    TEAM_CODES,
    VENUE_ALIASES,
    VENUE_CITIES,
)

# Suffixes the feeds append to player names inside squad listings.
_PLAYER_SUFFIX_RE = re.compile(r"\s*\((?:c|wk|c/?wk|wk/?c|sub)\)\s*$", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
# "190/9 (20.0 Ov)" / "184/7 (20.0 Overs)"
_SUMMARY_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\(\s*([\d.]+)\s*Ov", re.IGNORECASE)
# "Won by 6 Runs" / "won by 4 wickets"
_MARGIN_RE = re.compile(r"won\s+by\s+(\d+)\s*(run|wicket)", re.IGNORECASE)

# Stand-ins the feeds use where a real team is not known: playoff slots before
# the league stage decides who fills them, and "0" on the innings fields of a
# fixture abandoned before a ball was bowled. None of these name a real side.
PLACEHOLDER_TEAM_NAMES = frozenset(
    {
        "tbd", "tba", "to be decided", "to be announced",
        "qualifier", "winner", "loser", "-", "0", "n/a", "na", "none", "null",
    }
)


# ---------------------------------------------------------------------------
# Generic scalar coercion
# ---------------------------------------------------------------------------
def clean_text(value: Any) -> str | None:
    """Collapse whitespace and map empty/``NaN``-ish values to ``None``."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = _WHITESPACE_RE.sub(" ", str(value)).strip()
    return text or None


def to_int(value: Any, default: int | None = None) -> int | None:
    """Best-effort integer coercion; feeds send ints as ``"3"`` or ``""``."""
    if value is None or value == "":
        return default
    try:
        return int(str(value).strip().split(".")[0])
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float | None = None) -> float | None:
    """Best-effort float coercion."""
    if value is None or value == "":
        return default
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def to_bool(value: Any, default: bool = False) -> bool:
    """Coerce the feeds' ``"1"``/``"0"``/``1``/``True`` spellings to ``bool``."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_date(value: Any) -> date | None:
    """Parse the several date spellings used across the feeds.

    Handles ``2025-06-03``, ``03 Jun 2025``, ``2025-06-03 19:30:00`` and
    ``3 Jun 2025``.
    """
    text = clean_text(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%d %b %Y", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text[: len(fmt) + 4], fmt).date()
        except ValueError:
            continue
    # Last resort: let pandas' flexible parser try.
    try:
        import pandas as pd

        parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
        return None if parsed is None or parsed is not parsed else parsed.date()
    except Exception:  # pragma: no cover - defensive
        return None


def parse_datetime(value: Any) -> datetime | None:
    """Parse ``"2025-06-03 19:30:00"``-style timestamps."""
    text = clean_text(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Cricket-specific numeric conversions
# ---------------------------------------------------------------------------
def overs_to_balls(overs: Any) -> int | None:
    """Convert cricket over notation (``19.4``) to a ball count (``118``).

    Note that ``.4`` means "4 balls", not four tenths -- a naive ``float * 6``
    would be wrong, which is exactly the bug this helper exists to prevent.
    """
    value = to_float(overs)
    if value is None:
        return None
    whole = int(value)
    # Round to kill float noise like 19.399999999999999.
    balls = int(round((value - whole) * 10))
    balls = min(balls, BALLS_PER_OVER - 1)
    return whole * BALLS_PER_OVER + balls


def balls_to_overs(balls: Any) -> float | None:
    """Inverse of :func:`overs_to_balls` (``118`` -> ``19.4``)."""
    count = to_int(balls)
    if count is None:
        return None
    return round(count // BALLS_PER_OVER + (count % BALLS_PER_OVER) / 10, 1)


def run_rate(runs: Any, balls: Any) -> float | None:
    """Runs per over, or ``None`` when no legal ball has been bowled."""
    r, b = to_int(runs), to_int(balls)
    if r is None or not b:
        return None
    return round(r * BALLS_PER_OVER / b, 3)


# ---------------------------------------------------------------------------
# Canonical names
# ---------------------------------------------------------------------------
def canonical_team(name: Any) -> str | None:
    """Map any observed team spelling to its canonical franchise name.

    Unknown names pass through unchanged rather than being dropped, so a new
    franchise still lands in the database (and shows up as a data-quality
    warning) instead of silently disappearing. Placeholder names used for
    not-yet-determined playoff fixtures resolve to ``None``.
    """
    text = clean_text(name)
    if not text:
        return None
    key = text.lower().strip(" .")
    # A purely numeric "name" is always a sentinel, never a franchise.
    if key in PLACEHOLDER_TEAM_NAMES or key.isdigit():
        return None
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    # Try again after stripping a trailing city qualifier, e.g. "Mumbai Indians, Mumbai".
    head = key.split(",")[0].strip()
    if head in TEAM_ALIASES:
        return TEAM_ALIASES[head]
    return text


def canonical_venue(name: Any, city: Any = None) -> tuple[str | None, str | None]:
    """Return ``(canonical_venue_name, city)`` from a raw ground string.

    The official feed writes ``"Narendra Modi Stadium, Ahmedabad"`` while
    Cricsheet writes ``"Narendra Modi Stadium"`` with the city in a separate
    field; both collapse to the same pair here.
    """
    text = clean_text(name)
    resolved_city = clean_text(city)
    if not text:
        return None, resolved_city

    key = text.lower().strip(" .")
    if key in VENUE_ALIASES:
        canonical = VENUE_ALIASES[key]
        return canonical, resolved_city or VENUE_CITIES.get(canonical)

    # Progressively drop trailing comma-separated qualifiers ("X, Chepauk, Chennai").
    parts = [p.strip() for p in text.split(",")]
    for stop in range(len(parts) - 1, 0, -1):
        candidate = ", ".join(parts[:stop]).lower().strip(" .")
        if candidate in VENUE_ALIASES:
            canonical = VENUE_ALIASES[candidate]
            trailing = clean_text(parts[stop]) if stop < len(parts) else None
            return canonical, resolved_city or VENUE_CITIES.get(canonical) or trailing

    # Unknown ground: keep the head segment, treat the tail as the city.
    head = parts[0]
    tail = parts[-1] if len(parts) > 1 else None
    return head, resolved_city or VENUE_CITIES.get(head) or tail


def canonical_player(name: Any) -> str | None:
    """Strip captain/keeper markers and normalise whitespace in a player name."""
    text = clean_text(name)
    if not text:
        return None
    text = _PLAYER_SUFFIX_RE.sub("", text).strip()
    # A few feed rows carry a trailing team qualifier: "Krunal Pandya (RCB)".
    if text.endswith(")") and "(" in text:
        head, _, tail = text.rpartition("(")
        inner = tail.rstrip(")").strip()
        if canonical_team(inner) in TEAM_CODES or inner.upper() in TEAM_CODES.values():
            text = head.strip()
    return _WHITESPACE_RE.sub(" ", text).strip() or None


# ---------------------------------------------------------------------------
# Free-text match facts
# ---------------------------------------------------------------------------
def parse_toss(toss_details: Any, toss_team: Any = None) -> tuple[str | None, str | None]:
    """Extract ``(toss_winner, decision)`` from the feed's toss sentence.

    Example input: ``"Punjab Kings Won The Toss And Elected To Field"``.
    Returns decision as ``"bat"`` or ``"field"``.
    """
    text = clean_text(toss_details)
    winner = canonical_team(toss_team)

    if not text:
        return winner, None

    lowered = text.lower()
    decision: str | None = None
    if "elected to bat" in lowered or "opt to bat" in lowered or "chose to bat" in lowered:
        decision = "bat"
    elif (
        "elected to field" in lowered
        or "elected to bowl" in lowered
        or "opt to field" in lowered
        or "chose to field" in lowered
    ):
        decision = "field"

    if winner is None:
        # The team name is the text preceding "won the toss".
        match = re.split(r"\s+won\s+the\s+toss", text, flags=re.IGNORECASE)
        if match and match[0] != text:
            winner = canonical_team(match[0])

    return winner, decision


def parse_result(comments: Any) -> dict[str, Any]:
    """Parse the feed's result sentence into structured outcome fields.

    Recognised shapes::

        "Chennai Super Kings Won by 6 Runs (Winners)"
        "Mumbai Indians won by 4 wickets"
        "Match Tied (Kolkata Knight Riders Won the One Over Eliminator)"
        "Match Abandoned without a ball bowled"

    Returns a dict with ``winner``, ``result_type``, ``win_margin_runs``,
    ``win_margin_wickets``, ``is_tie``, ``is_no_result``, ``is_super_over``.
    """
    out: dict[str, Any] = {
        "winner": None,
        "result_type": None,
        "win_margin_runs": None,
        "win_margin_wickets": None,
        "is_tie": False,
        "is_no_result": False,
        "is_super_over": False,
        "result_summary": clean_text(comments),
    }
    text = clean_text(comments)
    if not text:
        return out

    lowered = text.lower()

    if "abandon" in lowered or "no result" in lowered or "washed out" in lowered:
        out["is_no_result"] = True
        out["result_type"] = "no result"
        return out

    if "tied" in lowered or "match tie" in lowered:
        out["is_tie"] = True
        out["result_type"] = "tie"
        if "eliminator" in lowered or "super over" in lowered:
            out["is_super_over"] = True
            # "(Kolkata Knight Riders Won the One Over Eliminator)"
            inner = re.search(r"\(([^)]*won[^)]*)\)", text, re.IGNORECASE)
            if inner:
                head = re.split(r"\s+won\s+", inner.group(1), flags=re.IGNORECASE)[0]
                out["winner"] = canonical_team(head)
        return out

    margin = _MARGIN_RE.search(text)
    # The winning team's name is whatever precedes "won by".
    head = re.split(r"\s+won\s+by\s+", text, flags=re.IGNORECASE)[0]
    if head and head != text:
        out["winner"] = canonical_team(head)

    if margin:
        value, unit = int(margin.group(1)), margin.group(2).lower()
        if unit.startswith("run"):
            out["result_type"] = "runs"
            out["win_margin_runs"] = value
        else:
            out["result_type"] = "wickets"
            out["win_margin_wickets"] = value

    if "super over" in lowered or "eliminator" in lowered:
        out["is_super_over"] = True

    return out


def parse_innings_summary(summary: Any) -> dict[str, Any]:
    """Parse ``"190/9 (20.0 Ov)"`` into runs / wickets / overs / balls."""
    out: dict[str, Any] = {"runs": None, "wickets": None, "overs": None, "balls": None}
    text = clean_text(summary)
    if not text:
        return out
    match = _SUMMARY_RE.search(text)
    if not match:
        return out
    out["runs"] = int(match.group(1))
    out["wickets"] = int(match.group(2))
    out["overs"] = float(match.group(3))
    out["balls"] = overs_to_balls(out["overs"])
    return out


def parse_player_of_match(value: Any) -> str | None:
    """Extract the bare player name from ``"Krunal Pandya (RCB)"``."""
    return canonical_player(value)


def parse_dismissal(out_desc: Any) -> dict[str, Any]:
    """Classify a dismissal string into a kind plus the bowler/fielder involved.

    Examples::

        "c Shreyas Iyer b Kyle Jamieson" -> caught,  bowler=Kyle Jamieson
        "b Jasprit Bumrah"               -> bowled,  bowler=Jasprit Bumrah
        "run out (Virat Kohli)"          -> run out, fielder=Virat Kohli
        "not out"                        -> not out (is_out False)
    """
    out: dict[str, Any] = {
        "is_out": True, "dismissal_kind": None, "bowler": None, "fielder": None,
        "dismissal_text": clean_text(out_desc),
    }
    text = clean_text(out_desc)
    if not text:
        out["is_out"] = False
        return out

    lowered = text.lower()
    if lowered in {"not out", "notout", "-", "did not bat", "dnb"} or lowered.startswith("not out"):
        out["is_out"] = False
        out["dismissal_kind"] = "not out"
        return out

    if lowered.startswith("run out"):
        out["dismissal_kind"] = "run out"
        inner = re.search(r"\(([^)]*)\)", text)
        if inner:
            out["fielder"] = canonical_player(inner.group(1).split("/")[0])
        return out

    if lowered.startswith("st "):
        out["dismissal_kind"] = "stumped"
    elif lowered.startswith("c and b") or lowered.startswith("c & b"):
        out["dismissal_kind"] = "caught and bowled"
    elif lowered.startswith("c "):
        out["dismissal_kind"] = "caught"
    elif lowered.startswith("b "):
        out["dismissal_kind"] = "bowled"
    elif "lbw" in lowered:
        out["dismissal_kind"] = "lbw"
    elif "hit wicket" in lowered:
        out["dismissal_kind"] = "hit wicket"
    elif "retired" in lowered:
        out["dismissal_kind"] = "retired hurt"
        out["is_out"] = False
    elif "obstruct" in lowered:
        out["dismissal_kind"] = "obstructing the field"
    else:
        out["dismissal_kind"] = "other"

    # Fielder sits inside "c <fielder> b <bowler>".
    fielder = re.match(r"(?:c|st)\s+(.+?)\s+b\s+", text, re.IGNORECASE)
    if fielder:
        out["fielder"] = canonical_player(fielder.group(1))

    # Bowler is whatever follows the final " b ".
    bowler = re.search(r"(?:^|\s)b\s+(.+)$", text, re.IGNORECASE)
    if bowler:
        out["bowler"] = canonical_player(bowler.group(1))

    return out


def detect_stage(*hints: Any) -> tuple[str | None, bool]:
    """Infer the tournament stage from any available label.

    Returns ``(stage, is_playoff)`` -- e.g. ``("Final", True)`` or
    ``("League", False)``.
    """
    joined = " ".join(clean_text(h) or "" for h in hints).lower()
    if not joined.strip():
        return None, False

    # Check the most specific labels first: "Qualifier 1" also contains "final"
    # in some feeds' round names, so order matters.
    if "eliminator" in joined:
        return "Eliminator", True
    if "qualifier 1" in joined or "qualifier1" in joined:
        return "Qualifier 1", True
    if "qualifier 2" in joined or "qualifier2" in joined:
        return "Qualifier 2", True
    if "qualifier" in joined:
        return "Qualifier", True
    if "semi" in joined:
        return "Semi Final", True
    if "3rd place" in joined:
        return "3rd Place", True
    if "final" in joined:
        return "Final", True
    if any(keyword in joined for keyword in PLAYOFF_KEYWORDS):
        return "Playoff", True
    return "League", False


def season_from_competition_name(name: Any) -> int | None:
    """Pull the four-digit year out of ``"Tata Ipl 2025"``."""
    text = clean_text(name)
    if not text:
        return None
    match = re.search(r"(20\d{2})", text)
    return int(match.group(1)) if match else None


def make_match_key(source: str, season: int, source_match_id: Any) -> str:
    """Build the globally-unique match key stored on :class:`~ipl.db.Match`."""
    return f"{source}:{season}:{clean_text(source_match_id) or 'na'}"
