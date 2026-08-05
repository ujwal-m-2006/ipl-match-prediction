"""Data-quality validation applied to every parsed match before it is stored.

Two severities:

``ERROR``
    The record is internally inconsistent in a way that would corrupt
    downstream analysis (e.g. a winner who did not play in the match). These
    matches are rejected.
``WARNING``
    Something is odd but usable (e.g. ball-by-ball totals a run or two off the
    scorecard). The record is stored and the issue is logged and counted.

The report is surfaced on the dashboard's Admin page so data problems are
visible rather than silently absorbed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..constants import STANDARD_OVERS
from ..logging_utils import get_logger
from .records import MatchRecord

logger = get_logger(__name__)

# An IPL innings cannot plausibly exceed this; anything above indicates a
# parsing fault rather than a real score.
MAX_PLAUSIBLE_INNINGS_RUNS = 350
MAX_WICKETS = 10


@dataclass
class ValidationIssue:
    """One problem found in one match."""

    match_key: str
    severity: str  # "error" | "warning"
    code: str
    message: str


@dataclass
class ValidationReport:
    """Accumulated issues across an ingestion run."""

    issues: list[ValidationIssue] = field(default_factory=list)
    checked: int = 0
    rejected: int = 0

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def code_counts(self) -> dict[str, int]:
        """Issue counts grouped by code, most frequent first."""
        return dict(Counter(i.code for i in self.issues).most_common())

    def summary(self) -> str:
        counts = self.code_counts()
        detail = ", ".join(f"{code}={n}" for code, n in counts.items()) or "none"
        return (
            f"checked={self.checked} rejected={self.rejected} "
            f"errors={len(self.errors)} warnings={len(self.warnings)} [{detail}]"
        )


def validate_match(record: MatchRecord, report: ValidationReport) -> bool:
    """Validate one match. Returns ``False`` if it must be rejected.

    Appends every issue found to ``report``.
    """
    report.checked += 1
    errors_before = len(report.errors)

    def problem(severity: str, code: str, message: str) -> None:
        report.add(ValidationIssue(record.match_key, severity, code, message))

    # --- identity ---
    if not record.match_key:
        problem("error", "missing_key", "Match has no match_key")
    if not record.season:
        problem("error", "missing_season", "Match has no season")
    if record.match_date is None:
        problem("warning", "missing_date", "Match has no date")

    # --- participants ---
    teams = {t for t in (record.team1, record.team2) if t}
    if len(teams) < 2:
        problem("error", "missing_teams", f"Fewer than two teams: {sorted(teams)}")

    if record.team1 and record.team1 == record.team2:
        problem("error", "same_team", f"Both sides are {record.team1}")

    # --- toss ---
    if record.toss_winner and teams and record.toss_winner not in teams:
        problem(
            "warning",
            "toss_winner_not_playing",
            f"Toss winner {record.toss_winner!r} is not one of {sorted(teams)}",
        )
    if record.toss_decision and record.toss_decision not in {"bat", "field"}:
        problem(
            "warning", "bad_toss_decision", f"Unexpected toss decision {record.toss_decision!r}"
        )

    # --- result ---
    if record.is_completed and not (
        record.winner or record.is_tie or record.is_no_result
    ):
        problem("warning", "completed_no_result", "Marked completed but has no outcome")

    if record.winner and teams and record.winner not in teams:
        # A winner who did not play means team parsing went wrong; storing this
        # would silently corrupt every head-to-head and win-rate figure.
        problem(
            "error",
            "winner_not_playing",
            f"Winner {record.winner!r} is not one of {sorted(teams)}",
        )

    if record.win_margin_runs is not None and record.win_margin_wickets is not None:
        problem("warning", "double_margin", "Both run and wicket margins are set")

    if record.win_margin_wickets is not None and not 0 <= record.win_margin_wickets <= MAX_WICKETS:
        problem(
            "warning",
            "implausible_margin",
            f"Wicket margin {record.win_margin_wickets} out of range",
        )

    # --- innings ---
    seen_innings: set[int] = set()
    for innings in record.innings:
        if innings.innings_no in seen_innings:
            problem("error", "duplicate_innings", f"Innings {innings.innings_no} appears twice")
        seen_innings.add(innings.innings_no)

        if innings.runs is not None and not 0 <= innings.runs <= MAX_PLAUSIBLE_INNINGS_RUNS:
            problem(
                "error",
                "implausible_score",
                f"Innings {innings.innings_no} scored {innings.runs}",
            )

        if innings.wickets is not None and not 0 <= innings.wickets <= MAX_WICKETS:
            problem(
                "error",
                "implausible_wickets",
                f"Innings {innings.innings_no} lost {innings.wickets} wickets",
            )

        if innings.overs is not None and innings.overs > STANDARD_OVERS + 0.5:
            problem(
                "warning",
                "overs_exceed_limit",
                f"Innings {innings.innings_no} lasted {innings.overs} overs",
            )

        if innings.batting_team and teams and innings.batting_team not in teams:
            problem(
                "warning",
                "innings_team_mismatch",
                f"Innings {innings.innings_no} batting team {innings.batting_team!r} "
                f"not in {sorted(teams)}",
            )

    # --- cross-checks between deliveries and the scorecard ---
    if record.deliveries:
        by_innings: dict[int, int] = {}
        for delivery in record.deliveries:
            by_innings[delivery.innings_no] = (
                by_innings.get(delivery.innings_no, 0) + delivery.total_runs
            )
        for innings in record.innings:
            derived = by_innings.get(innings.innings_no)
            if derived is None or innings.runs is None:
                continue
            if derived != innings.runs:
                problem(
                    "warning",
                    "ballbyball_mismatch",
                    f"Innings {innings.innings_no}: deliveries sum to {derived} "
                    f"but scorecard says {innings.runs}",
                )

    # --- duplicate child rows ---
    _check_duplicates(record, problem)

    rejected = len(report.errors) > errors_before
    if rejected:
        report.rejected += 1
        logger.warning(
            "Rejecting %s: %s",
            record.match_key,
            "; ".join(i.message for i in report.errors[errors_before:]),
        )
    return not rejected


def _check_duplicates(record: MatchRecord, problem) -> None:  # noqa: ANN001
    """Flag duplicated keys inside a match's child collections."""
    batting_keys = Counter((b.innings_no, b.player) for b in record.batting)
    for key, count in batting_keys.items():
        if count > 1:
            problem("warning", "duplicate_batting", f"Batting card duplicated for {key}")

    bowling_keys = Counter((b.innings_no, b.player) for b in record.bowling)
    for key, count in bowling_keys.items():
        if count > 1:
            problem("warning", "duplicate_bowling", f"Bowling card duplicated for {key}")

    delivery_keys = Counter((d.innings_no, d.ball_seq) for d in record.deliveries)
    for key, count in delivery_keys.items():
        if count > 1:
            problem("warning", "duplicate_delivery", f"Delivery duplicated at {key}")


def deduplicate_match(record: MatchRecord) -> MatchRecord:
    """Drop duplicate child rows in place, keeping the first occurrence.

    The database enforces these as unique constraints, so silently collapsing
    duplicates here is what keeps a re-ingest from failing on a feed that
    occasionally repeats a row.
    """
    record.batting = _unique_by(record.batting, lambda b: (b.innings_no, b.player))
    record.bowling = _unique_by(record.bowling, lambda b: (b.innings_no, b.player))
    record.fall_of_wickets = _unique_by(
        record.fall_of_wickets, lambda f: (f.innings_no, f.wicket_no)
    )
    record.partnerships = _unique_by(
        record.partnerships, lambda p: (p.innings_no, p.wicket_no)
    )
    record.deliveries = _unique_by(record.deliveries, lambda d: (d.innings_no, d.ball_seq))
    record.squads = _unique_by(record.squads, lambda s: (s.team, s.player))
    record.innings = _unique_by(record.innings, lambda i: i.innings_no)
    return record


def _unique_by(items: list, key) -> list:  # noqa: ANN001
    """Return ``items`` with later duplicates of ``key`` removed."""
    seen = set()
    out = []
    for item in items:
        k = key(item)
        if k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out
