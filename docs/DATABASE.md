# Database schema

Twelve tables holding 19 IPL seasons: 1,246 matches, 280,125 deliveries, 21,771
batting-card rows and 1,457 players.

The application creates the schema itself via SQLAlchemy, so nothing here needs
to be run by hand:

```bash
python scripts/init_db.py
```

Reference DDL for provisioning out-of-band lives in
[`../sql/schema_postgres.sql`](../sql/schema_postgres.sql) and
[`../sql/schema_mysql.sql`](../sql/schema_mysql.sql).

---

## Supported backends

| Backend | URL | Notes |
|---|---|---|
| **SQLite** | `sqlite:///data/ipl.db` | Default. Zero setup. WAL mode so the dashboard can read while the pipeline writes |
| **PostgreSQL** | `postgresql+psycopg2://user:pass@host:5432/ipl` | Recommended for deployment |
| **MySQL / MariaDB** | `mysql+pymysql://user:pass@host:3306/ipl` | `utf8mb4` required — player names contain non-ASCII characters |

Relative SQLite paths resolve against the project root, not the working
directory, so the CLI and the dashboard always open the same file.

---

## Design decisions

**Surrogate keys plus a natural unique key.** Every table has an integer primary
key. Matches additionally carry `match_key`, formatted `<source>:<season>:<id>`,
because the official feed and Cricsheet number matches independently — feed match
`1872` and Cricsheet match `1872` are unrelated. Namespacing them is what makes
re-ingestion idempotent.

**Names are canonicalised before insert.** `Delhi Daredevils` and `Delhi Capitals`
both resolve to the current branding, so `teams.name` is safe to join and group
on. The mapping lives in `src/ipl/constants.py`.

**Children cascade from the match.** Re-ingesting a match deletes its child rows
and re-inserts them. Simpler than diffing, and correct because a re-ingest is
always a full re-parse of the same fixture.

**Explicit column widths.** MySQL cannot index an unbounded `TEXT`, so every
indexed string column has a length.

**Booleans as 0/1 in SQLite.** The repository layer normalises these back to
Python `bool` on read, so downstream code is backend-agnostic.

---

## Entity relationships

```
teams ─────┬──< matches >──┬───── venues
           │               │
players ───┼───────────────┤
           │               │
           │       ┌───────┴────────────────────────────┐
           │       │                                    │
           └──< batting_cards      innings        deliveries >──┘
               bowling_cards       partnerships   match_players
               fall_of_wickets

ingestion_runs   (standalone audit log)
```

---

## Dimension tables

### `teams`

One row per franchise, keyed on its **current** name.

| Column | Type | Notes |
|---|---|---|
| `id` | int PK | |
| `name` | varchar(80) unique | Canonical current branding |
| `short_code` | varchar(10) | `CSK`, `MI`, `RCB`, … |
| `is_active` | bool | Fielded a side in the most recent season |
| `primary_color` | varchar(16) | Brand colour used by the dashboard |
| `logo_url` | varchar(400) | |

15 rows: 10 active franchises plus 5 defunct (Deccan Chargers, Pune Warriors,
Kochi Tuskers Kerala, Gujarat Lions, Rising Pune Supergiant).

### `venues`

| Column | Type | Notes |
|---|---|---|
| `id` | int PK | |
| `name` | varchar(160) unique | Sponsor prefixes and trailing city names stripped |
| `city` | varchar(80) | Stored separately from the ground name |
| `country` | varchar(60) | Defaults to India; covers the 2009 South Africa and UAE seasons |

42 rows.

### `players`

| Column | Type | Notes |
|---|---|---|
| `id` | int PK | |
| `name` | varchar(120) unique | Captain/keeper markers stripped |
| `source_player_id` | varchar(80) | The official feed's opaque GUID, when known |
| `batting_style` | varchar(60) | e.g. `Right Hand Batsman` |
| `bowling_style` | varchar(80) | e.g. `Right Arm Off Spinner` |
| `role` | varchar(40) | Batsman / Bowler / All-Rounder / Wicketkeeper |
| `is_overseas` | bool | |

1,457 rows.

> **Known limitation.** Cricsheet writes `V Kohli`; the official feed writes
> `Virat Kohli`. These are stored as distinct players. Matching on surname plus
> first initial would collide (`R Sharma` is both Rohit and Rahul), so no
> automatic reconciliation is attempted. Player analytics are therefore
> era-scoped: 2008–2018 and 2019–2026 use different name spaces.

---

## Fact tables

### `matches`

One row per fixture, completed or scheduled. 1,246 rows.

**Identity and provenance**

| Column | Notes |
|---|---|
| `match_key` | `<source>:<season>:<source_id>`, globally unique |
| `source` | `iplt20` (primary) or `cricsheet` (2008–2018 supplement) |
| `source_match_id` | The source's own ID |
| `competition_id` | The official feed's competition number |

**Scheduling**

`season`, `match_date`, `start_datetime`, `match_number`, `stage`, `is_playoff`,
`is_day_night`, `overs_per_innings`.

`stage` is one of `League`, `Qualifier 1`, `Qualifier 2`, `Eliminator`, `Final`.

**Location** — `venue_id`, `city`, `is_neutral_venue`.

`is_neutral_venue` is true when the nominal home side is not at its own ground.
This matters: IPL 2009 (South Africa), 2014, 2020 and part of 2021 (UAE) were
played away from home, and treating those as home advantage would inject a
systematic bias into the models.

**Participants** — `team1_id`, `team2_id`, `home_team_id`, `away_team_id`.

**Toss** — `toss_winner_id`, `toss_decision` (`bat` | `field`).

**Innings order** — `first_batting_team_id`, `second_batting_team_id`. Not
derivable from the toss alone, since the toss winner may choose either.

**Outcome**

| Column | Notes |
|---|---|
| `is_completed` | The filter the training pipeline uses |
| `winner_id` | NULL for a no-result |
| `result_type` | `runs` \| `wickets` \| `tie` \| `no result` |
| `win_margin_runs` / `win_margin_wickets` | Exactly one is set |
| `is_tie`, `is_super_over`, `is_no_result`, `is_dls_applied` | |
| `target_runs` | Second-innings target |
| `player_of_match_id` | |
| `result_summary` | The source's own sentence, kept verbatim |

**Officials** — `umpire1`, `umpire2`, `third_umpire`, `match_referee`.

**Indexes** — `season`, `match_date`, `is_completed`, `(season, match_date)`,
`(team1_id, team2_id)`, `winner_id`, `venue_id`.

### `innings`

Per-innings totals with phase splits. 2,483 rows. Unique on
`(match_id, innings_no)`.

Totals: `runs`, `wickets`, `overs` (cricket notation, e.g. `19.4`), `balls`
(legal deliveries), `run_rate`.

Extras: `extras`, `byes`, `leg_byes`, `wides`, `no_balls`, `penalty`.

Phase splits, derived from ball-by-ball data:

| Column pair | Overs |
|---|---|
| `powerplay_runs` / `powerplay_wickets` | 1–6 |
| `middle_runs` / `middle_wickets` | 7–15 |
| `death_runs` / `death_wickets` | 16–20 |

Also `fours`, `sixes`, `dot_balls`, `target`.

> `overs` uses cricket notation, where `19.4` means 19 overs and 4 balls — *not*
> 19.4 decimal overs. Use `balls` for arithmetic. `ipl.ingestion.normalize`
> provides `overs_to_balls` / `balls_to_overs`, and the conversion is unit-tested.

### `batting_cards`

One row per batter per innings. 21,771 rows. Unique on
`(match_id, innings_no, player_id)`.

`batting_position`, `runs`, `balls`, `fours`, `sixes`, `dot_balls`,
`strike_rate`, `is_out`, `dismissal_kind`, `dismissal_text`, `bowler_id`,
`fielder_id`, `wicket_number`.

`is_out` distinguishes a dismissal from a not-out innings, which is what makes a
correct batting average (runs ÷ *dismissals*) possible.

### `bowling_cards`

One row per bowler per innings. 13,949 rows. Unique on
`(match_id, innings_no, player_id)`.

`bowling_order`, `overs`, `balls`, `maidens`, `runs_conceded`, `wickets`,
`wides`, `no_balls`, `dot_balls`, `economy`.

For Cricsheet seasons these are **derived** from ball-by-ball data using standard
scoring conventions: byes and leg-byes are not charged to the bowler; only
bowled, caught, LBW, stumped, caught-and-bowled and hit-wicket are credited as
their wickets.

### `deliveries`

The ball-by-ball grain and the largest table: **280,125 rows**. Unique on
`(match_id, innings_no, ball_seq)`.

| Column group | Columns |
|---|---|
| Position | `over_no` (1-based), `ball_no`, `ball_seq` |
| Participants | `batting_team_id`, `bowling_team_id`, `batter_id`, `non_striker_id`, `bowler_id` |
| Runs | `batter_runs`, `extra_runs`, `total_runs` |
| Flags | `is_wide`, `is_no_ball`, `is_bye`, `is_leg_bye`, `is_legal`, `is_four`, `is_six`, `is_wicket` |
| Dismissal | `wicket_type`, `dismissed_player_id` |
| Running state | `cumulative_runs`, `cumulative_wickets` |

`is_legal` is false for wides and no-balls, which do not count towards the over.
`cumulative_runs`/`cumulative_wickets` are precomputed so live chase inference
needs no window function.

Boundary flags for Cricsheet seasons are inferred from runs off the bat (4 or 6),
the standard convention — an all-run four is indistinguishable.

### `partnerships`

22,306 rows. Unique on `(match_id, innings_no, wicket_no)`. Runs, balls, each
batter's contribution, start/end over and `is_unbroken` for a partnership still
going when the innings ended.

### `fall_of_wickets`

13,919 rows. Unique on `(match_id, innings_no, wicket_no)`. `fall_score` and
`fall_overs` give the classic "18/1 (1.4)" reading.

### `match_players`

Squad and Playing XI membership. 28,818 rows. Unique on
`(match_id, team_id, player_id)`.

`is_playing_xi`, `is_captain`, `is_wicketkeeper`, `is_overseas`,
`is_impact_sub`, `playing_order`, `role`.

Covers 1,179 of 1,246 matches. **IPL 2021 is largely missing** — the official
squad feed returns data for only 2 of that season's 60 matches.

---

## Operational tables

### `ingestion_runs`

Audit log for the pipeline. The dashboard's Admin page reads the latest row to
report data freshness.

`started_at`, `finished_at`, `status` (`running` | `success` | `failed`),
`trigger` (`cli` | `dashboard` | `schedule`), `sources`, `seasons`,
`matches_seen`, `matches_inserted`, `matches_updated`, `matches_skipped`,
`deliveries_inserted`, `duration_seconds`, `message`.

---

## Views

Both reference schemas define convenience views:

- **`v_matches`** — matches joined to human-readable team, venue and player
  names. Mirrors the join the application performs.
- **`v_team_records`** (PostgreSQL) — all-time franchise record computed from
  both team slots.

---

## Useful queries

**Highest team totals**

```sql
SELECT t.name AS team, i.runs, i.wickets, i.overs, m.season, v.name AS venue
FROM innings i
JOIN matches m ON m.id = i.match_id
JOIN teams   t ON t.id = i.batting_team_id
JOIN venues  v ON v.id = m.venue_id
ORDER BY i.runs DESC
LIMIT 10;
```

**Leading run scorers, with a correct average**

```sql
SELECT p.name,
       COUNT(DISTINCT b.match_id)        AS innings,
       SUM(b.runs)                       AS runs,
       SUM(b.balls)                      AS balls,
       ROUND(SUM(b.runs) * 100.0 / NULLIF(SUM(b.balls), 0), 2) AS strike_rate,
       -- Runs per DISMISSAL, not per innings.
       ROUND(SUM(b.runs) * 1.0 / NULLIF(SUM(CASE WHEN b.is_out THEN 1 ELSE 0 END), 0), 2)
                                         AS average
FROM batting_cards b
JOIN players p ON p.id = b.player_id
GROUP BY p.name
HAVING SUM(b.balls) > 500
ORDER BY runs DESC
LIMIT 20;
```

**Death-overs economy (overs 16–20)**

```sql
SELECT p.name,
       COUNT(*)                                              AS balls,
       SUM(d.total_runs)                                     AS runs,
       ROUND(SUM(d.total_runs) * 6.0 / COUNT(*), 2)          AS economy,
       SUM(CASE WHEN d.is_wicket THEN 1 ELSE 0 END)          AS wickets
FROM deliveries d
JOIN players p ON p.id = d.bowler_id
WHERE d.over_no >= 16 AND d.is_legal
GROUP BY p.name
HAVING COUNT(*) >= 300
ORDER BY economy ASC
LIMIT 20;
```

**Does batting first help, by season?**

```sql
SELECT m.season,
       COUNT(*)                                                       AS matches,
       SUM(CASE WHEN m.winner_id = m.first_batting_team_id THEN 1 ELSE 0 END)
                                                                      AS bat_first_wins,
       ROUND(100.0 * SUM(CASE WHEN m.winner_id = m.first_batting_team_id THEN 1 ELSE 0 END)
             / COUNT(*), 1)                                           AS bat_first_pct
FROM matches m
WHERE m.is_completed AND m.winner_id IS NOT NULL
GROUP BY m.season
ORDER BY m.season;
```

---

## Sizing

| Backend | Full history (with deliveries) | Without deliveries |
|---|---|---|
| SQLite | ~145 MB | ~14 MB |
| PostgreSQL | ~210 MB | ~20 MB |

Set `IPL_INGEST_DELIVERIES=false` to skip ball-by-ball data. Everything works
except the chase model and phase analytics.

---

## Migrations

Alembic is included in `requirements.txt` but no migration chain is committed —
the schema is created from the ORM models. If you fork this and need versioned
migrations:

```bash
alembic init migrations
alembic revision --autogenerate -m "initial"
alembic upgrade head
```

Point `sqlalchemy.url` in `alembic.ini` at `IPL_DATABASE_URL` and import
`ipl.db.models` in `migrations/env.py` so autogenerate can see the metadata.
