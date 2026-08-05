# Data sources

The **official IPL website is the primary source**. Cricsheet supplements only
the seasons the official feed does not publish.

---

## Why feeds, not HTML scraping

`iplt20.com` is a single-page JavaScript application. Its HTML contains no match
data — the page fetches a set of public JSON feeds from the league's CDN and
renders them client-side.

Reading those feeds directly is therefore:

- **The same data**, one step earlier in the same pipeline the website uses;
- **More reliable**, because a CSS redesign does not break the parser;
- **Far lighter on the site**, since one request replaces a full page load plus
  every asset;
- **Structured**, so nothing has to be recovered from rendered markup.

No headless browser is used, and no page requiring JavaScript execution is
scraped.

---

## Official feed endpoints

Base URL:

```
https://ipl-stats-sports-mechanic.s3.ap-south-1.amazonaws.com/ipl/feeds
```

All four are JSONP — the payload is wrapped in a callback such as
`MatchSchedule({...})` or `onScoring({...})`, which `ipl.ingestion.http_client`
strips before parsing.

### 1. Season schedule — `{competition}-matchschedule.js`

Every fixture in a competition. Fields consumed:

| Feed field | Maps to |
|---|---|
| `MatchID`, `CompetitionID`, `CompetitionName` | Identity and season |
| `MatchDate`, `MATCH_COMMENCE_START_DATE` | Date and start time |
| `HomeTeamName`, `AwayTeamName`, `MatchName` | Participants |
| `FirstBattingTeamName`, `SecondBattingTeamName` | Innings order |
| `GroundName`, `city` | Venue |
| `TossTeam`, `TossDetails` | Toss winner and decision |
| `Comments` | Result — parsed into winner, type and margin |
| `FirstBattingSummary`, `SecondBattingSummary` | `"190/9 (20.0 Ov)"` → runs/wickets/overs |
| `MatchOrder`, `ROUND_ID` | Stage (League, Qualifier 1/2, Eliminator, Final) |
| `MatchStatus` | `Post` = completed |
| `GroundUmpire1/2`, `ThirdUmpire` | Officials |
| `MATCH_NO_OF_OVERS`, `MatchType` | Format, day/night |

### 2. Innings detail — `{match}-Innings{n}.js`

The richest feed. Contains `BattingCard`, `BowlingCard`, `FallOfWickets`,
`PartnershipScores`, `PartnershipBreak`, `Extras` and **`OverHistory`** — full
ball-by-ball data with commentary.

### 3. Match summary — `{match}-matchsummary.js`

Result confirmation, `MOM` (Player of the Match), `Target`, `RevisedOver` /
`RevisedTarget` (DLS), `IsSuperOver`, umpires and referee.

### 4. Squads — `{match}-squad.js`

`squadA` / `squadB`, each with `PlayerName`, `PlayingOrder`, `IsCaptain`,
`IsWK`, `IsNonDomestic`, `BattingType`, `BowlingProficiency`, `PlayerSkill`.
Players with `PlayingOrder` 1–11 are the Playing XI.

---

## Feed quirks handled

These were found by reconciling parsed output against published scorecards, and
each has a regression test.

### Over-break sentinel rows

`OverHistory` terminates each over with a row whose `BallNo` is `"99"` and whose
other fields are blank. It is a rendering marker, not a delivery. Left in, a
20-over innings appears to contain 123 legal balls instead of 120.

```python
if ball_no_raw == "99" or not clean_text(row.get("ActualBallNo")):
    continue
```

### `BallRuns` is `"W"` on a wicket

The field is a *display* string, so it cannot be summed. Run totals are rebuilt
from `ActualRuns` (off the bat) plus `Extras`:

```python
batter_runs = to_int(row.get("ActualRuns"), 0)
extra_runs  = to_int(row.get("Extras"), 0)
total_runs  = batter_runs + extra_runs
```

Verified against the IPL 2025 final: 181 + 9 = 190, matching the published
`190/9`.

### `BallNo` vs `ActualBallNo`

`BallNo` counts legal balls within the over; `ActualBallNo` counts every
delivery including wides and no-balls. The warehouse stores the latter and marks
legality separately.

### `"0"` as a sentinel team name

A fixture abandoned before the toss reports `FirstBattingTeamName: "0"`. Treated
as a placeholder, not a franchise — otherwise a team literally named `0` appears
in the dimension table.

### `TBD` playoff placeholders

Before the league stage ends, playoff slots are published with both sides as
`TBD`. These are skipped quietly rather than reported as invalid data.

### Cricket over notation

`19.4` overs means 19 overs and 4 balls — 118 deliveries, not `19.4 × 6 = 116`.
`overs_to_balls` / `balls_to_overs` handle this and are unit-tested across the
full 0–120 range.

---

## Competition ID registry

The feed host indexes every tournament it has ever served — national tours, the
WPL, ICC events — under one flat integer namespace. IPL seasons must be
identified by name.

| Season | ID | Season | ID |
|---|---|---|---|
| 2019 | 8 | 2023 | 107 |
| 2020 | 24 | 2024 | 148 |
| 2021 | 33 | 2025 | 203 |
| 2022 | 60 | 2026 | 284 |

Seasons before 2019 are **not indexed** by this host, which is exactly why the
Cricsheet supplement exists.

When a new season starts, discover its ID and add it to `IPL_COMPETITIONS` in
`src/ipl/constants.py`:

```bash
python scripts/discover_competitions.py --start 250 --end 450
```

---

## Cricsheet supplement (2008–2018)

Source: <https://cricsheet.org/downloads/ipl_json.zip> (~5 MB, 1,243 matches).

Used **only** for seasons the official feed does not publish. A season present in
the official registry is never overwritten.

Cricsheet ships **no scorecards** — only deliveries — so batting cards, bowling
cards, fall of wickets and partnerships are all *derived* from the ball-by-ball
stream using standard scoring conventions:

- Balls faced by a batter exclude wides but include no-balls.
- Runs conceded by a bowler exclude byes and leg-byes, include wides and
  no-balls.
- Only bowled, caught, LBW, stumped, caught-and-bowled and hit-wicket are
  credited to the bowler; run-outs and retirements are not.
- A maiden is a completed over in which the bowler conceded nothing.

### Season labels are ambiguous

Cricsheet labels IPL 2008 as `2007/08`, IPL 2010 as `2009/10` and IPL 2020 as
`2020/21` — but IPL 2021 as `2021`. The label alone cannot be mapped
consistently.

The season is therefore taken from the **year of the first match date**. No IPL
season has ever crossed a calendar year, so this is unambiguous. Verified: the
label-to-date-year mapping is 1:1 across all 1,243 archived matches.

Cricsheet is published under the
[Open Data Commons Attribution Licence](https://cricsheet.org/register/).

---

## Responsible collection

| Practice | Implementation |
|---|---|
| Serialised requests | One at a time, never concurrent |
| Rate limiting | `IPL_REQUEST_DELAY`, default 0.6 s between requests |
| Retries | 4 attempts, exponential backoff with jitter |
| Graceful degradation | A stale cache entry is preferred over hammering the origin |
| Identification | Descriptive `User-Agent` naming the project |
| Caching | Payloads stored on disk; re-runs download nothing |
| Incremental | A match already stored *and* final is skipped |

**Measured effect.** A full first ingest of 2008–2026 takes ~22 minutes. A
re-ingest of 148 matches from cache takes **29 seconds**.

During development, one burst of ~200 concurrent requests caused the host to
reset connections. That is precisely why the client is serialised and backs off —
the polite path is also the reliable one.

---

## Data coverage

| Seasons | Source | Matches | Ball-by-ball | Scorecards | Playing XI |
|---|---|---|---|---|---|
| 2008–2018 | Cricsheet | 696 | ✅ | Derived | ✅ |
| 2019–2026 | **iplt20.com** | 550 | ✅ | ✅ Native | ✅ (except 2021) |

Totals: **1,246 matches · 280,125 deliveries · 1,457 players · 42 venues ·
15 franchises**.

### Known gaps

- **IPL 2021 Playing XIs** — the official squad feed returns data for only 2 of
  60 matches.
- **Player identity across sources** — Cricsheet's `V Kohli` and the feed's
  `Virat Kohli` are stored separately; see
  [DATABASE.md](DATABASE.md#players).
- **Boundaries in Cricsheet data** are inferred from runs off the bat.

---

## Validation

Every parsed match is checked before storage. Errors reject the record; warnings
are logged and counted.

| Severity | Check |
|---|---|
| **Error** | Winner did not play in the match |
| **Error** | Both sides are the same team |
| **Error** | Implausible innings score (>350) or wickets (>10) |
| **Error** | Duplicate innings number |
| **Warning** | Toss winner is not one of the two sides |
| **Warning** | Ball-by-ball total disagrees with the scorecard |
| **Warning** | Innings lasted more than 20.5 overs |
| **Warning** | Duplicate scorecard or delivery rows |

The full 2008–2026 load completes with **zero errors and zero warnings**.

Duplicate child rows are collapsed automatically (keeping the first occurrence),
so a feed that occasionally repeats a row cannot break a re-ingest.

---

## Legal note

This is an independent educational project. It reads publicly accessible
endpoints that `iplt20.com` serves to every visitor, at a deliberately
conservative rate. It is not affiliated with, endorsed by, or connected to the
BCCI or the Indian Premier League, and all trademarks belong to their owners.

If you fork this, keep the rate limiting in place.
