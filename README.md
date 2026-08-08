# 🏏 IPL Match Prediction & Analytics System

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=ujwal-m-2006%2Fipl-match-prediction&branch=main&mainModule=streamlit_app.py)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-242%20passing-brightgreen.svg)](tests/)
[![Licence](https://img.shields.io/badge/licence-MIT-green.svg)](LICENSE)

An end-to-end machine-learning system for the Indian Premier League: an automated
data pipeline that collects from the **official iplt20.com feeds**, a relational
warehouse, engineered features, four prediction models each selected from a field
of six algorithms, a multi-page Streamlit dashboard and a documented REST API.

**19 seasons · 1,246 matches · 280,125 deliveries · 1,457 players · 242 tests**

> **▶ Run it in one click.** The badge above deploys this repository to Streamlit
> Community Cloud. A prebuilt database and pre-trained models ship with the repo,
> so the app has full data the moment it boots — no ingest, no training, no
> configuration.

---

## Table of contents

- [What it does](#what-it-does)
- [Results](#results-measured-not-claimed)
- [Quickstart](#quickstart)
- [Data sources](#data-sources)
- [Architecture](#architecture)
- [Project layout](#project-layout)
- [The models](#the-models)
- [Dashboard](#dashboard)
- [REST API](#rest-api)
- [Database](#database)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Demonstrating the project](#demonstrating-the-project)
- [Keeping data fresh](#keeping-data-fresh)
- [Limitations](#limitations)
- [Licence & attribution](#licence--attribution)

---

## What it does

| Capability | Detail |
|---|---|
| **Automated collection** | Rate-limited, retrying, disk-cached client for the official IPL JSON feeds, with Cricsheet back-filling 2008–2018 |
| **Warehouse** | 12-table normalised schema on PostgreSQL, MySQL or SQLite |
| **Validation** | Every match is checked for internal consistency; inconsistent records are rejected, not silently stored |
| **Analytics** | Team, player, venue, head-to-head and phase-of-innings analysis |
| **Models** | Match winner · first-innings score · in-play chase success · Player of the Match · playoff qualification |
| **Comparison** | Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM and CatBoost, scored on held-out seasons |
| **Dashboard** | Nine-page Streamlit app with interactive Plotly charts |
| **API** | FastAPI service with OpenAPI docs at `/docs` |

---

## Results (measured, not claimed)

All figures are from **seasons held out of training entirely** (2024–2026, 215
matches). A random train/test split would leak the future into the past and
inflate every number here.

| Task | Best model | Headline metric | Other metrics |
|---|---|---|---|
| **Chase success** (in-play) | Random Forest | **ROC-AUC 0.897** | Accuracy 0.799 · F1 0.790 · Brier 0.144 |
| **Player of the Match** | Random Forest | **ROC-AUC 0.969** | Top-1 pick correct in **54%** of matches (from ~22 candidates) |
| **First-innings score** | Random Forest | **RMSE 38.2 runs** | MAE 30.7 · 37% of innings called within 20 runs |
| **Match winner** (pre-match) | CatBoost | **ROC-AUC 0.552** | Accuracy 0.554 |

### ⚠️ On the pre-match winner model — the honest version

The match-winner model scores **ROC-AUC ≈ 0.50–0.55**. That is close to a coin
toss, and it is reported here rather than hidden because it is a *finding*, not
a defect:

- **It was checked properly.** The model was re-evaluated at six independent
  cut-offs, holding out from the last two seasons (144 matches) up to the last
  eight (541 matches). Every window returned AUC between 0.50 and 0.55. That
  stability across 541 test matches rules out sampling noise.
- **The signal genuinely is not there.** No pre-match feature correlates with the
  result at more than **|r| ≈ 0.09** — not recent form, not head-to-head, not
  home advantage, not the Playing XI's career batting average. The home side
  wins ~52% of IPL matches. T20 is a short, high-variance format.
- **It was engineered properly anyway.** Features include Playing XI strength
  built from each selected player's career-to-date record, era-aware scoring
  levels, rest days, venue records and rolling form. The training set is
  *mirrored* so every fixture is seen from both sides, removing any advantage
  from the arbitrary "team 1 is listed first" convention. Those changes moved the
  model from **below** chance to chance. They did not invent signal that is
  absent from the data.
- **The contrast is the lesson.** The chase model, which sees the live match
  state, reaches AUC 0.90 on the same pipeline. Information — not algorithm
  choice — is what separates them.

A portfolio project reporting "78% accuracy" on pre-match IPL prediction is
almost always leaking the future into training. The numbers above are the real
ones.

---

## Quickstart

```bash
git clone https://github.com/ujwal-m-2006/ipl-match-prediction.git
```

```bash
cd ipl-match-prediction && python -m venv .venv && .venv/Scripts/activate
```

> On macOS/Linux use `source .venv/bin/activate`.

```bash
pip install -r requirements-dev.txt
```

```bash
cp .env.example .env
```

Collect the data (~20 minutes on a first run; subsequent runs take seconds
thanks to the on-disk cache):

```bash
python scripts/ingest.py
```

Train every model and keep the best of each:

```bash
python scripts/train_models.py
```

Check that everything works (51 checks, ~90 seconds):

```bash
python scripts/verify.py
```

Launch the dashboard:

```bash
streamlit run streamlit_app.py
```

Optionally, start the REST API on http://localhost:8000/docs:

```bash
python scripts/run_api.py
```

### Faster first run

Skip ball-by-ball ingestion (~3 minutes instead of ~20; the chase model is then
unavailable):

```bash
python scripts/ingest.py --no-deliveries && python scripts/train_models.py --skip-chase
```

---

## Data sources

**The official IPL website is the primary source.** iplt20.com is a JavaScript
application that renders itself from a set of public JSON feeds; this project
consumes those feeds directly rather than scraping the rendered HTML, which is
both more reliable and far lighter on the site.

| Source | Seasons | What it provides |
|---|---|---|
| **iplt20.com official feeds** | **2019–2026** (8 seasons, 550 matches) | Schedule, results, toss, venue, innings scores, margins, Player of the Match, batting and bowling scorecards, partnerships, fall of wickets, **ball-by-ball commentary**, Playing XIs, umpires |
| **Cricsheet** (supplement) | 2008–2018 (11 seasons, 696 matches) | Ball-by-ball data for the seasons the official feed does not index |

Endpoints used (all public, all served to any visitor of iplt20.com):

```
{competition}-matchschedule.js   every fixture in a season
{match}-Innings{n}.js            scorecards, partnerships, FoW, over-by-over
{match}-matchsummary.js          result, Player of the Match, target, DLS
{match}-squad.js                 both Playing XIs
```

Cricsheet fills only the gap. Where the official feed publishes a season, it wins
outright — the supplement never overwrites it. Every row records its `source`, so
provenance is queryable.

### Responsible collection

- One request at a time, with a configurable delay (default **0.6 s**).
- Exponential backoff with jitter on failure; a stale cache entry is preferred
  over hammering the origin.
- Descriptive `User-Agent` identifying the project.
- Responses cached on disk, so re-runs download nothing (a full 2024–2025
  re-ingest of 148 matches takes **29 seconds** from cache).
- Incremental by default: a match already stored *and* already final is skipped.

Cricsheet data is published under the
[Open Data Commons Attribution Licence](https://cricsheet.org/register/).

---

## Architecture

```
┌────────────────────┐     ┌──────────────────┐
│ iplt20.com feeds   │     │ Cricsheet        │
│ (primary, 2019+)   │     │ (2008–2018)      │
└─────────┬──────────┘     └────────┬─────────┘
          │                         │
          └───────────┬─────────────┘
                      ▼
        ┌───────────────────────────┐
        │  Ingestion pipeline       │
        │  fetch → parse → validate │
        │  → dedupe → upsert        │
        └────────────┬──────────────┘
                     ▼
        ┌───────────────────────────┐
        │  Warehouse (12 tables)    │
        │  Postgres / MySQL/ SQLite │
        └────────────┬──────────────┘
                     ▼
        ┌───────────────────────────┐
        │  Feature engineering      │
        │  strictly no leakage      │
        └────────────┬──────────────┘
                     ▼
        ┌───────────────────────────┐
        │  Model zoo (6 algorithms) │
        │  time-split evaluation    │
        └────────────┬──────────────┘
             ┌───────┴────────┐
             ▼                ▼
      ┌────────────┐   ┌─────────────┐
      │ Streamlit  │   │ FastAPI     │
      │ dashboard  │   │ REST API    │
      └────────────┘   └─────────────┘
```

Both serving layers call the same `PredictionService`, so a prediction is
computed exactly one way regardless of how it was requested.

---

## Project layout

```
ipl-analytics/
├── src/ipl/
│   ├── config.py               Settings, resolved from environment
│   ├── constants.py            Franchise/venue registries, feed IDs
│   ├── logging_utils.py        Console + rotating file logging
│   ├── cli.py                  ipl-ingest / ipl-train / ipl-eda / ipl-api
│   ├── ingestion/
│   │   ├── http_client.py      Rate-limited, retrying, cached HTTP
│   │   ├── iplt20_client.py    Official feed parser  (primary source)
│   │   ├── cricsheet_client.py Cricsheet parser      (2008–2018)
│   │   ├── normalize.py        Canonical names, free-text parsing
│   │   ├── records.py          Source-agnostic intermediate types
│   │   ├── validation.py       Consistency checks, deduplication
│   │   └── pipeline.py         Orchestration and loading
│   ├── db/
│   │   ├── base.py             Engine/session management
│   │   ├── models.py           12 SQLAlchemy ORM models
│   │   └── repository.py       Upserts and DataFrame read queries
│   ├── features/
│   │   ├── match_features.py   Pre-match features + rolling state
│   │   ├── inplay_features.py  Chase state, Player-of-Match ranking
│   │   └── preprocessing.py    Pipelines, time-based splitting
│   ├── analytics/
│   │   ├── team.py  player.py  venue.py  eda.py
│   ├── models/
│   │   ├── registry.py         The six algorithms
│   │   ├── train.py            Training for all four tasks
│   │   ├── evaluate.py         Metrics, selection, calibration
│   │   ├── predict.py          Inference service
│   │   ├── playoffs.py         Monte Carlo qualification
│   │   └── persistence.py      Artefact save/load
│   ├── dashboard/
│   │   ├── app.py  data.py  theme.py
│   │   └── views/              9 pages
│   └── api/
│       ├── main.py  schemas.py
├── scripts/                    CLI entry points
├── sql/                        Reference DDL (Postgres + MySQL)
├── tests/                      242 tests
├── docs/                       API, database, deployment, data sources
├── .github/workflows/          CI + scheduled data refresh
├── streamlit_app.py            Streamlit Cloud entry point
├── requirements.txt            runtime only (what a host installs)
└── requirements-dev.txt        + tests, DB drivers, tooling
```

---

## The models

### 1. Match winner — binary classification
Predicts whether the home/first-listed side wins. **38 numeric + 4 categorical
features**, all computed from matches that finished *strictly before* the fixture
being described: rolling form, career and venue win rates, head-to-head, rest
days, season points, and Playing XI strength built from each selected player's
career record. Training fixtures are mirrored so the model learns a symmetric
function of the two sides.

### 2. First-innings score — regression
Predicts the first-innings total. Includes **era-aware features** (a rolling
league-wide scoring level and a recent venue level) because T20 totals have risen
sharply over 19 seasons and a tree model cannot extrapolate a raw `season` value
past its training range.

### 3. Chase success — in-play classification
One training row per second-innings ball: target, score, wickets, balls left,
required rate, recent momentum. **45,096 ball-states from 1,170 chases.** Rows
are cut off once the chase is mathematically decided, so the label cannot leak.

### 4. Player of the Match — learning to rank
Framed as binary classification over every player-match row, then used as a
ranker: the highest-probability player in a match is the prediction. This keeps
the label space fixed as the player pool changes season to season. Reported as
**top-1 accuracy**, because plain accuracy on the binary task is misleading —
predicting "nobody" scores 96%.

### 5. Playoff qualification — Monte Carlo
Simulates the remainder of a season thousands of times, drawing each remaining
fixture from the winner model's probability, and reports the share of simulations
in which each side finished in the top four.

### How models are selected

| Task | Selection metric | Why |
|---|---|---|
| Winner, chase, Player of the Match | ROC-AUC | Threshold-independent, and the dashboard reports probabilities |
| First-innings score | RMSE | Penalises the large misses that matter most |

Log loss and Brier score are also recorded: a model can be accurate while being
badly calibrated, and a "78% confident" prediction should be right about 78% of
the time. The dashboard plots predicted-vs-observed rates so you can check.

---

## Dashboard

```bash
streamlit run streamlit_app.py
```

| Page | Contents |
|---|---|
| **Home** | Headline counts, latest results, titles, league-wide scoring/toss/chase trends |
| **Schedule & Results** | Fixture list, live league table with NRR, full scorecards |
| **Team Analytics** | Franchise profile, form timeline, leading performers, multi-team comparison |
| **Player Analytics** | Batting/bowling leaderboards, career profile, season trends, venue & opposition splits, player comparison |
| **Head to Head** | Complete record between two franchises by season, venue and meeting |
| **Venue Statistics** | Scoring profile, chase bias, franchise records, phase run rates |
| **Predictions** | Winner, score, chase simulator, Player of the Match, playoff race, upcoming fixtures |
| **Model Comparison** | Every algorithm's held-out scores, calibration curves, feature importance |
| **Admin** | Data refresh, retraining, health, effective configuration |

**Chart design.** Analytical series use a palette validated for colour-vision
deficiency (worst adjacent CVD ΔE 9.1 light / 8.4 dark, verified by script);
team-identity charts use franchise brand colours and always carry a legend or
direct labels, so colour is never the only channel carrying meaning. Every chart
is accompanied by its underlying table. No chart uses two y-axes.

---

## REST API

```bash
python scripts/run_api.py
```

Interactive documentation: **http://localhost:8000/docs**

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness, row counts, which models are trained |
| `POST` | `/predict/winner` | Match winner and win probabilities |
| `POST` | `/predict/score` | First-innings total with a range |
| `POST` | `/predict/chase` | In-play chase success probability |
| `GET` | `/predict/player-of-match/{match_id}` | Ranked award candidates |
| `GET` | `/predict/playoffs/{season}` | Monte Carlo qualification odds |
| `GET` | `/teams` `/venues` `/matches` `/head-to-head` | Analytics tables |
| `GET` | `/models/comparison` | Every algorithm's held-out metrics |

```bash
curl -X POST http://localhost:8000/predict/chase -H "Content-Type: application/json" -d '{"batting_team":"Mumbai Indians","bowling_team":"Chennai Super Kings","venue":"Wankhede Stadium","target":180,"current_runs":120,"wickets_fallen":3,"balls_bowled":78}'
```

```json
{ "runs_required": 60, "balls_remaining": 42, "required_run_rate": 8.57,
  "chase_success_probability": 74.03, "model": "Random Forest" }
```

Full reference: [`docs/API.md`](docs/API.md).

---

## Database

Twelve tables: `teams`, `venues`, `players`, `matches`, `innings`,
`batting_cards`, `bowling_cards`, `fall_of_wickets`, `partnerships`,
`deliveries`, `match_players`, `ingestion_runs`.

SQLite is the zero-config default. For PostgreSQL or MySQL, set
`IPL_DATABASE_URL` — the application creates the schema itself:

```bash
IPL_DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/ipl python scripts/init_db.py
```

Reference DDL lives in [`sql/schema_postgres.sql`](sql/schema_postgres.sql) and
[`sql/schema_mysql.sql`](sql/schema_mysql.sql). Full documentation, including
every column and the design rationale, is in
[`docs/DATABASE.md`](docs/DATABASE.md).

---

## Configuration

Every setting is an environment variable with a working default; see
[`.env.example`](.env.example).

| Variable | Default | Purpose |
|---|---|---|
| `IPL_DATABASE_URL` | `sqlite:///data/ipl.db` | Any SQLAlchemy URL |
| `IPL_REQUEST_DELAY` | `0.6` | Seconds between feed requests |
| `IPL_USE_HTTP_CACHE` | `true` | Cache feed payloads on disk |
| `IPL_ENABLE_CRICSHEET` | `true` | Back-fill 2008–2018 |
| `IPL_INGEST_DELIVERIES` | `true` | Ingest ball-by-ball data |
| `IPL_TEST_SEASON_FROM` | `2024` | First held-out season |
| `IPL_ADMIN_PASSWORD` | `change-me` | Gates the Admin page — **change this** |

---

## Testing

Two layers, because they catch different things.

### `scripts/verify.py` — end-to-end, against the real database

```bash
python scripts/verify.py
```

**51 checks in ~90 seconds.** This is the one to run before a demo or a
deployment. It verifies dependencies, database contents, trained artefacts,
every prediction path, **all nine dashboard pages rendered against the real
1,246-match database**, **every button on every page**, fourteen API endpoints
including their error paths, and the pytest suite. It prints one verdict and
exits non-zero on any failure.

This layer exists because the unit tests run against an *empty* database, so on
their own they only prove the "no data" path works. Three real crashes were
found only by rendering pages against real data — a chart label formatter that
raised on float counts, a duplicate widget key on the Admin page, and an Arrow
serialisation error on mixed-type tables.

### `pytest` — isolated correctness

```bash
pytest
```

**242 tests**, covering:

- **Parsing** — over notation (`19.4` overs is 118 balls, not 116), franchise
  alias folding, result and dismissal strings, placeholder sentinels
- **Feed quirks** — over-break sentinel rows are excluded; run totals are rebuilt
  from batter runs plus extras because `BallRuns` is the string `"W"` on a wicket
- **No leakage** — head-to-head counts at row *n* are asserted to equal exactly
  the meetings that happened before it
- **Cricket conventions** — a batting average is runs per *dismissal*; a batter
  never dismissed has an undefined average, not a huge one
- **Model machinery** — metric computation, deterministic tie-breaking in model
  selection, reproducible Monte Carlo
- **Dashboard** — all nine pages executed through Streamlit's `AppTest` harness

Skip the one test that touches the network:

```bash
pytest -m "not network"
```

---

## Deployment

### Streamlit Cloud

1. Push to GitHub.
2. Point Streamlit Cloud at `streamlit_app.py`.
3. Add `IPL_ADMIN_PASSWORD` (and optionally `IPL_DATABASE_URL`) under
   **Settings → Secrets**.

Streamlit Cloud has an ephemeral filesystem, so for persistent data point
`IPL_DATABASE_URL` at a hosted Postgres. For a read-only demo, commit a
pre-built SQLite file and pre-trained artefacts.

### Docker

```bash
docker compose up --build
```

Full guide, including the GitHub Actions refresh job:
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## Demonstrating the project

[`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md) is a presentation runbook: a
pre-flight command, a 10-minute walkthrough with what to say at each screen,
the questions an examiner is likely to ask (and honest answers), and a recovery
table for when something misbehaves live.

Always start with:

```bash
python scripts/verify.py
```

## Keeping data fresh

The pipeline is incremental: a match already stored **and** already final is
skipped, so a daily run costs one schedule request per active season plus detail
requests for whatever finished since.

```bash
python scripts/ingest.py --seasons 2026
```

A GitHub Actions workflow
([`.github/workflows/refresh-data.yml`](.github/workflows/refresh-data.yml))
runs this daily during the season and retrains when new matches land. The
dashboard's **Admin** page triggers the same pipeline from the browser.

When a new season starts, find its feed ID and add it to `IPL_COMPETITIONS`:

```bash
python scripts/discover_competitions.py --start 250 --end 450
```

---

## Limitations

Stated plainly, because knowing them is part of the work:

- **Pre-match prediction is near chance.** See [Results](#results-measured-not-claimed).
- **Player identity is not reconciled across sources.** Cricsheet writes
  `V Kohli`, the official feed writes `Virat Kohli`. Matching on surname plus
  initial would collide (`R Sharma` is both Rohit and Rahul), so the two are kept
  distinct. Player analytics are therefore era-scoped: 2008–2018 and 2019–2026
  are separate name spaces.
- **Playing XIs are missing for IPL 2021** — the official squad feed returns data
  for only 2 of 60 matches. Squad-strength features fall back to imputation there.
- **Boundaries in Cricsheet data are inferred** from runs off the bat (4 or 6),
  the standard convention; an all-run four is indistinguishable.
- **Net run rate in the playoff simulator** uses each team's *current* NRR to
  break points ties. Simulating future NRR would require simulating scores ball
  by ball.
- **The score model has slightly negative R²** on held-out seasons: it beats the
  training-era mean but not the test-era mean, because T20 scoring keeps rising.
  The era features narrow this gap; they do not close it.

---

## Licence & attribution

Released under the MIT Licence.

Match data is sourced from the **official IPL website**
([iplt20.com](https://www.iplt20.com/)) via its public feeds, supplemented by
[Cricsheet](https://cricsheet.org/) for 2008–2018. This is an independent
educational project and is not affiliated with, endorsed by, or connected to the
BCCI or the Indian Premier League. All trademarks belong to their owners.

Built as a B.Tech AI/ML portfolio project.
