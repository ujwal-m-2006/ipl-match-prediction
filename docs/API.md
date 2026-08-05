# REST API reference

The FastAPI service exposes every prediction and analytics table the dashboard
uses. Interactive documentation is generated from the code itself:

| URL | Contents |
|---|---|
| `/docs` | Swagger UI — try requests in the browser |
| `/redoc` | ReDoc — reference-style reading view |
| `/openapi.json` | Raw OpenAPI 3.1 schema |

## Running

```bash
python scripts/run_api.py
```

```bash
python scripts/run_api.py --port 8080 --reload
```

```bash
uvicorn ipl.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The service reads the same `.env` as the rest of the project. Models are loaded
lazily on first use and cached for the process lifetime.

## Conventions

- All request bodies are JSON; all responses are JSON.
- **Probabilities are returned as percentages (0–100)**, not fractions.
- Timestamps are ISO-8601. Dates are `YYYY-MM-DD`.
- Team and venue names must be the **canonical** spellings. Fetch valid values
  from `GET /teams` and `GET /venues`.

### Status codes

| Code | Meaning |
|---|---|
| `200` | Success |
| `404` | The requested match, season or fixture pair does not exist |
| `422` | Request body failed validation (see `detail` for the offending field) |
| `500` | Unexpected server error |
| `503` | The required model has not been trained — run `scripts/train_models.py` |

---

## System

### `GET /health`

Liveness and readiness. Use this as a container health check.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "database": "sqlite",
  "matches": 1246,
  "seasons": [2008, 2009, "…", 2026],
  "models_ready": { "winner": true, "score": true, "chase": true, "pom": true }
}
```

`status` is `"degraded"` when the warehouse holds no completed matches.

### `GET /system/last-run`

Summary of the most recent data-collection run: status, trigger, counts and
duration. Returns `404` if the pipeline has never run.

---

## Predictions

### `POST /predict/winner`

Predict the winner of a fixture.

| Field | Type | Required | Notes |
|---|---|---|---|
| `team1` | string | ✅ | Home / first-listed side |
| `team2` | string | ✅ | Must differ from `team1` |
| `venue` | string | | Defaults to `team1`'s home ground |
| `season` | integer | | Defaults to the latest stored season |
| `match_date` | date | | Used for rest-day features |
| `toss_winner` | string | | If already known |
| `toss_decision` | `bat` \| `field` | | |
| `is_playoff` | boolean | | Default `false` |
| `is_neutral_venue` | boolean | | Default `false` |

```bash
curl -X POST http://localhost:8000/predict/winner \
  -H "Content-Type: application/json" \
  -d '{"team1":"Chennai Super Kings","team2":"Mumbai Indians","venue":"MA Chidambaram Stadium"}'
```

```json
{
  "team1": "Chennai Super Kings",
  "team2": "Mumbai Indians",
  "venue": "MA Chidambaram Stadium",
  "predicted_winner": "Mumbai Indians",
  "team1_win_probability": 26.41,
  "team2_win_probability": 73.59,
  "confidence": 47.18,
  "model": "CatBoost",
  "drivers": [
    { "label": "Recent form (last 5)", "team1_value": 40.0, "team2_value": 40.0, "unit": "%" },
    { "label": "Record at this venue", "team1_value": 64.6, "team2_value": 47.7, "unit": "%" }
  ]
}
```

> ⚠️ **Interpretation.** This model scores ROC-AUC ≈ 0.50–0.55 on held-out
> seasons — barely better than chance. Pre-match T20 outcomes are close to
> genuinely unpredictable from team-level information. See the README for the
> supporting evidence. Do not build anything consequential on this endpoint.

### `POST /predict/score`

Predict the first-innings total.

| Field | Type | Required |
|---|---|---|
| `batting_team` | string | ✅ |
| `bowling_team` | string | ✅ |
| `venue` | string | |
| `season` | integer | |
| `is_playoff`, `is_neutral_venue`, `batting_won_toss` | boolean | |

```json
{
  "batting_team": "Royal Challengers Bengaluru",
  "bowling_team": "Punjab Kings",
  "venue": "M Chinnaswamy Stadium",
  "predicted_score": 183,
  "range_low": 145,
  "range_high": 221,
  "model": "Random Forest"
}
```

`range_low`/`range_high` are ±1 standard deviation of the model's held-out
residuals, so roughly two thirds of real totals fall inside.

### `POST /predict/chase`

Predict whether a run chase succeeds from the current state. **This is the
strongest model in the system** (ROC-AUC 0.897).

| Field | Type | Constraints |
|---|---|---|
| `batting_team`, `bowling_team`, `venue` | string | required |
| `target` | integer | 1–400; first-innings total **+ 1** |
| `current_runs` | integer | 0–400 |
| `wickets_fallen` | integer | 0–10 |
| `balls_bowled` | integer | 0–120, legal deliveries only |
| `runs_last_5_overs`, `wickets_last_5_overs` | integer | optional momentum inputs |

```bash
curl -X POST http://localhost:8000/predict/chase \
  -H "Content-Type: application/json" \
  -d '{"batting_team":"Mumbai Indians","bowling_team":"Chennai Super Kings",
       "venue":"Wankhede Stadium","target":180,"current_runs":120,
       "wickets_fallen":3,"balls_bowled":78}'
```

```json
{
  "target": 180, "current_runs": 120, "wickets_fallen": 3, "balls_bowled": 78,
  "runs_required": 60, "balls_remaining": 42, "required_run_rate": 8.57,
  "chase_success_probability": 74.03,
  "win_probability_batting": 74.03,
  "win_probability_bowling": 25.97,
  "model": "Random Forest"
}
```

Terminal states are decided by the laws of cricket rather than the model: once
`current_runs >= target` the probability is 100%, and at ten wickets down or 120
balls bowled it is 0%.

### `GET /predict/player-of-match/{match_id}`

Rank a completed match's players by award probability.

| Parameter | In | Default | Notes |
|---|---|---|---|
| `match_id` | path | — | From `GET /matches` |
| `top_n` | query | `5` | 1–22 |

```json
{
  "match_id": 1872,
  "predicted": "Krunal Pandya",
  "actual": "Krunal Pandya",
  "candidates": [
    { "player": "Krunal Pandya", "team": "Royal Challengers Bengaluru",
      "runs": 0, "wickets": 4, "total_impact": 71.5, "award_probability": 34.2 }
  ]
}
```

Probabilities are normalised across the players in that match, so they sum to
100.

### `GET /predict/playoffs/{season}`

Monte Carlo projection of playoff qualification.

| Parameter | In | Default | Range |
|---|---|---|---|
| `season` | path | — | Any stored season |
| `simulations` | query | `5000` | 100–50,000 |

```json
{
  "season": 2026,
  "simulations": 5000,
  "matches_remaining": 18,
  "projections": [
    { "team": "Gujarat Titans", "current_points": 14, "net_run_rate": 0.612,
      "matches_played": 9, "matches_remaining": 5, "max_possible_points": 24,
      "qualification_pct": 88.4, "expected_position": 2.1 }
  ]
}
```

Each remaining fixture is drawn using the winner model's probability. When the
league stage is already complete, `matches_remaining` is 0 and the percentages
are 0 or 100.

---

## Analytics

### `GET /teams`

All-time record for every franchise. Query: `min_matches` (default 1).

```json
[{ "team": "Chennai Super Kings", "short_code": "CSK", "matches": 268,
   "wins": 149, "losses": 119, "win_pct": 55.6, "titles": 5 }]
```

### `GET /venues`

Scoring and result profile per ground. Query: `min_matches` (default 1).

```json
[{ "venue": "Wankhede Stadium", "city": "Mumbai", "matches": 118,
   "avg_first_innings": 172.4, "chase_win_pct": 54.2 }]
```

### `GET /matches`

Paginated fixture list.

| Parameter | Default | Notes |
|---|---|---|
| `season` | — | Filter to one season |
| `team` | — | Fixtures involving this franchise |
| `completed_only` | `false` | Exclude scheduled fixtures |
| `limit` | `100` | 1–1000 |
| `offset` | `0` | For pagination |

### `GET /head-to-head`

Aggregate record between two franchises. Query: `team_a`, `team_b` (both
required, must differ). Returns `404` if they have never met.

```json
{ "team_a": "Chennai Super Kings", "team_b": "Mumbai Indians",
  "matches": 39, "team_a_wins": 16, "team_b_wins": 23, "no_result": 0,
  "team_a_win_pct": 41.03, "team_b_win_pct": 58.97,
  "current_streak_team": "Mumbai Indians", "current_streak": 2 }
```

---

## Models

### `GET /models/comparison`

Held-out scores for every algorithm across every task. Returns `503` if nothing
has been trained.

```json
{
  "tasks": { "winner": "CatBoost", "score": "Random Forest",
             "chase": "Random Forest", "pom": "Random Forest" },
  "metrics": [
    { "task": "chase", "model": "Random Forest", "is_best": true,
      "accuracy": 0.7989, "precision": 0.873, "recall": 0.7215,
      "f1": 0.79, "roc_auc": 0.8974, "rmse": null, "mae": null, "r2": null }
  ]
}
```

Metrics that do not apply to a task are `null` (regression metrics on a
classifier, and vice versa).

---

## Errors

Errors use FastAPI's standard envelope:

```json
{ "detail": "The 'chase' model has not been trained. Run `python scripts/train_models.py` on the server." }
```

Validation failures list the offending field:

```json
{ "detail": [{ "type": "value_error", "loc": ["body", "team2"],
               "msg": "Value error, team1 and team2 must be different" }] }
```

---

## Notes for integrators

- **CORS** is open (`*`) for `GET` and `POST`. Restrict `allow_origins` in
  `src/ipl/api/main.py` before exposing this publicly.
- **No authentication.** Put it behind a gateway or reverse proxy if it is
  reachable from the internet.
- **No rate limiting.** Predictions are cheap (a single `predict_proba` on a
  loaded model), but `/predict/playoffs` with 50,000 simulations is not — cache
  it or lower the cap.
- **First request after startup** builds the rolling feature state by replaying
  the full match history, so it is slower than subsequent ones.
