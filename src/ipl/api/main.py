"""FastAPI application.

Serves the analytics tables and every prediction the dashboard offers, so the
models are usable from outside Streamlit. Interactive documentation is at
``/docs`` (Swagger) and ``/redoc``; the raw OpenAPI schema is at
``/openapi.json``.

Run it with::

    python scripts/run_api.py           # or: uvicorn ipl.api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .. import __version__
from ..analytics.team import head_to_head, team_summary
from ..analytics.venue import venue_summary
from ..config import get_settings
from ..db import repository as repo
from ..logging_utils import get_logger, setup_logging
from ..models.persistence import ARTIFACTS, load_metrics
from ..models.playoffs import simulate_playoff_qualification
from ..models.predict import PredictionService, get_prediction_service
from .schemas import (
    ChaseRequest,
    ChaseResponse,
    HeadToHeadResponse,
    HealthResponse,
    IngestionRunResponse,
    MatchItem,
    ModelComparisonResponse,
    ModelMetricItem,
    PlayerOfMatchResponse,
    PlayoffResponse,
    ScoreRequest,
    ScoreResponse,
    TeamItem,
    VenueItem,
    WinnerRequest,
    WinnerResponse,
)

logger = get_logger(__name__)

DESCRIPTION = """
Machine-learning predictions and analytics for the Indian Premier League.

**Data sources.** Match data comes from the official
[iplt20.com](https://www.iplt20.com/) feeds, supplemented by
[Cricsheet](https://cricsheet.org/) for the 2008-2018 seasons the official feed
does not publish.

**Models.** Every prediction endpoint is backed by the best of six algorithms
(Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM and
CatBoost), selected on seasons held out of training. See `/models/comparison`
for the measured scores.

**Note on probabilities.** These are model outputs, not guarantees. Check the
calibration figures on `/models/comparison` before relying on them.
"""

TAGS = [
    {"name": "System", "description": "Health and pipeline status."},
    {"name": "Predictions", "description": "Match, score, chase and award forecasts."},
    {"name": "Analytics", "description": "Teams, venues, matches and head-to-head records."},
    {"name": "Models", "description": "Algorithm comparison and metrics."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    """Warm the prediction service at startup so the first request is fast."""
    setup_logging()
    logger.info("Starting IPL Analytics API v%s", __version__)
    try:
        service = get_prediction_service()
        ready = {task: service.has_model(task) for task in ARTIFACTS}
        logger.info("Model availability: %s", ready)
    except Exception as exc:  # pragma: no cover - never block startup
        logger.warning("Prediction service unavailable at startup: %s", exc)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="IPL Match Prediction & Analytics API",
    description=DESCRIPTION,
    version=__version__,
    openapi_tags=TAGS,
    lifespan=lifespan,
    contact={"name": "IPL Analytics", "url": "https://www.iplt20.com/"},
    license_info={"name": "MIT"},
)

# The dashboard and any browser client may call this API from another origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _service() -> PredictionService:
    """Return the shared prediction service."""
    return get_prediction_service()


def _require_model(service: PredictionService, task: str) -> None:
    """Raise a 503 when a model has not been trained yet."""
    if not service.has_model(task):
        raise HTTPException(
            status_code=503,
            detail=(
                f"The '{task}' model has not been trained. "
                "Run `python scripts/train_models.py` on the server."
            ),
        )


def _clean(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace pandas NaN with None so the JSON encoder accepts the payload."""
    return [
        {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()}
        for row in records
    ]


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Send browsers straight to the interactive docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Liveness and readiness, including which models are trained."""
    settings = get_settings()
    try:
        summary = repo.database_summary()
        seasons = repo.season_range()
        matches = summary.get("completed_matches", 0)
        season_list = list(range(seasons[0], seasons[1] + 1)) if seasons else []
    except Exception as exc:  # pragma: no cover - database missing
        logger.error("Health check could not read the database: %s", exc)
        matches, season_list = 0, []

    service = _service()
    models_ready = {task: service.has_model(task) for task in ARTIFACTS}

    return HealthResponse(
        status="ok" if matches > 0 else "degraded",
        version=__version__,
        database=settings.dialect,
        matches=matches,
        seasons=season_list,
        models_ready=models_ready,
    )


@app.get("/system/last-run", response_model=IngestionRunResponse, tags=["System"])
async def last_run() -> IngestionRunResponse:
    """Summary of the most recent data-collection run."""
    run = repo.latest_ingestion_run()
    if run is None:
        raise HTTPException(status_code=404, detail="No ingestion run has been recorded.")
    return IngestionRunResponse(**run)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
@app.post("/predict/winner", response_model=WinnerResponse, tags=["Predictions"])
async def predict_winner(request: WinnerRequest) -> WinnerResponse:
    """Predict the winner of a fixture and each side's win probability."""
    service = _service()
    _require_model(service, "winner")
    try:
        prediction = service.predict_winner(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Winner prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return WinnerResponse(**prediction.as_dict())


@app.post("/predict/score", response_model=ScoreResponse, tags=["Predictions"])
async def predict_score(request: ScoreRequest) -> ScoreResponse:
    """Predict the first-innings total, with a one-standard-deviation range."""
    service = _service()
    _require_model(service, "score")
    try:
        prediction = service.predict_first_innings_score(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Score prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return ScoreResponse(**prediction.as_dict())


@app.post("/predict/chase", response_model=ChaseResponse, tags=["Predictions"])
async def predict_chase(request: ChaseRequest) -> ChaseResponse:
    """Predict whether a run chase will succeed from the current match state."""
    service = _service()
    _require_model(service, "chase")
    try:
        prediction = service.predict_chase(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Chase prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return ChaseResponse(**prediction.as_dict())


@app.get(
    "/predict/player-of-match/{match_id}",
    response_model=PlayerOfMatchResponse,
    tags=["Predictions"],
)
async def predict_player_of_match(
    match_id: int,
    top_n: int = Query(default=5, ge=1, le=22, description="How many candidates to return."),
) -> PlayerOfMatchResponse:
    """Rank a completed match's players by Player-of-the-Match probability."""
    service = _service()
    _require_model(service, "pom")

    ranking = service.predict_player_of_match(match_id, top_n=top_n)
    if ranking.empty:
        raise HTTPException(
            status_code=404, detail=f"No player data for match {match_id}."
        )

    matches = repo.load_matches()
    row = matches[matches["match_id"] == match_id]
    actual = row.iloc[0]["player_of_match"] if not row.empty else None

    return PlayerOfMatchResponse(
        match_id=match_id,
        predicted=ranking.iloc[0]["player"],
        actual=actual if isinstance(actual, str) else None,
        candidates=_clean(ranking.to_dict("records")),
    )


@app.get("/predict/playoffs/{season}", response_model=PlayoffResponse, tags=["Predictions"])
async def predict_playoffs(
    season: int,
    simulations: int = Query(default=5000, ge=100, le=50000),
) -> PlayoffResponse:
    """Monte Carlo projection of each team's playoff-qualification chances."""
    service = _service()
    matches = repo.load_matches()
    if matches.empty or season not in set(matches["season"]):
        raise HTTPException(status_code=404, detail=f"No data for season {season}.")

    projection = simulate_playoff_qualification(
        matches,
        repo.load_innings(),
        season,
        service=service if service.has_model("winner") else None,
        simulations=simulations,
    )
    if projection.table.empty:
        raise HTTPException(
            status_code=404, detail=f"Not enough completed matches in {season} to project."
        )

    return PlayoffResponse(
        season=season,
        simulations=projection.simulations,
        matches_remaining=projection.matches_remaining,
        projections=_clean(projection.table.to_dict("records")),
    )


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
@app.get("/teams", response_model=list[TeamItem], tags=["Analytics"])
async def list_teams(
    min_matches: int = Query(default=1, ge=1, description="Minimum matches played."),
) -> list[TeamItem]:
    """All-time record for every franchise."""
    summary = team_summary(repo.load_matches(), min_matches=min_matches)
    if summary.empty:
        return []
    from ..constants import TEAM_CODES

    summary["short_code"] = summary["team"].map(TEAM_CODES)
    columns = ["team", "short_code", "matches", "wins", "losses", "win_pct", "titles"]
    return _clean(summary[columns].to_dict("records"))


@app.get("/venues", response_model=list[VenueItem], tags=["Analytics"])
async def list_venues(
    min_matches: int = Query(default=1, ge=1),
) -> list[VenueItem]:
    """Scoring and result profile for every ground."""
    summary = venue_summary(repo.load_matches(), repo.load_innings(), min_matches=min_matches)
    if summary.empty:
        return []
    columns = [c for c in ("venue", "city", "matches", "avg_first_innings", "chase_win_pct")
               if c in summary.columns]
    return _clean(summary[columns].to_dict("records"))


@app.get("/matches", response_model=list[MatchItem], tags=["Analytics"])
async def list_matches(
    season: int | None = Query(default=None, description="Filter to one season."),
    team: str | None = Query(default=None, description="Filter to one franchise."),
    completed_only: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> list[MatchItem]:
    """Paginated fixture list."""
    frame = repo.load_matches(completed_only=completed_only, season=season)
    if frame.empty:
        return []
    if team:
        frame = frame[(frame["team1"] == team) | (frame["team2"] == team)]

    frame = frame.sort_values("match_date", ascending=False)
    window = frame.iloc[offset : offset + limit]

    columns = [
        "match_id", "season", "match_date", "stage", "venue", "team1", "team2",
        "toss_winner", "toss_decision", "winner", "result_summary",
        "player_of_match", "is_completed",
    ]
    records = window[columns].copy()
    records["match_date"] = records["match_date"].dt.date
    return _clean(records.to_dict("records"))


@app.get("/head-to-head", response_model=HeadToHeadResponse, tags=["Analytics"])
async def get_head_to_head(
    team_a: str = Query(description="First franchise."),
    team_b: str = Query(description="Second franchise."),
) -> HeadToHeadResponse:
    """Aggregate record between two franchises."""
    if team_a == team_b:
        raise HTTPException(status_code=422, detail="team_a and team_b must differ.")

    record = head_to_head(repo.load_matches(), team_a, team_b)
    if record["matches"] == 0:
        raise HTTPException(
            status_code=404, detail=f"{team_a} and {team_b} have never met."
        )
    # Drop the DataFrame fields; only the scalar summary is serialisable here.
    scalars = {
        k: v for k, v in record.items()
        if not isinstance(v, pd.DataFrame) and k != "highest_total"
    }
    return HeadToHeadResponse(**scalars)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@app.get("/models/comparison", response_model=ModelComparisonResponse, tags=["Models"])
async def model_comparison() -> ModelComparisonResponse:
    """Held-out scores for every algorithm, across every task."""
    tasks: dict[str, str] = {}
    metrics: list[dict[str, Any]] = []

    for task, artifact in ARTIFACTS.items():
        payload = load_metrics(artifact)
        if not payload:
            continue
        best = payload.get("best_model")
        tasks[task] = best or "unknown"
        for row in payload.get("metrics") or []:
            metrics.append(
                {
                    "task": task,
                    "model": row.get("model"),
                    "is_best": row.get("model") == best,
                    "accuracy": row.get("accuracy"),
                    "precision": row.get("precision"),
                    "recall": row.get("recall"),
                    "f1": row.get("f1"),
                    "roc_auc": row.get("roc_auc"),
                    "rmse": row.get("rmse"),
                    "mae": row.get("mae"),
                    "r2": row.get("r2"),
                }
            )

    if not tasks:
        raise HTTPException(
            status_code=503,
            detail="No trained models found. Run `python scripts/train_models.py`.",
        )

    return ModelComparisonResponse(
        tasks=tasks,
        metrics=[ModelMetricItem(**_sanitise(m)) for m in metrics],
    )


def _sanitise(row: dict[str, Any]) -> dict[str, Any]:
    """Convert NaN metrics to None so they serialise as JSON null."""
    return {
        k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()
    }
