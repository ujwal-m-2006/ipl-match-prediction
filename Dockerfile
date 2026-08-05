# ---------------------------------------------------------------------------
# IPL Analytics - single image serving either the dashboard or the API.
#
#   docker build -t ipl-analytics .
#   docker run -p 8501:8501 ipl-analytics
#   docker run -p 8000:8000 ipl-analytics api
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Build tools are needed for a few wheels (LightGBM); libgomp is required at
# runtime by both LightGBM and XGBoost.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so a source change does not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY streamlit_app.py pyproject.toml README.md ./
COPY .streamlit/config.toml ./.streamlit/config.toml

RUN mkdir -p data/raw data/external models/artifacts logs reports

# Run as a non-root user.
RUN useradd --create-home --uid 1000 ipl && chown -R ipl:ipl /app
USER ipl

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

COPY --chown=ipl:ipl docker-entrypoint.sh /app/docker-entrypoint.sh
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["dashboard"]
