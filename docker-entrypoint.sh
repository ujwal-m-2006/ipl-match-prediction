#!/usr/bin/env bash
# Dispatch on the first argument so one image can serve every role.
set -euo pipefail

case "${1:-dashboard}" in
  dashboard)
    exec streamlit run streamlit_app.py \
      --server.port "${STREAMLIT_PORT:-8501}" \
      --server.address 0.0.0.0 \
      --server.headless true
    ;;

  api)
    exec uvicorn ipl.api.main:app \
      --host "${IPL_API_HOST:-0.0.0.0}" \
      --port "${IPL_API_PORT:-8000}" \
      --workers "${API_WORKERS:-2}"
    ;;

  ingest)
    shift
    exec python scripts/ingest.py "$@"
    ;;

  train)
    shift
    exec python scripts/train_models.py "$@"
    ;;

  *)
    # Anything else is run verbatim, so `docker run ... bash` works.
    exec "$@"
    ;;
esac
