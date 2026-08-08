# Deployment guide

Four routes, from simplest to most production-like.

---

## 1. Streamlit Community Cloud (free)

The quickest way to get a public URL.

### Steps

1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set **Main file path** to `streamlit_app.py`.
4. Under **Advanced settings → Secrets**, paste:

   ```toml
   IPL_ADMIN_PASSWORD = "a-strong-password"
   IPL_LOG_LEVEL = "INFO"
   ```

5. Deploy.

Streamlit exposes top-level secrets as environment variables, which is exactly
how this project reads its configuration — no code changes needed.

### The ephemeral-filesystem problem — already solved

Streamlit Cloud restarts containers freely and does **not** persist writes, so
the deployed app cannot run the ingest. This repository therefore **ships a
pre-built database and pre-trained models**, and the app finds them with no
configuration:

| Committed | Size | Purpose |
|---|---|---|
| `data/ipl_deploy.db` | 9 MB | All 1,246 matches, scorecards, partnerships and squads |
| `models/artifacts/*.joblib` | 6 MB | All four trained models |

When `IPL_DATABASE_URL` is unset and `data/ipl.db` is missing — exactly the
situation on a fresh deploy — `ipl.config` falls back to `data/ipl_deploy.db`
automatically.

**What the deployment copy leaves out.** The 280k-row `deliveries` table, which
is what takes the full database to 48 MB. Dropping it costs only the venue
"scoring by phase" chart. Every prediction still works, **including the chase
model** — a live chase prediction is computed from the match state the user
types in, not from stored deliveries.

Rebuild it after a data refresh:

```bash
python scripts/build_deploy_db.py
```

Then commit — `.gitignore` already whitelists both paths, so a plain `git add`
picks them up:

```bash
git add data/ipl_deploy.db models/artifacts && git commit -m "Refresh deployment data"
```

**Alternative — hosted Postgres.** For a genuinely live app that can refresh
itself, create a free database on [Neon](https://neon.tech),
[Supabase](https://supabase.com) or [Railway](https://railway.app), set
`IPL_DATABASE_URL` in Streamlit secrets, and populate it once from your machine:

```bash
IPL_DATABASE_URL="postgresql+psycopg2://..." python scripts/ingest.py
```

### Resource limits

Streamlit Cloud gives ~1 GB of RAM. The shipped setup stays well inside it, but
note:

- Models are loaded pre-trained. *Training* the chase model on 45k rows will not
  fit in-container, which is why the Admin page's retrain is not for use there.
- The Admin page's write actions are **disabled entirely** while
  `IPL_ADMIN_PASSWORD` is still `change-me`, so a public URL cannot be used to
  hammer the IPL feed. Set a real password in Secrets to enable them.

---

## 2. Docker

```bash
docker compose up --build
```

- Dashboard: http://localhost:8501
- API: http://localhost:8000/docs

The compose file starts PostgreSQL, the dashboard and the API together, with the
database on a named volume so data survives restarts.

First-time population:

```bash
docker compose exec dashboard python scripts/ingest.py
```

```bash
docker compose exec dashboard python scripts/train_models.py
```

### Production notes

- Change `POSTGRES_PASSWORD` and `IPL_ADMIN_PASSWORD` in `docker-compose.yml`,
  or move them to a `.env` file that compose reads.
- Put a reverse proxy (Caddy, nginx, Traefik) in front for TLS.
- Run the API with several workers:
  `uvicorn ipl.api.main:app --workers 4`.

---

## 3. Any VPS

```bash
git clone https://github.com/ujwal-m-2006/ipl-match-prediction.git && cd ipl-match-prediction
```

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt
```

```bash
cp .env.example .env   # then edit it
```

```bash
python scripts/ingest.py && python scripts/train_models.py
```

### systemd units

`/etc/systemd/system/ipl-dashboard.service`:

```ini
[Unit]
Description=IPL Analytics dashboard
After=network.target

[Service]
Type=simple
User=ipl
WorkingDirectory=/opt/ipl-analytics
Environment="PATH=/opt/ipl-analytics/.venv/bin"
EnvironmentFile=/opt/ipl-analytics/.env
ExecStart=/opt/ipl-analytics/.venv/bin/streamlit run streamlit_app.py \
          --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/ipl-api.service`:

```ini
[Unit]
Description=IPL Analytics API
After=network.target

[Service]
Type=simple
User=ipl
WorkingDirectory=/opt/ipl-analytics
EnvironmentFile=/opt/ipl-analytics/.env
ExecStart=/opt/ipl-analytics/.venv/bin/uvicorn ipl.api.main:app \
          --host 0.0.0.0 --port 8000 --workers 2
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now ipl-dashboard ipl-api
```

### nginx reverse proxy

```nginx
server {
    listen 80;
    server_name ipl.example.com;

    location / {
        proxy_pass         http://127.0.0.1:8501;
        proxy_http_version 1.1;
        # Streamlit needs WebSocket upgrade for its live connection.
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 86400;
    }

    location /api/ {
        proxy_pass       http://127.0.0.1:8000/;
        proxy_set_header Host $host;
    }
}
```

Then `sudo certbot --nginx -d ipl.example.com` for TLS.

---

## 4. Scheduled data refresh

### GitHub Actions (included)

`.github/workflows/refresh-data.yml` runs daily during the IPL window, ingests
whatever finished, retrains and commits the updated artefacts. Configure a
`IPL_DATABASE_URL` repository secret if you are using hosted Postgres.

Trigger it manually from the **Actions** tab at any time.

### cron

```cron
# Ingest new results every day at 02:00, then retrain on Sundays.
0 2 * * * cd /opt/ipl-analytics && .venv/bin/python scripts/ingest.py >> logs/cron.log 2>&1
0 3 * * 0 cd /opt/ipl-analytics && .venv/bin/python scripts/train_models.py >> logs/cron.log 2>&1
```

A daily run is cheap: matches already stored and already final are skipped, so
only genuinely new fixtures are fetched.

---

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `IPL_DATABASE_URL` | `sqlite:///data/ipl.db` | Any SQLAlchemy URL |
| `IPL_ADMIN_PASSWORD` | `change-me` | **Change before exposing publicly** |
| `IPL_REQUEST_DELAY` | `0.6` | Do not lower this |
| `IPL_USE_HTTP_CACHE` | `true` | |
| `IPL_INGEST_DELIVERIES` | `true` | `false` cuts the database from ~145 MB to ~14 MB |
| `IPL_ENABLE_CRICSHEET` | `true` | `false` restricts to 2019+ |
| `IPL_TEST_SEASON_FROM` | `2024` | First held-out season |
| `IPL_LOG_LEVEL` | `INFO` | |
| `IPL_API_HOST` / `IPL_API_PORT` | `0.0.0.0` / `8000` | |

---

## Pre-flight checklist

- [ ] `IPL_ADMIN_PASSWORD` changed from the default
- [ ] `IPL_DATABASE_URL` points at persistent storage
- [ ] `pytest` passes
- [ ] `python scripts/ingest.py` has populated the database
- [ ] `python scripts/train_models.py` has produced `models/artifacts/`
- [ ] `curl /health` returns `"status": "ok"` with all models `true`
- [ ] CORS narrowed in `src/ipl/api/main.py` if the API is public
- [ ] TLS terminated by a proxy
- [ ] A scheduled refresh is configured

---

## Troubleshooting

**"No matches in the database"** — run `python scripts/ingest.py`. Check that
the dashboard and the CLI share one `IPL_DATABASE_URL`; relative SQLite paths
resolve against the project root, so this should be automatic.

**"The model has not been trained"** — run `python scripts/train_models.py`, and
confirm `models/artifacts/` is present and readable in the deployed image.

**LightGBM or CatBoost missing** — they are optional. The trainer logs a warning
and compares the remaining algorithms. Wheels lag new Python releases; 3.11–3.13
are the safest targets.

**Dashboard shows stale data** — Streamlit caches for an hour. Use the Admin
page's refresh, which clears the cache, or restart the process.

**Ingestion is slow** — the first run fetches ~2,300 feed payloads at 0.6 s
apart. Subsequent runs read from cache. Use `--no-deliveries` for a much faster
load, or `--seasons 2026` to restrict scope.

**Feed returns connection resets** — you have lowered `IPL_REQUEST_DELAY` too
far or are running several ingests at once. Raise the delay and run one at a
time.
