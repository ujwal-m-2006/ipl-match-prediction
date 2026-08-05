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

### The ephemeral-filesystem problem

Streamlit Cloud restarts containers freely and does **not** persist writes. A
SQLite file written by the Admin page will vanish. Two ways round it:

**Option A — hosted Postgres (recommended for a live app)**

Create a free database on [Neon](https://neon.tech),
[Supabase](https://supabase.com) or [Railway](https://railway.app), then add:

```toml
IPL_DATABASE_URL = "postgresql+psycopg2://user:pass@host:5432/ipl?sslmode=require"
```

Populate it once from your machine:

```bash
IPL_DATABASE_URL="postgresql+psycopg2://..." python scripts/ingest.py
```

**Option B — commit a pre-built database (simplest, read-only)**

Build locally, then commit the artefacts:

```bash
python scripts/ingest.py --no-deliveries && python scripts/train_models.py --skip-chase
```

`data/*.db` and `models/artifacts/` are git-ignored by default, so force-add
them:

```bash
git add -f data/ipl.db models/artifacts/ && git commit -m "Add prebuilt database and models"
```

Without deliveries this is ~14 MB of database plus a few MB of models — well
inside GitHub's comfortable range. The app then works out of the box, and the
Admin page's refresh simply won't survive a restart.

### Resource limits

Streamlit Cloud gives ~1 GB of RAM. Two consequences:

- Skip the chase model (`--skip-chase`) or ship it pre-trained; its artefact is
  ~3 MB and loads fine, but *training* it on 45k rows in-container will not.
- `load_deliveries()` on the full 280k-row table is heavy. The Venue page calls
  it; if you hit memory limits, ingest with `IPL_INGEST_DELIVERIES=false`.

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
git clone https://github.com/your-username/ipl-analytics.git && cd ipl-analytics
```

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
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
