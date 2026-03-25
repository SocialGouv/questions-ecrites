# QE — Questions Écrites

Ingests French parliamentary written questions (Assemblée Nationale + Sénat) from DILA open data into a local PostgreSQL database.

## Installation

```bash
# Start Postgres locally
docker compose up postgres -d

# Install dependencies
poetry install

# Run database migrations
poetry run alembic upgrade head
```

The Postgres service is available at `postgresql://qe:qe@localhost:5433/qe` by default. Override with `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, and `PGDATABASE` environment variables.

## Download open data archives

Downloads `.taz` archives from the DILA open data server. Use `--years 2` to include the current year plus the 2 prior calendar years:

```bash
poetry run python scripts/download_opendata.py --dir data/opendata/ --years 2
```

Files already present are skipped automatically (idempotent).

## Ingest into PostgreSQL

Parses the downloaded archives and upserts questions into the `questions` table:

```bash
poetry run python scripts/ingest_opendata.py --dir data/opendata/
```

Add `--dry-run` to parse archives without writing to the database.
