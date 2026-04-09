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

## Compute question clusters

Embeds all questions into Qdrant, then clusters them by semantic similarity and saves the results to PostgreSQL.

```bash
# 1. Start Qdrant
docker compose up qdrant -d

# 2. Embed questions (all statuses, so the UI can filter later)
poetry run python scripts/embed_questions.py

# 3. Cluster and persist to DB
poetry run python scripts/cluster_questions.py
```

Results are stored in `question_cluster_runs` and `question_cluster_members`. Re-run step 3 at any time to refresh — each run creates a new set of rows.

## Assign questions to offices

Routes each QE to the most relevant office based on office responsibility descriptions.

### 1. Ingest office responsibilities

Place XLSX files in `data/office_responsibilities/`. Each file must have columns: `direction`, `office_id`, `office_name`, `responsibilities`, `keywords`.

```bash
poetry run python scripts/ingest_office_responsibilities.py
```

Re-run at any time — unchanged files are skipped automatically.

### 2. Assign a question

```bash
poetry run python scripts/assign_qe_to_office.py --question "Quel est le montant du RSA ?"
```

Returns a ranked JSON list of offices. Options:

```
--top-k 20        candidates retrieved per query unit (default: 20)
--top-offices 5   offices to return (default: 5)
--collection      Qdrant collection name (default: office_responsibilities)
```

### Evaluate assignment quality

Measures Hit@1/3/5 and MRR against a ground-truth XLSX file with columns `question_id`, `question_text`, `expected_office_id`:

```bash
poetry run python scripts/eval_office_assignment.py --input data/qe_attributions_DGCS.xlsx
```

Options:

```
--top-k 20          candidates retrieved per question (default: 20)
--top-offices 10    offices to rank per question (default: 10)
--chunks all        chunk types to search: all, responsibilities, keywords (default: all)
```

### Reset

```bash
poetry run python scripts/reset_dbs.py
```
