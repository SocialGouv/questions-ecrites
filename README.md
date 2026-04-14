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

## Attribution API

Exposes office attribution suggestions over HTTP for the frontend (`qe-front`).

### Prerequisites

Both Qdrant collections must be populated before starting the server:

```bash
# 1. Office responsibilities
poetry run python scripts/ingest_office_responsibilities.py

# 2. Questions (embed into questions_opendata)
poetry run python scripts/embed_questions.py
```

### Start the server

```bash
poetry run uvicorn api.main:app --reload
```

The server starts on `http://localhost:8000` by default.

### `GET /api/questions/{question_id}/attributions`

Returns the top 3 office suggestions for a question. The question's embedding is read directly from Qdrant — no call to Socle IA is made.

```bash
curl http://localhost:8000/api/questions/AN-17-QE-12345/attributions
```

```json
{
  "question_id": "AN-17-QE-12345",
  "attributions": [
    {
      "rank": 1,
      "office_id": "...",
      "office_name": "Sous-direction des affaires sociales",
      "direction": "Direction générale du travail",
      "score": 1.8432,
      "confidence": 0.87
    }
  ]
}
```

`confidence` is a calibrated 0–1 value (sigmoid of the Albert cross-encoder logit). It is meaningful in absolute terms: values above ~0.7 indicate a strong match; values below ~0.3 indicate the question is likely outside this office's scope.

Optional query param: `top_k` (default `3`).

### Environment variables

| Variable        | Required | Default                  | Description                     |
| --------------- | -------- | ------------------------ | ------------------------------- |
| `ALBERT_API_KEY`| Yes      | —                        | Albert reranking API key        |
| `QDRANT_URL`    | No       | `http://localhost:6333`  | Qdrant base URL                 |
| `CORS_ORIGINS`  | No       | `http://localhost:3000`  | Comma-separated allowed origins |
