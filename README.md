# QE — Questions Écrites

Assigns French parliamentary written questions to the most relevant ministry office. Questions are downloaded directly from the Assemblée Nationale and Sénat open-data portals, ingested into PostgreSQL, embedded into pgvector, then matched to office responsibility descriptions using semantic search and Albert reranking.

## Installation

```bash
docker compose up postgres -d
poetry install
poetry run alembic upgrade head
```

## Data ingestion

### Assemblée Nationale

```bash
# Download ZIP archives for legislatures XIV–XVII (--legislature 17 for one only)
poetry run python scripts/download_an_legacy.py --dir data/an_archives/

# Parse archives → PostgreSQL; auto-embeds newly ingested answers into pgvector
poetry run python scripts/ingest_an_legacy.py --dir data/an_archives/
```

Legislature XVII is a live archive — re-download periodically to pick up new questions and answers.

### Sénat

```bash
# Download full SQL dump covering all legislatures (--force to re-fetch)
poetry run python scripts/download_senat.py --dir data/senat/

# Parse dump → PostgreSQL (legislatures 14–17); auto-embeds answers into pgvector
poetry run python scripts/ingest_senat.py --file data/senat/questions.zip
```

## Embed questions

Reads questions from PostgreSQL and upserts embeddings into the `questions_opendata` pgvector table. Incremental — already-embedded questions are skipped.

```bash
poetry run python scripts/embed_questions.py
```

Filters (combinable): `--filter-status EN_COURS|REPONDU`, `--ministry TEXT`, `--source AN|SENAT`, `--legislature N`, `--date-from YYYY-MM-DD`, `--date-to YYYY-MM-DD`.

## Ingest office responsibilities

Place XLSX files in `data/office_responsibilities/` (columns: `direction`, `office_id`, `office_name`, `responsibilities`, `keywords`), then:

```bash
poetry run python scripts/ingest_office_responsibilities.py
```

Unchanged files are skipped automatically.

## Assign a question

```bash
poetry run python scripts/assign_qe_to_office.py --question "Quel est le montant du RSA ?"
```

Returns a ranked JSON list of offices. Options: `--top-k 20`, `--top-offices 5`.

## Evaluate assignment quality

Measures Hit@1/3/5 and MRR against a ground-truth XLSX (`question_id`, `question_text`, `expected_office_id`):

```bash
poetry run python scripts/eval_office_assignment.py --input data/qe_attributions_DGCS.xlsx
```

## Find similar questions

```bash
poetry run python scripts/find_similar_questions.py --question-id AN-17-QE-12345
poetry run python scripts/find_similar_questions.py --text "Ma question porte sur les aides au logement..."
poetry run python scripts/find_similar_questions.py --file data/qe_no_answers/qe.docx
```

Options: `--collection questions_opendata|answers_opendata`, `--filter-status REPONDU`, `--threshold 0.70`.

## API server

```bash
ALBERT_API_KEY=... poetry run uvicorn api.main:app --reload
```

### `GET /api/questions/{question_id}/attributions?top_k=3`

Returns the top-N office suggestions. The question's embedding is read from pgvector — no call to Socle IA is made.

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
      "relevance": 73.4
    }
  ]
}
```

`relevance` is a 0–100 score blending an absolute signal (sigmoid of the Albert reranker logit) and a relative signal (deviation from the pool median). High values indicate a strong match regardless of the other candidates.

### `GET /api/questions/{question_id}/similar?collection=answers&top_k=10`

Returns semantically similar items from another collection, reranked with Albert.

- `collection`: `questions`, `answers`, or `offices`
- `top_k`: 1–50 (default 10)
- `score_threshold`: optional minimum cosine similarity (0.0–1.0)

## Environment variables

| Variable               | Required       | Default                     | Description                         |
| ---------------------- | -------------- | --------------------------- | ----------------------------------- |
| `PLIAGE_API_KEY`       | Yes            | —                           | Socle IA API key (embeddings + LLM) |
| `LLM_BASE_URL`         | Yes            | —                           | Base URL for Socle IA services      |
| `LLM_MODEL`            | Yes            | —                           | LLM model name                      |
| `ALBERT_API_KEY`       | Yes (API only) | —                           | Albert reranking API key            |
| `EMBEDDING_MODEL`      | No             | `BAAI/bge-m3`               | Embedding model                     |
| `EMBEDDINGS_URL`       | No             | derived from `LLM_BASE_URL` | Override embeddings endpoint        |
| `CHAT_COMPLETIONS_URL` | No             | derived from `LLM_BASE_URL` | Override chat completions endpoint  |
| `CORS_ORIGINS`         | No             | `http://localhost:3000`     | Comma-separated allowed origins     |
| `PGHOST`               | No             | `localhost`                 | PostgreSQL host                     |
| `PGPORT`               | No             | `5433`                      | PostgreSQL port                     |
| `PGUSER`               | No             | `qe`                        | PostgreSQL user                     |
| `PGPASSWORD`           | No             | `qe`                        | PostgreSQL password                 |
| `PGDATABASE`           | No             | `qe`                        | PostgreSQL database                 |

## Interact with DB in Socle Data

```bash
kubectl --kubeconfig kubeconfig.yaml port-forward svc/questions-ecrites-db-rw -n questions-ecrites 5431:5432
```
