# CLAUDE.md

## Project overview

Ingestion and semantic search system for French parliamentary questions ("questions écrites"). Questions and answers are ingested from the Assemblée Nationale and Sénat open-data portals, embedded, and made searchable.

**Pipeline:**

1. **Ingest** questions → parse → store in PostgreSQL → embed → store in pgvector (`vec_questions_opendata` table)
2. **Ingest** answers → parse → store in PostgreSQL → embed → store in pgvector (`vec_answers_opendata` table)
3. **Serve** semantic search via HTTP API (`api/main.py` — FastAPI, `GET /api/questions/{question_id}/similar`)

## Project conventions

- **No `__init__.py` files.** This project uses implicit namespace packages (PEP 420). Do not create `__init__.py` files. Use direct imports to submodules (e.g. `from qe.clients.pgvector_client import PgvectorClient`, not `from qe.clients import PgvectorClient`).
- **Frozen dataclasses** for immutable config (e.g. `Settings`).
- **Protocol classes** for plugin interfaces: `Chunker`, `VectorStore`.
- **Deterministic UUIDs** from SHA-256 hashes of file paths/content for idempotent vector upserts.
- **Python >= 3.12**, managed with Poetry. Use `poetry run` to run scripts.
- **Keep code DRY.** Before adding logic to a script, check whether it already exists in `qe/`. Shared helpers belong in the package, not copy-pasted across scripts.

## Key directories and files

```bash
api/
└── main.py                 # FastAPI server: GET /api/questions/{question_id}/similar

qe/                         # Main package (no __init__.py)
├── clients/
│   ├── embedding.py        # EmbeddingClient → Albert embeddings API
│   ├── pgvector_client.py  # PgvectorClient → pgvector-backed vector store
│   ├── vector_store.py     # VectorStore Protocol — backend-agnostic interface
│   └── rerank.py           # RerankClient → Albert reranking API
├── assignment.py           # retrieve_candidates(), rerank_candidates() — generic retrieval + reranking
├── chunking.py             # Chunk dataclass, Chunker protocol
├── config.py               # Settings dataclass, get_settings()
├── db.py                   # PostgreSQL: ingest_manifest + chunk_cache tables
├── documents.py            # load_documents(), read_document() (.txt/.pdf/.doc/.docx)
├── answer_embedding.py     # embed_answers() — embeds Reponse rows into vec_answers_opendata pgvector table
├── hashing.py              # stable_point_id(), stable_chunk_id(), stable_question_point_id(), stable_answer_point_id(), compute_content_hash()
└── models.py               # SQLAlchemy models: Question, Reponse, QuestionStateChange, …

scripts/
├── embed_questions.py                 # Embed questions from PostgreSQL into pgvector (vec_questions_opendata)
├── embed_answers.py                   # Embed answers from PostgreSQL into pgvector (vec_answers_opendata)
├── dump_qdrant.py                     # One-time: export Qdrant collections → JSONL
├── load_pgvector.py                   # One-time: import JSONL dump → pgvector tables
├── find_similar_questions.py          # CLI: semantically similar questions/answers
└── reset_dbs.py                       # Reset pgvector tables + PostgreSQL state

data/
└── qe_no_answers/            # Input files for find_similar_questions.py --file
```

## External services

| Service        | Purpose                         | Config                                                         |
| -------------- | ------------------------------- | -------------------------------------------------------------- |
| **Albert**     | Embeddings + Reranking           | `ALBERT_API_KEY`, `ALBERT_BASE_URL`, `ALBERT_EMBEDDING_MODEL`, `ALBERT_RERANK_MODEL` |
| **PostgreSQL** | Application data + vector store | `PGHOST/PORT/USER/PASSWORD/DATABASE`, local via docker-compose |

Default embedding model: `BAAI/bge-m3` (via `ALBERT_EMBEDDING_MODEL` env var).

PLIAGE (the OpenWebUI instance at `ia.social.gouv.fr`) is a separate provider
used only by qe-front's spelling-correction feature — it is never wired into
this repo.

## pgvector tables

| Table                         | Contents                                    | Populated by                                                                                           |
| ----------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `vec_questions_opendata`      | Embedded parliamentary questions            | `scripts/embed_questions.py`                                                                           |
| `vec_answers_opendata`        | Embedded parliamentary answers (Reponse)    | `scripts/embed_answers.py`, auto-called by `scripts/ingest_an.py` and `scripts/ingest_senat.py` |

Row IDs in all tables are deterministic UUID strings derived from SHA-256 hashes (see `qe/hashing.py`). Use `stable_question_point_id(question_id)` to resolve a question's vector row ID, and `stable_answer_point_id(reponse_id)` for answers.

## API server

```bash
ALBERT_API_KEY=... poetry run uvicorn api.main:app --reload
```

`GET /api/questions/{question_id}/similar?collection=questions|answers&top_k=10` — returns semantically similar questions or answers. The source question's embedding is fetched from `questions_opendata` (no embedding API call); results are reranked with Albert before being returned.

Configurable via env vars: `ALBERT_API_KEY` (required), `CORS_ORIGINS` (default `http://localhost:3000`).

## Database schema (Alembic)

- `ingest_manifest(path PK, document_hash, updated_at)` — tracks ingested files for incremental updates
- `vec_questions_opendata`, `vec_answers_opendata` — pgvector tables (1024-dim HNSW index)

Run migrations: `poetry run alembic upgrade head`

## Testing

Run the test suite with:

```bash
poetry run pytest
```

**Philosophy — test what matters, skip what doesn't:**

- **Favour real logic over mocks.** Write tests against actual code paths using
  real inputs (XML strings, Python objects, etc.).  Mocking is acceptable only
  when the alternative is standing up an external service (HTTP, database).
  Never mock just to avoid thinking about the input.

- **Do not chase 100% coverage.**  Cover the important logic and real edge
  cases.  Do not write tests that merely confirm that Python evaluates
  `True == True`, or that a third-party library works as documented.

- **Categorise tests by what they need:**
  - *Pure-logic tests* (no I/O): XML parsers, data transformations, string
    builders — test these directly, no fixtures or mocks required.
  - *Integration tests* (DB, HTTP): require a running PostgreSQL / WS
    endpoint.  Mark with `@pytest.mark.integration` and skip by default in CI
    unless the service is available.
  - *End-to-end tests*: run the full ingestion pipeline against a staging DB.

- **Use subclassing instead of `mock.patch` when testing client code.**
  Override only the transport method (`_post`, `_request`, …) in a local
  subclass so the rest of the client logic runs for real.  This is more
  readable and less brittle than patching module-level names.

- **No `pytest-mock` / `unittest.mock` for internal logic.**  Reserve the
  `mocker` fixture for cases where subclassing is impractical (e.g. patching
  `datetime.date.today()`).

- **Test files live in `tests/`**, named `test_<module>.py`.  Group tests with
  plain functions (not classes) unless shared fixtures make classes worthwhile.
