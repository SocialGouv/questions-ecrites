# CLAUDE.md

## Project overview

Office assignment system for French parliamentary questions ("questions écrites"). Given unanswered parliamentary questions and responsibility descriptions of offices within French ministry directions, the system assigns each question to the most relevant office using semantic embeddings, vector search, and reranking.

**Pipeline:**

1. **Ingest** office responsibilities (XLSX) → chunk → embed → store in Qdrant
2. **Assign** questions → embed → search Qdrant → rerank → output JSON

## Project conventions

- **No `__init__.py` files.** This project uses implicit namespace packages (PEP 420). Do not create `__init__.py` files. Use direct imports to submodules (e.g. `from qe.clients.qdrant import QdrantClient`, not `from qe.clients import QdrantClient`).
- **Frozen dataclasses** for immutable config (e.g. `Settings`, `SocleLLMClient`).
- **Protocol classes** for plugin interfaces: `Chunker`, `DutyExtractor`.
- **Deterministic UUIDs** from SHA-256 hashes of file paths/content for idempotent Qdrant upserts.
- **Python >= 3.12**, managed with Poetry. Use `poetry run` to run scripts.

## Key directories and files

```bash
qe/                         # Main package (no __init__.py)
├── clients/
│   ├── embedding.py        # EmbeddingClient → Socle IA embeddings API
│   ├── llm.py              # SocleLLMClient → Socle IA chat completions
│   ├── qdrant.py           # QdrantClient → Qdrant vector DB (REST)
│   └── rerank.py           # RerankClient → Albert reranking API
├── assignment.py           # retrieve_candidates(), build_matches(), aggregate_matches(), match_question_to_offices()
├── chunking.py             # Chunk dataclass, Chunker protocol
├── config.py               # Settings dataclass, get_settings()
├── db.py                   # PostgreSQL: ingest_manifest + chunk_cache tables
├── documents.py            # load_documents(), read_document() (.txt/.pdf/.doc/.docx)
├── hashing.py              # stable_point_id(), stable_chunk_id(), compute_content_hash()
├── llm_duties.py           # DutyExtractor protocol
└── office_ingestion.py     # parse_office_xlsx(), ingest_office_xlsx()

scripts/
├── ingest_office_responsibilities.py  # Ingest office XLSX files into Qdrant
├── assign_qe_to_office.py             # Assign a question to the most relevant office
├── eval_office_assignment.py          # Evaluate assignment quality against a ground-truth XLSX
└── reset_dbs.py                       # Reset Qdrant collection + PostgreSQL state

data/
├── office_responsibilities/  # Input: office responsibility XLSX files
└── qe_no_answers/            # Input: questions to assign
```

## External services

| Service        | Purpose                | Config                                                         |
| -------------- | ---------------------- | -------------------------------------------------------------- |
| **Socle IA**   | Embeddings + LLM chat  | `LLM_BASE_URL`, `SOCLE_IA_API_KEY`, `LLM_MODEL`                |
| **Albert**     | Reranking              | `ALBERT_API_KEY`, default model `openweight-rerank`            |
| **Qdrant**     | Vector DB              | Local via docker-compose                                       |
| **PostgreSQL** | Ingest manifest        | `PGHOST/PORT/USER/PASSWORD/DATABASE`, local via docker-compose |

Default embedding model: `BAAI/bge-m3` (via `EMBEDDING_MODEL` env var).

## Database schema (Alembic)

- `ingest_manifest(path PK, document_hash, updated_at)` — tracks ingested files for incremental updates

Run migrations: `poetry run alembic upgrade head`

## Office chunking

Each office row in an XLSX produces 2 Qdrant chunks:

- `responsibilities` — `"{office_name}\n{responsibilities}"` — semantic coverage
- `keywords` — `"{office_name}: {keywords}"` — exact-term matching via embedding

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
