# CLAUDE.md

## Project overview

Expert assignment system for French parliamentary questions ("questions écrites"). Given unanswered parliamentary questions and job descriptions of agents in French ministries, the system assigns each question to the most relevant expert using semantic embeddings, vector search, and reranking.

**Pipeline:**

1. **Ingest** job descriptions (PDF/DOCX) → chunk → embed → store in Qdrant
2. **Assign** questions → extract duty units via LLM → embed → search Qdrant → rerank → output JSON

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
├── chunking.py             # HeuristicChunker, LLMDutyChunker, ChunkCache
├── config.py               # Settings dataclass, get_settings()
├── db.py                   # PostgreSQL: ingest_manifest + chunk_cache tables
├── documents.py            # load_documents(), read_document() (.txt/.pdf/.doc/.docx)
├── hashing.py              # stable_point_id(), stable_chunk_id(), compute_content_hash()
└── llm_duties.py           # LLMJobDescriptionDutyExtractor, LLMQuestionDutyExtractor

scripts/
├── ingest_job_descriptions.py      # Main ingestion script
├── assign_questions_to_expert.py   # Main assignment script
├── reset_dbs.py                    # Reset Qdrant collection + PostgreSQL state
└── eval_assignments.py             # Evaluation metrics (Hit@1, Hit@3, MRR, etc.)

data/
├── job_descriptions/       # Input: job description files (PDF/DOCX)
├── qe_no_answers/          # Input: questions to assign
├── qe_with_answers/        # Input: questions with reference answers (for eval)
├── assignments.json        # Output: detailed assignments
├── assignments_summary.json # Output: aggregated scores per question
└── eval_results.xlsx       # Output: evaluation metrics
```

## External services

| Service        | Purpose                | Config                                                         |
| -------------- | ---------------------- | -------------------------------------------------------------- |
| **Socle IA**   | Embeddings + LLM chat  | `LLM_BASE_URL`, `SOCLE_IA_API_KEY`, `LLM_MODEL`                |
| **Albert**     | Reranking              | `ALBERT_API_KEY`, default model `openweight-rerank`            |
| **Qdrant**     | Vector DB              | Local via docker-compose                                       |
| **PostgreSQL** | Manifest + chunk cache | `PGHOST/PORT/USER/PASSWORD/DATABASE`, local via docker-compose |

Default embedding model: `BAAI/bge-m3` (via `EMBEDDING_MODEL` env var).

## Database schema (Alembic)

- `ingest_manifest(path PK, document_hash, updated_at)` — tracks ingested files for incremental updates
- `chunk_cache(strategy, document_hash PK, chunks JSON, updated_at)` — caches LLM-chunked results to avoid re-processing

Run migrations: `poetry run alembic upgrade head`

## Chunking strategies

- `heuristic` — splits by detected headings (uppercase lines, numbered sections), then by char limit (1800 chars, 200 overlap, min 350)
- `llm_duty` / `llm_responsibility` — LLM extracts 5–25 duties as semantic chunks, cached in PostgreSQL

## Evaluation metrics

Computed by `eval_assignments.py` against `data/attributions.json` (ground truth):

- **Hit@1, Hit@3** — whether the correct expert appears in top-1/3 results
- **MRR** — mean reciprocal rank
- **Score Share, Score Ratio** — rerank score fraction on correct experts
