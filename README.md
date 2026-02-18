## QE Expert Assignment Toolkit

This repository ingests job descriptions into a Qdrant vector store and uses retrieval
augmented reranking to match unanswered questions with the people most likely to have the
right expertise.

### Key components

- `scripts/ingest_job_descriptions.py` — parses job description files, splits them into
  semantically meaningful chunks, embeds each chunk with Socle IA, and upserts the vectors
  into Qdrant.
- `scripts/assign_questions_to_expert.py` — embeds unanswered questions, retrieves the
  most relevant job chunks from Qdrant, reranks them with Albert, and outputs suggested
  expert matches.

### Local stack & database setup

```bash
# start Qdrant + Postgres locally
docker compose up qdrant postgres -d

# install deps (once)
poetry install

# run database migrations
poetry run alembic upgrade head
```

By default the Postgres service exposes `postgresql://qe:qe@localhost:5433/qe`. You can
override this by setting the standard `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, and
`PGDATABASE` environment variables.

- `scripts/ingest_job_descriptions.py` now stores the chunk cache and ingest manifest in
  Postgres tables managed via Alembic migrations.

### Environment variables

| Variable           | Required       | Used by                             | Purpose / default                                                  |
| ------------------ | -------------- | ----------------------------------- | ------------------------------------------------------------------ |
| `SOCLE_IA_API_KEY` | ✅              | ingestion, assignment, test scripts | Socle IA API key for embeddings + LLM chunking.                    |
| `ALBERT_API_KEY`   | ✅ (assignment) | assignment                          | Albert rerank API key.                                             |
| `LLM_BASE_URL`     | ✅ (ingestion)  | ingestion                           | Base URL for Socle IA API (embeddings + chat). No default in code. |
| `LLM_MODEL`        | ✅ (ingestion)  | ingestion                           | Model name for LLM chunking. No default in code.                   |
| `EMBEDDING_MODEL`  | ✅ (ingestion)  | ingestion                           | Embedding model name. No default in code.                          |
| `PGHOST`           | ❌              | ingestion                           | Postgres host (default: `localhost`).                              |
| `PGPORT`           | ❌              | ingestion                           | Postgres port (default: `5433`).                                   |
| `PGUSER`           | ❌              | ingestion                           | Postgres user (default: `qe`).                                     |
| `PGPASSWORD`       | ❌              | ingestion                           | Postgres password (default: `qe`).                                 |
| `PGDATABASE`       | ❌              | ingestion                           | Postgres database (default: `qe`).                                 |

### Chunk-aware ingestion

Recent improvements keep fine-grained context by chunking each job description before
embedding:

1. **Section detection**: headings (uppercase lines, numbered sections, or lines ending
   with `:`) start new sections. When no headings are present, the whole document becomes
   a single section.
2. **Size-controlled chunks**: sections are further split into ~1,800-character windows
   with a minimum chunk size of 350 characters and 200-character overlaps to avoid losing
   context.
3. **Rich metadata**: every chunk stores job title, section title, chunk indices, a short
   preview, and the document hash so downstream consumers can display precise snippets and
   keep only changed chunks up to date.
4. **Manifest-driven idempotence**: manifest rows in Postgres track the last ingested hash
   for each file. Unchanged files are skipped automatically, and removed files trigger
   deletion of their Qdrant points.

### Running the ingestion script
```bash
python scripts/ingest_job_descriptions.py \\
  --input-dir data/job_descriptions \\
  --collection job_descriptions \\
  --qdrant-url http://localhost:6333
```

Supported file types: `.txt`, `.pdf`, `.doc`, `.docx`.

The default chunking strategy uses the Socle chat completions endpoint at
`https://pliage-prod.socle-ia.data-ia.prod.atlas.fabrique.social.gouv.fr/api/v1/chat/completions`
to turn job descriptions into single-sentence responsibilities. Override this target with
`--llm-base-url` if you have another Socle cluster, or disable the LLM step entirely with
`--chunking-strategy heuristic`. A 405 response typically means the host you pointed to
doesn’t expose `/api/v1/chat/completions`—pass the correct base URL or fall back to the
heuristic chunker to keep ingestion running.
Supported file types: `.txt`, `.pdf`, `.doc`, `.docx`.

### Assigning questions to experts

```bash
export SOCLE_IA_API_KEY=...
export ALBERT_API_KEY=...
python scripts/assign_questions_to_expert.py \
  --input-dir data/qe_no_answers \
  --collection job_descriptions \
  --top-k 30 \
  --top-n 5 \
  --max-chunks-per-job 3
```

The assignment process now deduplicates rerank results per job so you see at most the
`--max-chunks-per-job` highest scoring chunks for each expert. The JSON output stored at
`data/assignments.json` includes chunk previews, section names, and rerank positions to
make manual review easier.

### Suggested workflow

1. Keep job descriptions up to date in `data/job_descriptions/` and re-run ingestion when
   files change.
2. Drop unanswered questions (one per file) into `data/qe_no_answers/`.
3. Run the assignment script and inspect `data/assignments.json` for the suggested experts
   and sections.
4. Iterate by reviewing chunk previews to confirm that the right expertise areas are being
   surfaced. Adjust chunk sizes or retrieval parameters as needed.

### Resetting Qdrant state

If you want to wipe the job-description collection and start fresh:

```bash
python scripts/reset_dbs.py \
  --collection job_descriptions \
  --qdrant-url http://localhost:6333 \
  --input-dir data/job_descriptions
```

This deletes the collection in Qdrant (ignoring the request if it already disappeared)
and removes the `.ingest_manifest.json` file so the next ingestion run reprocesses every
document from scratch.

### Troubleshooting

- Ensure Qdrant is reachable at the URL you pass in `--qdrant-url`.
- If embeddings fail, confirm the Socle IA API key has access to the selected model.
- If LLM chunking fails with `405 Method Not Allowed`, supply a working Socle chat base
  via `--llm-base-url` or rerun with `--chunking-strategy heuristic`.
- When reranking returns no results, double-check the Albert API key and model name.

Feel free to tailor chunk sizes, overlaps, or rerank limits via the respective CLI flags
to better match your dataset.
