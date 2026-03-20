#!/usr/bin/env python
r"""Find questions similar to a given input from the Qdrant questions collection.

Embeds the query and searches the Qdrant collection populated by
embed_questions.py (DB-sourced) or ingest_questions.py (file-sourced).
Results above the similarity threshold are returned, sorted by cosine
similarity (descending).

Query input (mutually exclusive):
  --file        Path to a question file (.docx, .pdf, .txt)
  --text        Raw question text
  --question-id Question ID as stored in PostgreSQL (e.g. AN-17-QE-12345)

Usage:
  # Search by file
  python scripts/find_similar_questions.py --file data/qe_no_answers/qe\ an\ 4487.docx

  # Search by raw text
  python scripts/find_similar_questions.py --text "Ma question porte sur les aides au logement..."

  # Search by DB question ID, find similar answered questions
  python scripts/find_similar_questions.py \
    --question-id AN-17-QE-12345 \
    --collection questions_opendata \
    --filter-status REPONDU \
    --threshold 0.70

Requires:
  - SOCLE_IA_API_KEY environment variable set
  - LLM_BASE_URL (or EMBEDDINGS_URL) environment variable set
  - A populated Qdrant collection (run embed_questions.py first)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from qe import db
from qe.clients.embedding import EmbeddingClient
from qe.clients.qdrant import QdrantClient
from qe.config import get_settings, require_api_key
from qe.documents import read_document
from qe.models import Question

DEFAULT_COLLECTION = "questions_opendata"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_TOP_K = 10
DEFAULT_THRESHOLD = 0.75


@dataclass(frozen=True)
class SearchConfig:
    question_text: str
    collection: str
    qdrant_url: str
    embedding_model: str
    embeddings_url: str
    api_key: str
    top_k: int
    threshold: float
    filter_status: str | None
    output: Path | None


def _lookup_question_text(question_id: str) -> str:
    """Fetch texte_question from PostgreSQL for the given question ID."""
    with db.get_session() as session:
        question = session.get(Question, question_id)
    if question is None:
        print(
            f"Error: question '{question_id}' not found in database.", file=sys.stderr
        )
        sys.exit(1)
    return question.texte_question


def parse_args() -> SearchConfig:
    parser = argparse.ArgumentParser(
        description="Find questions similar to a given input."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file",
        type=Path,
        help="Path to a question file (.docx, .pdf, .txt).",
    )
    input_group.add_argument(
        "--text",
        help="Question text to search for (as a string).",
    )
    input_group.add_argument(
        "--question-id",
        metavar="ID",
        help="Question ID from PostgreSQL (e.g. AN-17-QE-12345). Looks up texte_question from the DB.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help="Base URL for Qdrant (e.g. http://localhost:6333).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Socle IA embedding model name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of candidates to retrieve before threshold filtering.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum cosine similarity (0–1) to include in results (default: 0.75).",
    )
    parser.add_argument(
        "--filter-status",
        default=None,
        metavar="STATUS",
        help=(
            "Restrict search to questions with this etat_question "
            "(e.g. REPONDU, EN_COURS). Default: no filter."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON results (default: print to stdout).",
    )
    args = parser.parse_args()

    settings = get_settings()
    api_key = require_api_key("SOCLE_IA_API_KEY")

    if args.file:
        question_text = read_document(args.file).strip()
        if not question_text:
            print(f"Error: {args.file} is empty.", file=sys.stderr)
            sys.exit(1)
    elif args.question_id:
        question_text = _lookup_question_text(args.question_id).strip()
        if not question_text:
            print(f"Error: question '{args.question_id}' has no text.", file=sys.stderr)
            sys.exit(1)
    else:
        question_text = args.text.strip()
        if not question_text:
            print("Error: --text is empty.", file=sys.stderr)
            sys.exit(1)

    return SearchConfig(
        question_text=question_text,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        embedding_model=args.embedding_model or settings.embedding_model,
        embeddings_url=settings.embeddings_url,
        api_key=api_key,
        top_k=args.top_k,
        threshold=args.threshold,
        filter_status=args.filter_status,
        output=args.output,
    )


def find_similar(
    question_text: str,
    collection: str,
    embedder: EmbeddingClient,
    qdrant: QdrantClient,
    top_k: int,
    threshold: float,
    filter_status: str | None = None,
) -> list[dict]:
    """Embed the query and return similar questions above the threshold.

    Args:
        question_text: Raw text of the query question.
        collection: Qdrant collection to search.
        embedder: Client for generating dense embeddings.
        qdrant: Qdrant REST client.
        top_k: Number of nearest neighbours to retrieve from Qdrant.
        threshold: Minimum cosine similarity to include in output.
        filter_status: If set, restrict results to this etat_question value.

    Returns:
        List of result dicts sorted by similarity descending.
    """
    vector = embedder.embed(question_text)

    qdrant_filter = None
    if filter_status:
        qdrant_filter = {
            "must": [{"key": "etat_question", "match": {"value": filter_status}}]
        }

    candidates = qdrant.search(collection, vector, top_k, filter=qdrant_filter)

    results = []
    for candidate in candidates:
        score = candidate["score"]
        if score < threshold:
            continue
        payload = candidate.get("payload", {})
        results.append(
            {
                "question_id": payload.get("question_id", ""),
                "etat_question": payload.get("etat_question", ""),
                "source": payload.get("source", ""),
                "auteur_nom": payload.get("auteur_nom"),
                "ministre_attributaire_libelle": payload.get(
                    "ministre_attributaire_libelle"
                ),
                "date_publication_jo": payload.get("date_publication_jo"),
                "texte_preview": payload.get("texte_preview")
                or payload.get("question_preview", ""),
                "similarity": round(score, 6),
            }
        )

    return results


def main() -> None:
    config = parse_args()

    embedder = EmbeddingClient(
        url=config.embeddings_url,
        model=config.embedding_model,
        api_key=config.api_key,
    )
    qdrant = QdrantClient(config.qdrant_url)

    results = find_similar(
        question_text=config.question_text,
        collection=config.collection,
        embedder=embedder,
        qdrant=qdrant,
        top_k=config.top_k,
        threshold=config.threshold,
        filter_status=config.filter_status,
    )

    output_json = json.dumps(results, ensure_ascii=False, indent=2)

    if config.output:
        config.output.write_text(output_json, encoding="utf-8")
        print(f"Wrote {len(results)} result(s) to {config.output}.")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
