#!/usr/bin/env python
"""Assign unanswered questions to job descriptions using retrieval + rerank."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from qe.assignment import aggregate_matches, build_matches, retrieve_candidates
from qe.clients.embedding import EmbeddingClient
from qe.clients.llm import SocleLLMClient
from qe.clients.qdrant import QdrantClient
from qe.clients.rerank import RerankClient
from qe.config import get_settings, require_api_key
from qe.documents import load_documents, read_document
from qe.llm_duties import LLMQuestionDutyExtractor

DEFAULT_INPUT_DIR = Path("data/qe_no_answers")
DEFAULT_COLLECTION = "job_descriptions"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_RERANK_URL = "https://albert.api.etalab.gouv.fr"
DEFAULT_RERANK_MODEL = "openweight-rerank"
DEFAULT_TOP_K = 20
DEFAULT_MAX_CHUNKS_PER_JOB = 5
DEFAULT_OUTPUT = Path("data/assignments.json")
DEFAULT_SUMMARY_OUTPUT = Path("data/assignments_summary.json")


@dataclass(frozen=True)
class AssignmentConfig:
    input_dir: Path
    collection: str
    qdrant_url: str
    embedding_model: str
    embeddings_url: str
    chat_completions_url: str
    llm_model: str
    rerank_url: str
    rerank_model: str
    top_k: int
    max_chunks_per_job: int
    output: Path
    summary_output: Path


def parse_args() -> AssignmentConfig:
    parser = argparse.ArgumentParser(
        description="Assign questions to job descriptions using Qdrant + rerank."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing unanswered question documents.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Qdrant collection containing job descriptions.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help="Qdrant base URL (e.g. http://localhost:6333).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Socle IA embedding model name.",
    )
    parser.add_argument(
        "--rerank-url",
        default=DEFAULT_RERANK_URL,
        help="Base URL for the Albert rerank API.",
    )
    parser.add_argument(
        "--rerank-model",
        default=DEFAULT_RERANK_MODEL,
        help="Albert rerank model name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of candidates to retrieve from Qdrant per query unit before reranking.",
    )
    parser.add_argument(
        "--max-chunks-per-job",
        type=int,
        default=DEFAULT_MAX_CHUNKS_PER_JOB,
        help=(
            "Maximum number of reranked chunks to include per job when summing scores. "
            "Rewards jobs that cover multiple aspects of the question while preventing "
            "volume from dominating. Default: 5."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the JSON file where assignments will be saved.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Path to the JSON file where aggregated assignment scores will be saved.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Model name for LLM question duty extraction.",
    )
    args = parser.parse_args()
    settings = get_settings()

    return AssignmentConfig(
        input_dir=args.input_dir,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        embedding_model=args.embedding_model or settings.embedding_model,
        embeddings_url=settings.embeddings_url,
        chat_completions_url=settings.chat_completions_url,
        llm_model=args.llm_model or settings.llm_model,
        rerank_url=args.rerank_url,
        rerank_model=args.rerank_model,
        top_k=args.top_k,
        max_chunks_per_job=args.max_chunks_per_job,
        output=args.output,
        summary_output=args.summary_output,
    )


def main() -> None:
    config = parse_args()

    socle_api_key = require_api_key("SOCLE_IA_API_KEY")
    rerank_api_key = require_api_key("ALBERT_API_KEY")

    embedder = EmbeddingClient(
        url=config.embeddings_url,
        model=config.embedding_model,
        api_key=socle_api_key,
    )
    qdrant = QdrantClient(config.qdrant_url)
    reranker = RerankClient(
        base_url=config.rerank_url,
        model=config.rerank_model,
        api_key=rerank_api_key,
    )

    assignments: list[dict] = []
    summary_assignments: dict[str, list[dict[str, float | str]]] = {}
    question_paths = list(load_documents(config.input_dir))
    if not question_paths:
        raise FileNotFoundError(
            f"No documents found in input directory '{config.input_dir}'."
        )

    for question_path in question_paths:
        question_text = read_document(question_path).strip()
        if not question_text:
            print(f"Skipping empty question file: {question_path}")
            continue

        # Extract duty units for retrieval diversity; stored for traceability.
        duty_units: list[str] = []
        if config.top_k > 0:
            chat_client = SocleLLMClient(
                url=config.chat_completions_url,
                model=config.llm_model,
                api_key=socle_api_key,
            )
            duty_units = LLMQuestionDutyExtractor(client=chat_client).request_duties(
                question_text
            )

        # Full question leads; duty units broaden recall.
        query_units = [question_text] + duty_units

        candidates = retrieve_candidates(
            query_units=query_units,
            embedder=embedder,
            qdrant=qdrant,
            collection=config.collection,
            top_k=config.top_k,
        )
        if not candidates:
            print(
                f"No candidates found in collection '{config.collection}'"
                f" for {question_path}"
            )
            continue

        matches = build_matches(
            candidates=candidates,
            reranker=reranker,
            query=question_text,
        )
        if not matches:
            print(f"No rerank results for {question_path}")
            continue

        kept_matches, score_by_user = aggregate_matches(
            matches,
            max_chunks_per_job=config.max_chunks_per_job,
        )

        assignments.append(
            {
                "question_file": question_path.name,
                "question_path": str(question_path),
                "question_text": question_text,
                "question_units": duty_units,
                "retrieval_candidates": len(candidates),
                "matches": kept_matches,
            }
        )

        question_key = question_path.stem.lower()
        summary_assignments[question_key] = [
            {"user": user, "cumulative_score": cumulative_score}
            for user, cumulative_score in sorted(
                score_by_user.items(), key=lambda item: item[1], reverse=True
            )
        ]

    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(
        json.dumps(assignments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config.summary_output.parent.mkdir(parents=True, exist_ok=True)
    config.summary_output.write_text(
        json.dumps(summary_assignments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Saved assignments for {len(assignments)} questions"
        f" to {config.output} and {config.summary_output}"
    )


if __name__ == "__main__":
    main()
