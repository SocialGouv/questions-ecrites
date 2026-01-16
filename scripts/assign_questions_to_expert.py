#!/usr/bin/env python
"""Assign unanswered questions to job descriptions using retrieval + rerank."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import requests

from scripts.ingest_job_descriptions import (  # type: ignore
    EmbeddingClient,
    load_documents,
    read_document,
)

DEFAULT_INPUT_DIR = Path("data/qe_no_answers")
DEFAULT_COLLECTION = "job_descriptions"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_BASE_URL = (
    "https://pliage-prod.socle-ia.data-ia.prod.atlas.fabrique.social.gouv.fr"
)
DEFAULT_RERANK_URL = "https://albert.api.etalab.gouv.fr"
DEFAULT_RERANK_MODEL = "openweight-rerank"
DEFAULT_TOP_K = 20
DEFAULT_OUTPUT = Path("data/assignments.json")
DEFAULT_SUMMARY_OUTPUT = Path("data/assignments_summary.json")


@dataclass(frozen=True)
class AssignmentConfig:
    input_dir: Path
    collection: str
    qdrant_url: str
    embedding_model: str
    embedding_base_url: str
    rerank_url: str
    rerank_model: str
    top_k: int
    output: Path
    summary_output: Path


class QdrantSearchClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def search(
        self, collection: str, vector: Sequence[float], top_k: int
    ) -> list[dict]:
        payload = {
            "vector": vector,
            "top": top_k,
            "with_payload": True,
            "with_vectors": False,
        }
        response = requests.post(
            f"{self.base_url}/collections/{collection}/points/search",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("result", [])


class RerankClient:
    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> list[dict]:
        if not documents:
            return []
        payload = {
            "model": self.model,
            "query": query,
            "documents": list(documents),
            "top_n": top_n,
        }
        response = requests.post(
            f"{self.base_url}/v1/rerank",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data") or data.get("results") or []


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
        default=DEFAULT_EMBEDDING_MODEL,
        help="Socle IA embedding model name.",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=DEFAULT_EMBEDDING_BASE_URL,
        help="Base URL for the Socle IA embeddings API.",
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
        help="Number of candidates to retrieve from Qdrant before reranking.",
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
        help=(
            "Path to the JSON file where aggregated assignment scores will be saved."
        ),
    )
    args = parser.parse_args()
    return AssignmentConfig(
        input_dir=args.input_dir,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        rerank_url=args.rerank_url,
        rerank_model=args.rerank_model,
        top_k=args.top_k,
        output=args.output,
        summary_output=args.summary_output,
    )


def iter_question_documents(folder: Path) -> Iterable[Path]:
    yield from load_documents(folder)


def main() -> None:  # noqa: C901
    config = parse_args()

    socle_api_key = os.environ.get("SOCLE_IA_API_KEY")
    if not socle_api_key:
        raise ValueError("SOCLE_IA_API_KEY environment variable is not set")

    rerank_api_key = os.environ.get("ALBERT_API_KEY")
    if not rerank_api_key:
        raise ValueError("ALBERT_API_KEY environment variable is not set")

    embedder = EmbeddingClient(
        base_url=config.embedding_base_url,
        model=config.embedding_model,
        api_key=socle_api_key,
    )
    qdrant = QdrantSearchClient(config.qdrant_url)
    reranker = RerankClient(
        base_url=config.rerank_url,
        model=config.rerank_model,
        api_key=rerank_api_key,
    )

    assignments: list[dict] = []
    summary_assignments: dict[str, list[dict[str, float | str]]] = {}
    question_paths = list(iter_question_documents(config.input_dir))
    if not question_paths:
        raise FileNotFoundError(
            f"No documents found in input directory '{config.input_dir}'."
        )

    for question_path in question_paths:
        question_text = read_document(question_path).strip()
        if not question_text:
            print(f"Skipping empty question file: {question_path}")
            continue

        question_vector = embedder.embed(question_text)
        candidates = qdrant.search(config.collection, question_vector, config.top_k)
        if not candidates:
            print(
                f"No candidates found in collection '{config.collection}' for {question_path}"
            )
            continue

        candidate_texts: list[str] = []
        for candidate in candidates:
            payload = candidate.get("payload") or {}
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                candidate_texts.append("")
            else:
                candidate_texts.append(text)

        rerank_results = reranker.rerank(
            query=question_text,
            documents=candidate_texts,
            top_n=len(candidate_texts),
        )

        matches: list[dict] = []
        for rerank_position, result in enumerate(rerank_results, start=1):
            candidate_index = result.get("index")
            if candidate_index is None or candidate_index >= len(candidates):
                continue
            candidate = candidates[candidate_index]
            payload = candidate.get("payload") or {}

            job_id = payload.get("job_id") or payload.get("job_path")
            if not job_id:
                continue
            matches.append(
                {
                    "rank": len(matches) + 1,
                    "rerank_position": rerank_position,
                    "score": result.get("score") or result.get("relevance_score"),
                    "job_id": job_id,
                    "job_title": payload.get("job_title"),
                    "job_filename": payload.get("job_filename")
                    or payload.get("filename"),
                    "job_path": payload.get("job_path") or payload.get("path"),
                    "section_title": payload.get("section_title"),
                    "section_index": payload.get("section_index"),
                    "chunk_index": payload.get("chunk_index"),
                    "chunk_preview": payload.get("chunk_preview"),
                    # "chunk_text": payload.get("text"),
                }
            )

        assignments.append(
            {
                "question_file": question_path.name,
                "question_path": str(question_path),
                "question_text": question_text,
                "retrieval_candidates": len(candidates),
                "matches": matches,
            }
        )

        score_by_filename: dict[str, float] = {}
        for match in matches:
            job_filename = match.get("job_filename")
            score = match.get("score")
            if not job_filename or score is None:
                continue
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                continue
            score_by_filename[job_filename] = (
                score_by_filename.get(job_filename, 0.0) + numeric_score
            )

        summary_assignments[question_path.name] = [
            {
                "job_filename": job_filename,
                "cumulative_score": cumulative_score,
            }
            for job_filename, cumulative_score in sorted(
                score_by_filename.items(), key=lambda item: item[1], reverse=True
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
        "Saved assignments for"
        f" {len(assignments)} questions to {config.output}"
        f" and {config.summary_output}"
    )


if __name__ == "__main__":
    main()
