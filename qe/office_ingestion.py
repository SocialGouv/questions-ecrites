"""Ingestion logic for office responsibility XLSX files into Qdrant."""

from __future__ import annotations

from pathlib import Path

import openpyxl

from qe import db
from qe.chunking import Chunk
from qe.clients.embedding import EmbeddingClient
from qe.clients.qdrant import QdrantClient
from qe.hashing import compute_content_hash, make_preview, stable_chunk_id

OFFICE_COLLECTION = "office_responsibilities"


def parse_office_xlsx(path: Path) -> list[dict]:
    """Read an office responsibilities XLSX and return one dict per office row.

    Expected columns (in order): direction, office_id, office_name,
    responsibilities, keywords.

    Args:
        path: Path to the XLSX file.

    Returns:
        List of row dicts with keys ``direction``, ``office_id``,
        ``office_name``, ``responsibilities``, ``keywords``.
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue  # skip header
        direction, office_id, office_name, responsibilities, keywords = row
        if not office_id:
            continue
        rows.append(
            {
                "direction": str(direction or "").strip(),
                "office_id": str(office_id).strip(),
                "office_name": str(office_name or "").strip(),
                "responsibilities": str(responsibilities or "").strip(),
                "keywords": str(keywords or "").strip(),
            }
        )
    wb.close()
    return rows


def _office_rows_to_chunks(
    rows: list[dict],
) -> list[tuple[dict, int, Chunk]]:
    """Produce 2 chunks per office row: responsibilities text + keywords text.

    Returns a list of ``(row, row_index, chunk)`` triples, where each chunk
    carries ``chunk_type`` in its metadata.
    """
    result: list[tuple[dict, int, Chunk]] = []
    for row_index, row in enumerate(rows):
        responsibilities_text = f"{row['office_name']}\n{row['responsibilities']}"
        keywords_text = f"{row['office_name']}: {row['keywords']}"

        result.append(
            (
                row,
                row_index,
                Chunk(
                    title=row["office_name"],
                    text=responsibilities_text,
                    section_index=row_index,
                    chunk_index=0,
                    metadata={"chunk_type": "responsibilities"},
                ),
            )
        )
        result.append(
            (
                row,
                row_index,
                Chunk(
                    title=row["office_name"],
                    text=keywords_text,
                    section_index=row_index,
                    chunk_index=1,
                    metadata={"chunk_type": "keywords"},
                ),
            )
        )
    return result


def _build_office_chunk_payload(
    xlsx_path: Path,
    row: dict,
    chunk: Chunk,
    document_hash: str,
) -> dict:
    return {
        "kind": "office_chunk",
        "office_id": row["office_id"],
        "office_name": row["office_name"],
        "direction": row["direction"],
        "chunk_type": chunk.metadata["chunk_type"],
        "section_index": chunk.section_index,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "char_count": len(chunk.text),
        "chunk_preview": make_preview(chunk.text),
        "document_hash": document_hash,
        "source_file": str(xlsx_path.resolve()),
    }


def ingest_office_xlsx(
    *,
    xlsx_path: Path,
    collection: str,
    embedder: EmbeddingClient,
    qdrant: QdrantClient,
) -> None:
    """Ingest one office responsibilities XLSX file into a Qdrant collection.

    Incremental: skips the file if its content hash is unchanged in the
    manifest.  On change, deletes all existing points for this file before
    re-ingesting.  Creates the collection on first run.

    Args:
        xlsx_path: Path to the XLSX file.
        collection: Qdrant collection name.
        embedder: Client for generating dense embeddings.
        qdrant: Qdrant REST client.
    """
    raw_bytes = xlsx_path.read_bytes()
    document_hash = compute_content_hash(raw_bytes.hex())

    manifest = db.get_manifest_entries()
    path_key = str(xlsx_path.resolve())

    if manifest.get(path_key) == document_hash:
        print(f"Skipping unchanged file: {xlsx_path}")
        return

    rows = parse_office_xlsx(xlsx_path)
    if not rows:
        print(f"No office rows found in {xlsx_path}; skipping.")
        return

    collection_exists = qdrant.collection_exists(collection)

    # Delete stale points for this file before re-ingesting.
    if collection_exists:
        filter_payload = {
            "must": [{"key": "source_file", "match": {"value": path_key}}]
        }
        qdrant.delete_points_by_filter(collection, filter_payload)

    triples = _office_rows_to_chunks(rows)

    if not collection_exists:
        # Size the collection from the first chunk embedding.
        first_embedding = embedder.embed(triples[0][2].text)
        qdrant.create_collection(collection, vector_size=len(first_embedding))
        print(f"Created collection '{collection}' with size {len(first_embedding)}.")

    points: list[dict] = []
    for row, _row_index, chunk in triples:
        point_id = stable_chunk_id(xlsx_path, chunk.section_index, chunk.chunk_index)
        embedding = embedder.embed(chunk.text)
        payload = _build_office_chunk_payload(xlsx_path, row, chunk, document_hash)
        points.append({"id": point_id, "vector": embedding, "payload": payload})

    qdrant.upsert_points(collection, points)
    db.upsert_manifest(path_key, document_hash)
    print(f"Upserted {len(points)} chunks for '{xlsx_path.name}' into '{collection}'.")
