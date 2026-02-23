"""Core ingestion logic: embedding job descriptions and upserting into Qdrant."""

from __future__ import annotations

from pathlib import Path

from qe import db
from qe.chunking import Chunker, build_chunk_payload
from qe.clients.embedding import EmbeddingClient
from qe.clients.qdrant import QdrantClient
from qe.documents import load_documents, read_document
from qe.hashing import compute_content_hash, stable_chunk_id, stable_point_id


def delete_job_chunks(qdrant: QdrantClient, collection: str, job_id: str) -> None:
    """Delete all Qdrant points belonging to a single job description."""
    filter_payload = {"must": [{"key": "job_id", "match": {"value": job_id}}]}
    qdrant.delete_points_by_filter(collection, filter_payload)


def ingest_files(  # noqa: C901
    *,
    input_dir: Path,
    collection: str,
    embedder: EmbeddingClient,
    qdrant: QdrantClient,
    chunker: Chunker,
) -> None:
    """Ingest job description files from ``input_dir`` into a Qdrant collection.

    For each file:

    - Skips unchanged files by comparing content hashes against the manifest.
    - Deletes stale Qdrant points for removed or modified files before
      re-ingesting.
    - Chunks the document, embeds each chunk, and upserts the points.
    - Updates the manifest in PostgreSQL after a successful upsert.

    The Qdrant collection is created automatically on the first run, sized to
    match the embedding dimension of the first non-empty document.

    Args:
        input_dir: Directory containing job description files.
        collection: Qdrant collection name.
        embedder: Client for generating dense embeddings.
        qdrant: Qdrant REST client.
        chunker: Strategy used to split documents into chunks.
    """
    files = list(load_documents(input_dir))
    if not files:
        print(f"No supported files found in {input_dir}.")
        return

    # Find the first non-empty document to size the collection if needed.
    non_empty_text = None
    for file_path in files:
        text = read_document(file_path).strip()
        if text:
            non_empty_text = text
            break

    if not non_empty_text:
        print("No non-empty documents found to ingest.")
        return

    collection_exists = qdrant.collection_exists(collection)
    if not collection_exists:
        embedding = embedder.embed(non_empty_text)
        qdrant.create_collection(collection, vector_size=len(embedding))
        print(f"Created collection '{collection}' with size {len(embedding)}.")
        collection_exists = True

    manifest = db.get_manifest_entries()
    updated_manifest = dict(manifest)

    # Remove stale entries for files that no longer exist on disk.
    indexed_paths = {str(path) for path in files}
    removed_paths = [path_str for path_str in manifest if path_str not in indexed_paths]
    if removed_paths and collection_exists:
        for removed_path in removed_paths:
            job_id = stable_point_id(Path(removed_path))
            delete_job_chunks(qdrant, collection, job_id)
            updated_manifest.pop(removed_path, None)
            db.delete_manifest(removed_path)
            print(f"Removed stale entries for {removed_path}")

    for file_path in files:
        text = read_document(file_path).strip()
        if not text:
            print(f"Skipping empty file: {file_path}")
            continue

        document_hash = compute_content_hash(text)
        path_key = str(file_path)
        if manifest.get(path_key) == document_hash:
            print(f"Skipping unchanged file: {file_path}")
            continue

        job_id = stable_point_id(file_path)
        if collection_exists:
            delete_job_chunks(qdrant, collection, job_id)

        chunks = chunker.chunk(text=text)
        chunk_points: list[dict] = []
        chunk_counter = 0
        for chunk in chunks:
            chunk_id = stable_chunk_id(
                file_path, chunk.section_index, chunk.chunk_index
            )
            embedding = embedder.embed(chunk.text)
            payload = build_chunk_payload(
                path=file_path,
                job_id=job_id,
                chunk=chunk,
                document_hash=document_hash,
            )
            chunk_points.append(
                {"id": chunk_id, "vector": embedding, "payload": payload}
            )
            chunk_counter += 1

        if not chunk_points:
            print(f"No valid chunks produced for {file_path}; skipping.")
            continue

        qdrant.upsert_points(collection, chunk_points)
        updated_manifest[path_key] = document_hash
        db.upsert_manifest(path_key, document_hash)
        print(
            f"Upserted {chunk_counter} chunks for '{file_path.name}'"
            f" into '{collection}'."
        )
