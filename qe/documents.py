"""Document loading and reading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".doc", ".docx"}
IGNORED_PREFIXES = {"~$", "."}


def load_documents(folder: Path) -> Iterable[Path]:
    """Find all supported document files under *folder*."""
    if not folder.exists():
        raise FileNotFoundError(
            f"Input directory '{folder}' does not exist. Create it and add files."
        )
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(path.name.startswith(prefix) for prefix in IGNORED_PREFIXES)
    )


def read_document(path: Path) -> str:
    """Read a document and return its text content."""
    extension = path.suffix.lower()
    if extension == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if extension == ".pdf":
        return _read_pdf(path)
    if extension == ".docx":
        return _read_docx(path)
    if extension == ".doc":
        return _read_doc(path)
    raise ValueError(f"Unsupported file extension: {extension}")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _read_docx(path: Path) -> str:
    from docx import Document

    document = Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _read_doc(path: Path) -> str:
    import textract

    content = textract.process(str(path))
    return content.decode("utf-8", errors="ignore")
