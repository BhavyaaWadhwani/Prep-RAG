"""Turn uploaded files into (page, text) pairs."""
import io
from pathlib import Path

from pypdf import PdfReader


def load_document(filename: str, data: bytes) -> list[tuple[int | None, str]]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((i, text))
        return pages
    if ext in {".txt", ".md"}:
        return [(None, data.decode("utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type: {ext}")
