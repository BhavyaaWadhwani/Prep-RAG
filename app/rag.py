"""RAG pipeline: ingest -> retrieve -> generate."""
import logging
from pathlib import Path

from . import embeddings, llm, vectorstore
from .chunking import split_text
from .config import SUPPORTED_EXTENSIONS, settings
from .loaders import load_document

log = logging.getLogger("placementprep")
NOT_FOUND = "I couldn't find this in the knowledge base."


def ingest_file(filename: str, data: bytes) -> int:
    """Parse, chunk, embed and store a document. Returns the chunk count."""
    filename = Path(filename).name
    chunks: list[dict] = []
    for page, text in load_document(filename, data):
        for piece in split_text(text, settings.chunk_size, settings.chunk_overlap):
            chunks.append({"text": piece, "page": page})
    if not chunks:
        raise ValueError("No readable text found in the document")
    vectors = embeddings.embed_texts([c["text"] for c in chunks])
    vectorstore.add_chunks(filename, chunks, vectors)
    return len(chunks)


def ingest_knowledge_dir() -> None:
    """Load bundled documents on startup (free hosts wipe disk on redeploy)."""
    folder = Path(settings.knowledge_dir)
    if not folder.exists() or not settings.gemini_api_key:
        return
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and not vectorstore.has_source(path.name):
            try:
                n = ingest_file(path.name, path.read_bytes())
                log.info("Ingested %s (%d chunks)", path.name, n)
            except Exception as exc:  # keep the app booting
                log.warning("Could not ingest %s: %s", path.name, exc)


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    hits = vectorstore.query(embeddings.embed_query(question), top_k or settings.top_k)
    return [h for h in hits if h["similarity"] >= settings.min_similarity]


def answer(question: str, top_k: int | None = None) -> dict:
    contexts = retrieve(question, top_k)
    if not contexts:  # nothing relevant: skip the LLM call entirely
        return {"answer": NOT_FOUND, "sources": []}
    text = llm.generate_answer(question, contexts)
    sources = [
        {
            "id": i,
            "source": c["source"],
            "page": c["page"],
            "similarity": c["similarity"],
            "snippet": c["text"][:300],
        }
        for i, c in enumerate(contexts, start=1)
    ]
    return {"answer": text, "sources": sources}
