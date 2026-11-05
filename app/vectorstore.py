"""ChromaDB persistence layer (cosine similarity)."""
import hashlib
from collections import Counter

import chromadb

from .config import settings

_collection = None


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=settings.chroma_dir)
        _collection = client.get_or_create_collection(
            name="placement_kb", metadata={"hnsw:space": "cosine"}
        )
    return _collection


def _chunk_id(source: str, index: int) -> str:
    return hashlib.sha1(f"{source}:{index}".encode()).hexdigest()


def has_source(source: str) -> bool:
    return bool(get_collection().get(where={"source": source}, limit=1)["ids"])


def add_chunks(source: str, chunks: list[dict], embeddings: list[list[float]]) -> None:
    """chunks: [{'text': str, 'page': int | None}]"""
    delete_source(source)  # re-uploading a file replaces it
    get_collection().add(
        ids=[_chunk_id(source, i) for i in range(len(chunks))],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"source": source, "page": c["page"] or 0, "chunk": i}
            for i, c in enumerate(chunks)
        ],
    )


def delete_source(source: str) -> None:
    get_collection().delete(where={"source": source})


def list_documents() -> list[dict]:
    metas = get_collection().get(include=["metadatas"])["metadatas"]
    counts = Counter(m["source"] for m in metas)
    return [{"name": name, "chunks": n} for name, n in sorted(counts.items())]


def query(embedding: list[float], k: int) -> list[dict]:
    coll = get_collection()
    if coll.count() == 0:
        return []
    res = coll.query(
        query_embeddings=[embedding],
        n_results=min(k, coll.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for text, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            {
                "text": text,
                "source": meta["source"],
                "page": meta["page"] or None,
                "similarity": round(1 - dist, 4),  # cosine distance -> similarity
            }
        )
    return hits
