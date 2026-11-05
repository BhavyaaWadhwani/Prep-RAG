"""Recursive text chunking with overlap.

Splits on the largest natural boundary that fits (paragraph -> line ->
sentence -> word), packs pieces up to `chunk_size`, and carries a small tail of
the previous chunk into the next one so answers that straddle a boundary are
still retrievable.
"""
import re

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_units(text: str, size: int, level: int = 0) -> list[str]:
    """Break text into pieces that are each <= size characters."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    if level >= len(SEPARATORS):  # no separator left: hard slice
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep = SEPARATORS[level]
    units: list[str] = []
    for part in text.split(sep):
        part = part.strip()
        if not part:
            continue
        if sep == ". " and not part.endswith("."):
            part += "."
        units.extend(_split_units(part, size, level + 1))
    return units


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    units = _split_units(_normalize(text), chunk_size)
    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > chunk_size:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            space = tail.find(" ")  # start the overlap on a word boundary
            current = tail[space + 1 :] if space != -1 else tail
            if len(current) + 1 + len(unit) > chunk_size:
                current = ""
        current = f"{current} {unit}".strip() if current else unit
    if current.strip():
        chunks.append(current.strip())
    return chunks
