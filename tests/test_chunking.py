import pytest

from app.chunking import split_text


def test_short_text_is_single_chunk():
    assert split_text("Hello world.", 800, 100) == ["Hello world."]


def test_chunks_respect_size_limit():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = split_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_overlap_carries_context_forward():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = split_text(text, chunk_size=200, overlap=40)
    first_tail_word = chunks[0].split()[-1]
    assert first_tail_word in chunks[1]


def test_no_text_is_lost():
    text = " ".join(f"token{i}" for i in range(300))
    chunks = split_text(text, chunk_size=150, overlap=30)
    joined = " ".join(chunks)
    assert all(f"token{i}" in joined for i in range(300))


def test_long_unbroken_string_is_hard_split():
    chunks = split_text("x" * 2500, chunk_size=500, overlap=50)
    assert all(len(c) <= 500 for c in chunks)


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=100, overlap=100)
