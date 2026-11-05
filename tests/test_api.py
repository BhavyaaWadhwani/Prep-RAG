"""API tests with the Gemini calls replaced by deterministic fakes."""
import hashlib
import math

import pytest
from fastapi.testclient import TestClient

from app import embeddings, llm, rag
from app.main import app

DIM = 64


def fake_embed(text: str) -> list[float]:
    """Bag-of-words hashing embedding: similar wording => similar vectors."""
    vec = [0.0] * DIM
    for word in text.lower().split():
        word = word.strip(".,?!:;()[]")
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.fixture(autouse=True)
def patch_ai(monkeypatch):
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts, task_type="": [fake_embed(t) for t in texts])
    monkeypatch.setattr(embeddings, "embed_query", lambda q: fake_embed(q))
    monkeypatch.setattr(llm, "generate_answer", lambda q, ctx: f"Answer using {len(ctx)} passages [1]")
    monkeypatch.setattr(rag.settings.__class__, "min_similarity", 0.05, raising=False)


client = TestClient(app)
ADMIN = {"X-Admin-Key": "test-admin"}


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_upload_requires_admin_key():
    r = client.post("/api/documents", files={"file": ("a.txt", b"hello world", "text/plain")})
    assert r.status_code == 401


def test_rejects_unsupported_file_type():
    r = client.post("/api/documents", headers=ADMIN, files={"file": ("a.exe", b"x", "application/octet-stream")})
    assert r.status_code == 400


def test_upload_ask_and_delete_flow():
    text = b"Binary search works on sorted arrays. Two pointers help with pair sums.\n\nSQL joins combine tables."
    up = client.post("/api/documents", headers=ADMIN, files={"file": ("notes.md", text, "text/markdown")})
    assert up.status_code == 200 and up.json()["chunks"] >= 1

    assert any(d["name"] == "notes.md" for d in client.get("/api/documents").json())

    res = client.post("/api/ask", json={"question": "How does binary search work on sorted arrays?"})
    assert res.status_code == 200
    body = res.json()
    assert body["sources"] and body["sources"][0]["source"] == "notes.md"
    assert "[1]" in body["answer"]

    assert client.delete("/api/documents/notes.md", headers=ADMIN).status_code == 200
    assert not any(d["name"] == "notes.md" for d in client.get("/api/documents").json())


def test_empty_knowledge_base_returns_not_found():
    res = client.post("/api/ask", json={"question": "What is the meaning of life?"})
    assert res.status_code == 200
    assert res.json()["answer"] == rag.NOT_FOUND


def test_question_validation():
    assert client.post("/api/ask", json={"question": "a"}).status_code == 422
