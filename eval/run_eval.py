"""Evaluate retrieval and answer quality against eval_set.json.

Usage (needs GEMINI_API_KEY and an ingested knowledge base):
    python -m eval.run_eval

Metrics
- Retrieval hit rate@k : expected source appears in the retrieved chunks
- Keyword recall       : share of expected keywords present in the answer
- Refusal accuracy     : out-of-scope questions get the "not found" reply
Grow eval_set.json with your own questions: 30-50 is a solid size.
"""
import json
import time
from pathlib import Path

from app import rag
from app.config import settings

rag.ingest_knowledge_dir()
cases = json.loads((Path(__file__).parent / "eval_set.json").read_text())

hits = kw_total = kw_found = 0
in_scope = refusals_ok = out_scope = 0
latencies = []

for case in cases:
    start = time.time()
    result = rag.answer(case["question"])
    latencies.append(time.time() - start)

    if case["expected_source"] is None:
        out_scope += 1
        refusals_ok += result["answer"].strip() == rag.NOT_FOUND
        continue

    in_scope += 1
    hits += any(s["source"] == case["expected_source"] for s in result["sources"])
    answer_lower = result["answer"].lower()
    for kw in case["expected_keywords"]:
        kw_total += 1
        kw_found += kw.lower() in answer_lower
    time.sleep(1)  # stay inside free-tier rate limits

print(f"Questions: {len(cases)} (in-scope {in_scope}, out-of-scope {out_scope})")
print(f"Retrieval hit rate@{settings.top_k}: {hits}/{in_scope} = {hits / max(in_scope, 1):.0%}")
print(f"Keyword recall: {kw_found}/{kw_total} = {kw_found / max(kw_total, 1):.0%}")
print(f"Refusal accuracy: {refusals_ok}/{out_scope}")
print(f"Avg latency: {sum(latencies) / len(latencies):.2f}s")
