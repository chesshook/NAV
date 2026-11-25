"""Small repeatable latency/allocation benchmark for the seed knowledge base."""
from __future__ import annotations

import statistics
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcrl.engine import MCRLanguageEngine
from mcrl.knowledge import KnowledgeStore


QUERIES = [
    "What is MCR-L?",
    "Does MCR-L need a GPU?",
    "How should a large language database be stored?",
    "Why can a computer slow down when a game opens?",
    "What is a recurrent situation model?",
]


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[index]


def main() -> int:
    store = KnowledgeStore(":memory:")
    store.import_jsonl(str(ROOT / "data" / "seed.jsonl"))
    engine = MCRLanguageEngine(store)
    samples: list[float] = []
    tracemalloc.start()
    for _ in range(10):
        for query in QUERIES:
            started = time.perf_counter()
            engine.answer(query)
            samples.append((time.perf_counter() - started) * 1000.0)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print(f"Knowledge records: {store.count()}")
    print(f"Queries: {len(samples)}")
    print(f"Mean latency: {statistics.mean(samples):.3f} ms")
    print(f"Median latency: {statistics.median(samples):.3f} ms")
    print(f"P95 latency: {percentile(samples, 0.95):.3f} ms")
    print(f"Traced current/peak Python allocations: {current / 1024:.1f} / {peak / 1024:.1f} KiB")
    print("Note: tracemalloc is not total process RSS and timings are machine/database specific.")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
