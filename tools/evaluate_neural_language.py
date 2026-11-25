"""Run the multi-turn learned-language evaluation set against a live local model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcrl.knowledge import KnowledgeStore
from mcrl.neural_engine import NeuralMCRLanguageEngine


def evaluate(path: Path, base_url: str, model: str, deep_reasoning: bool) -> int:
    total = 0
    speech_correct = 0
    topic_checks = 0
    topic_correct = 0
    reference_checks = 0
    reference_correct = 0
    knowledge_checks = 0
    knowledge_correct = 0

    store = KnowledgeStore(":memory:")
    try:
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            session = json.loads(raw)
            engine = NeuralMCRLanguageEngine(
                store,
                base_url=base_url,
                model=model,
                deep_reasoning=deep_reasoning,
            )
            print(f"\n[{session['session']}]")
            for turn_index, case in enumerate(session["turns"], 1):
                result = engine.answer(case["user"])
                total += 1
                frame = result.diagnostics.get("interpretation", {})
                speech = frame.get("speech_act")
                expected_speech = case.get("expected_speech_acts", [])
                speech_ok = speech in expected_speech
                speech_correct += int(speech_ok)

                topic_ok = None
                if "expected_topic_null" in case:
                    topic_checks += 1
                    topic_ok = (frame.get("topic") is None) == case["expected_topic_null"]
                    topic_correct += int(topic_ok)

                ref_ok = None
                if "expected_reference_any" in case:
                    reference_checks += 1
                    refs = set(frame.get("references", []))
                    ref_ok = bool(refs.intersection(case["expected_reference_any"]))
                    reference_correct += int(ref_ok)

                knowledge_ok = None
                if "expected_requires_knowledge" in case:
                    knowledge_checks += 1
                    knowledge_ok = (
                        frame.get("requires_knowledge") is case["expected_requires_knowledge"]
                    )
                    knowledge_correct += int(knowledge_ok)

                print(
                    f"{turn_index}. {case['user']}\n"
                    f"   speech_act={speech!r} expected={expected_speech!r} "
                    f"{'PASS' if speech_ok else 'FAIL'}\n"
                    f"   topic={frame.get('topic')!r} topic_check={topic_ok}\n"
                    f"   references={frame.get('references')!r} reference_check={ref_ok}\n"
                    f"   response={result.text}"
                )
    finally:
        store.close()

    print("\n=== evaluation summary ===")
    print(f"turns: {total}")
    print(f"speech-act accuracy: {speech_correct}/{total} = {speech_correct / total if total else 0:.3f}")
    if topic_checks:
        print(f"topic-null accuracy: {topic_correct}/{topic_checks} = {topic_correct / topic_checks:.3f}")
    if reference_checks:
        print(f"reference accuracy: {reference_correct}/{reference_checks} = {reference_correct / reference_checks:.3f}")
    if knowledge_checks:
        print(f"knowledge-need accuracy: {knowledge_correct}/{knowledge_checks} = {knowledge_correct / knowledge_checks:.3f}")
    print("This is a small sanity set, not a statistically sufficient benchmark.")
    return 0 if total and speech_correct == total else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "mcrl" / "data" / "evaluation" / "neural_dialogue_cases.jsonl")
    parser.add_argument("--url", default=os.environ.get("MCRL_MODEL_URL", "http://127.0.0.1:8080/v1"))
    parser.add_argument("--model", default=os.environ.get("MCRL_MODEL_NAME", "gguf/Qwen3.5-2B-GGUF:Q4_K_M"))
    parser.add_argument("--think", action="store_true", help="include a separate deliberation pass")
    args = parser.parse_args()
    return evaluate(args.data, args.url, args.model, args.think)


if __name__ == "__main__":
    raise SystemExit(main())
