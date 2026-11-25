"""Command-line interface for MCR-L."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .engine import MCRLanguageEngine
from .knowledge import KnowledgeStore
from .neural_engine import NeuralMCRLanguageEngine


def _print_answer(engine: MCRLanguageEngine, utterance: str, debug: bool = False) -> None:
    result = engine.answer(utterance)
    print("\nMCR-L:", result.text)
    mode = result.diagnostics.get("mode", "")
    if mode == "calculation":
        print("[capability=bounded-arithmetic | evaluated=True]")
    elif mode == "neural_dialogue":
        reasoning = "multi-pass" if result.diagnostics.get("deliberated") else "standard"
        print(f"[language=local-neural | reasoning={reasoning} | model={result.diagnostics.get('model', 'local')}]")
    elif mode == "neural_backend_error":
        print("[language=local-neural | status=backend-error | no-symbolic-fallback]")
    elif mode == "dialogue_composition":
        print(f"[meaning-check={result.confidence:.3f} | abstained={result.abstained}]")
    elif mode == "fact_composition":
        print(f"[source-confidence-metadata={result.confidence:.3f} | abstained={result.abstained}]")
    elif mode == "unknown_topic_or_property":
        print("[knowledge-match=none | abstained=True]")
    else:
        print(f"[status={'abstained' if result.abstained else 'answered'} | confidence-metadata={result.confidence:.3f}]")
    if result.selected_title:
        print(f"[basis={result.selected_title} | source={result.source}]")
    if debug:
        print(json.dumps(result.diagnostics, indent=2, ensure_ascii=False))


def interactive(engine: MCRLanguageEngine, debug: bool = False) -> None:
    if isinstance(engine, NeuralMCRLanguageEngine):
        mode = "local learned-language mode"
        if engine.deep_reasoning:
            mode += " with separate deliberation pass"
        print(f"MCR-L 0.8 · {mode}")
    else:
        print("MCR-L 0.8 · symbolic baseline (bounded language coverage)")
    print("Type /exit to quit. Use --debug to inspect the interpretation and response plan.")
    while True:
        try:
            utterance = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not utterance:
            continue
        if utterance.lower() in {"/exit", "/quit"}:
            break
        _print_answer(engine, utterance, debug=debug)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MCR-L: CPU-first language understanding with an optional local learned model"
    )
    parser.add_argument("query", nargs="*", help="message or question to submit")
    parser.add_argument("--db", default=None, help="SQLite knowledge database path")
    parser.add_argument("--interactive", action="store_true", help="start a conversation loop")
    parser.add_argument("--load-demo-documents", action="store_true", help="explicitly load the small project/computing demo document set")
    parser.add_argument("--import-jsonl", metavar="FILE", help="stream sourced document records into the database")
    parser.add_argument("--import-facts-jsonl", metavar="FILE", help="stream source-attributed subject-predicate-object facts into the database")
    parser.add_argument("--debug", action="store_true", help="print structured semantic and response diagnostics")
    parser.add_argument("--neural", action="store_true", help="use the configured local pretrained language model for natural language")
    parser.add_argument("--think", action="store_true", help="add a separate structured deliberation pass before writing the reply (requires --neural)")
    parser.add_argument("--model-url", default=os.environ.get("MCRL_MODEL_URL", "http://127.0.0.1:8080/v1"), help="OpenAI-compatible local API base URL")
    parser.add_argument("--model-name", default=os.environ.get("MCRL_MODEL_NAME", "gguf/Qwen3.5-2B-GGUF:Q4_K_M"), help="model identifier sent to the local server")
    parser.add_argument("--model-api-key", default=os.environ.get("MCRL_MODEL_API_KEY", "none"), help="local server API key, if required")
    parser.add_argument("--model-timeout", type=float, default=180.0, help="timeout in seconds for each model pass")
    args = parser.parse_args(argv)

    if args.think and not args.neural:
        parser.error("--think requires --neural")

    database_path = args.db or ":memory:"
    store = KnowledgeStore(database_path)
    try:
        seed = Path(__file__).resolve().parent / "data" / "seed.jsonl"
        if args.load_demo_documents and seed.exists():
            count = store.import_jsonl(str(seed))
            print(f"Loaded {count} optional demo documents.")

        imported_any = False
        if args.import_jsonl:
            count = store.import_jsonl(args.import_jsonl)
            print(f"Imported/updated {count} document records. Database now contains {store.count()} documents.")
            imported_any = True
        if args.import_facts_jsonl:
            count = store.import_facts_jsonl(args.import_facts_jsonl)
            print(f"Imported/updated {count} semantic facts. Fact store now contains {store.fact_count()} facts.")
            imported_any = True
        if imported_any and not args.query and not args.interactive:
            return 0

        if args.neural:
            engine: MCRLanguageEngine = NeuralMCRLanguageEngine(
                store,
                base_url=args.model_url,
                model=args.model_name,
                api_key=args.model_api_key,
                timeout=args.model_timeout,
                deep_reasoning=args.think,
            )
        else:
            engine = MCRLanguageEngine(store)

        if args.interactive:
            interactive(engine, debug=args.debug)
            return 0
        utterance = " ".join(args.query).strip()
        if not utterance:
            parser.print_help()
            return 0
        _print_answer(engine, utterance, debug=args.debug)
        return 0
    except (OSError, ValueError) as exc:
        print(f"MCR-L error: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
