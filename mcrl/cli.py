"""Command-line interface for the MCR-L prototype."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .engine import MCRLanguageEngine
from .knowledge import KnowledgeStore


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _print_answer(engine: MCRLanguageEngine, question: str, debug: bool = False) -> None:
    result = engine.answer(question)
    print("\nMCR-L:", result.text)
    print(f"[confidence={result.confidence:.3f} | abstained={result.abstained}]")
    if result.selected_title:
        print(f"[reference={result.selected_title} | source={result.source}]")
    if debug:
        print(json.dumps(result.diagnostics, indent=2, ensure_ascii=False))


def interactive(engine: MCRLanguageEngine, debug: bool = False) -> None:
    print("MCR-L 0.1 · CPU-first structured reasoning prototype")
    print("Enter /good or /bad to give explicit feedback; /exit to quit.")
    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not question:
            continue
        if question.lower() in {"/exit", "/quit"}:
            break
        if question.lower() == "/good":
            engine.observe_feedback(True)
            print("MCR-L: positive feedback recorded for the current recurrent links.")
            continue
        if question.lower() == "/bad":
            engine.observe_feedback(False)
            print("MCR-L: negative feedback recorded for the current recurrent links.")
            continue
        _print_answer(engine, question, debug=debug)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MCR-L CPU-first language reasoning prototype")
    parser.add_argument("query", nargs="*", help="question to ask")
    parser.add_argument("--db", default=None, help="SQLite knowledge database path")
    parser.add_argument("--interactive", action="store_true", help="start a conversation loop")
    parser.add_argument("--import-jsonl", metavar="FILE", help="stream JSONL knowledge records into the selected database")
    parser.add_argument("--debug", action="store_true", help="print feature activations and answer-plan diagnostics")
    args = parser.parse_args(argv)

    database_path = args.db or ":memory:"
    store = KnowledgeStore(database_path)
    try:
        seed = _project_root() / "data" / "seed.jsonl"
        if seed.exists():
            store.import_jsonl(str(seed))

        if args.import_jsonl:
            count = store.import_jsonl(args.import_jsonl)
            print(f"Imported/updated {count} records. Database now contains {store.count()} records.")
            if not args.query and not args.interactive:
                return 0

        engine = MCRLanguageEngine(store)
        if args.interactive:
            interactive(engine, debug=args.debug)
            return 0
        question = " ".join(args.query).strip()
        if not question:
            parser.print_help()
            return 0
        _print_answer(engine, question, debug=args.debug)
        return 0
    except (OSError, ValueError) as exc:
        print(f"MCR-L error: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
