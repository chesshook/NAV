"""Command-line interface for the MCR-L semantic-composition prototype."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .engine import MCRLanguageEngine
from .knowledge import KnowledgeStore


def _print_answer(engine: MCRLanguageEngine, utterance: str, debug: bool = False) -> None:
    result = engine.answer(utterance)
    print("\nMCR-L:", result.text)
    print(f"[meaning-preservation/evidence-score={result.confidence:.3f} | abstained={result.abstained}]")
    if result.selected_title:
        print(f"[basis={result.selected_title} | source={result.source}]")
    if debug:
        print(json.dumps(result.diagnostics, indent=2, ensure_ascii=False))


def interactive(engine: MCRLanguageEngine, debug: bool = False) -> None:
    print("MCR-L 0.2 · CPU-first semantic-composition prototype")
    print("Type /exit to quit. Use --debug to inspect meaning graphs and sensitivity.")
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
    parser = argparse.ArgumentParser(description="MCR-L CPU-first semantic composition prototype")
    parser.add_argument("query", nargs="*", help="message or question to submit")
    parser.add_argument("--db", default=None, help="SQLite knowledge database path")
    parser.add_argument("--interactive", action="store_true", help="start a conversation loop")
    parser.add_argument("--import-jsonl", metavar="FILE", help="stream sourced JSONL records into the database")
    parser.add_argument("--debug", action="store_true", help="print meaning graph, sensitivity, and verification details")
    args = parser.parse_args(argv)

    database_path = args.db or ":memory:"
    store = KnowledgeStore(database_path)
    try:
        seed = Path(__file__).resolve().parent / "data" / "seed.jsonl"
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
