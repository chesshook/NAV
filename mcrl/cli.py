"""Command-line interface for MCR-L."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from .engine import MCRLanguageEngine
from .knowledge import KnowledgeStore
from .neural_engine import NeuralMCRLanguageEngine
from .semantic_network import LexicalSemanticNetwork
from .mathematical_engine import MathematicalComprehensionEngine


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


def _print_understanding(
    engine: MathematicalComprehensionEngine,
    utterance: str,
    debug: bool = False,
) -> None:
    result = engine.understand(utterance)
    print("\nMCR-L meaning:", result.meaning["meaning_summary"])
    print(
        "[mode=mathematical-comprehension | "
        f"parse_complete={result.meaning['parse_complete']} | "
        f"unknown_words={len(result.meaning['unknown_lexical_items'])} | "
        f"ambiguous_words={len(result.meaning['ambiguous_lexical_items'])} | "
        "LLM_used=False]"
    )
    if debug:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


def interactive(
    engine: MCRLanguageEngine | MathematicalComprehensionEngine,
    debug: bool = False,
) -> None:
    if isinstance(engine, MathematicalComprehensionEngine):
        counts = engine.lexicon.counts()
        print(
            "MCR-L 1.0 · independent mathematical-comprehension mode\n"
            f"Lexicon: {counts['concepts']} concepts, {counts['lemmas']} word-sense links, "
            f"{counts['relations']} concept relations, {counts['sense_relations']} sense relations\n"
            "Syntax: sparse averaged-perceptron POS tagger + dependency parser\n"
            "No LLM is called. This mode interprets; it does not yet generate learned answers."
        )
    elif isinstance(engine, NeuralMCRLanguageEngine):
        mode = "local learned-language + semantic-network mode"
        if engine.semantic_network is not None:
            counts = engine.semantic_network.counts()
            mode += (
                f" ({counts['concepts']} concepts, {counts['relations']} concept relations, "
                f"{counts['sense_relations']} sense relations)"
            )
        if engine.deep_reasoning:
            mode += " with deliberation"
        print(f"MCR-L 0.9 · {mode}")
    else:
        print("MCR-L · symbolic baseline (bounded language coverage)")
    print("Type /exit to quit. Use --debug to inspect the full semantic structure.")
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
        if isinstance(engine, MathematicalComprehensionEngine):
            _print_understanding(engine, utterance, debug=debug)
        else:
            _print_answer(engine, utterance, debug=debug)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MCR-L: mathematical lexical semantics, compositional understanding, and language reasoning"
    )
    parser.add_argument("query", nargs="*", help="message or question to submit")
    parser.add_argument("--db", default=None, help="SQLite factual knowledge database path")
    parser.add_argument("--interactive", action="store_true", help="start a conversation loop")
    parser.add_argument("--load-demo-documents", action="store_true", help="explicitly load the small project/computing demo document set")
    parser.add_argument("--import-jsonl", metavar="FILE", help="stream sourced document records into the factual database")
    parser.add_argument("--import-facts-jsonl", metavar="FILE", help="stream source-attributed subject-predicate-object facts into the factual database")
    parser.add_argument("--debug", action="store_true", help="print structured semantic and response diagnostics")

    parser.add_argument("--semantic-inspect", action="store_true", help="run the lexical-semantic network only and print word/phrase activations")
    parser.add_argument("--lexicon-db", default="data/semantic_network.sqlite", help="SQLite lexical-semantic graph database")
    parser.add_argument("--import-wordnet-lmf", metavar="FILE", help="stream-import a Global WordNet LMF XML, XML.GZ, or ZIP dictionary")
    parser.add_argument("--import-semantic-jsonl", metavar="FILE", help="stream-import custom concept JSONL into the semantic network")
    parser.add_argument("--lexicon-source", default="User supplied semantic dictionary", help="dataset attribution label for a custom JSONL import")
    parser.add_argument("--lexicon-license", default="Unspecified; verify before redistribution", help="license/usage terms recorded with imported lexical data")
    parser.add_argument("--lexicon-source-url", default="", help="origin URL recorded with imported lexical data")

    parser.add_argument(
        "--mathematical", action="store_true",
        help="understand utterances using only the dictionary graph and locally trained mathematical syntax model; no LLM",
    )
    parser.add_argument(
        "--syntax-model", default="data/models/mcrl_syntax_v1.json",
        help="trained sparse POS/dependency model produced by tools/train_mathematical_core.py",
    )
    parser.add_argument("--neural", action="store_true", help="use the local pretrained language model with the mathematical semantic network")
    parser.add_argument("--think", action="store_true", help="add a separate structured deliberation pass before writing the reply (requires --neural)")
    parser.add_argument("--model-url", default=os.environ.get("MCRL_MODEL_URL", "http://127.0.0.1:8080/v1"), help="OpenAI-compatible local API base URL")
    parser.add_argument("--model-name", default=os.environ.get("MCRL_MODEL_NAME", "gguf/Qwen3.5-2B-GGUF:Q4_K_M"), help="model identifier sent to the local server")
    parser.add_argument("--model-api-key", default=os.environ.get("MCRL_MODEL_API_KEY", "none"), help="local server API key, if required")
    parser.add_argument("--model-timeout", type=float, default=180.0, help="timeout in seconds for each model pass")
    args = parser.parse_args(argv)

    if args.think and not args.neural:
        parser.error("--think requires --neural")
    if args.mathematical and args.neural:
        parser.error("--mathematical and --neural are separate modes; choose one.")
    if args.semantic_inspect and not " ".join(args.query).strip():
        parser.error("--semantic-inspect requires a query.")
    if args.mathematical and not args.interactive and not " ".join(args.query).strip():
        parser.error("--mathematical requires a query or --interactive.")
    if args.import_wordnet_lmf and args.import_semantic_jsonl:
        parser.error("Import one lexical dataset per command so attribution remains unambiguous.")

    store = KnowledgeStore(args.db or ":memory:")
    lexicon: LexicalSemanticNetwork | None = None
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

        needs_lexicon = bool(
            args.import_wordnet_lmf or args.import_semantic_jsonl
            or args.semantic_inspect or args.neural or args.mathematical
        )
        if needs_lexicon:
            lexicon = LexicalSemanticNetwork(args.lexicon_db)

        if args.import_wordnet_lmf:
            counts = lexicon.import_lmf(
                args.import_wordnet_lmf,
                dataset="Open English WordNet 2025+",
                license_name="CC BY 4.0 (preserve source attribution and notices)",
                source_url="https://github.com/globalwordnet/english-wordnet",
            )
            print("Imported Open English WordNet lexical graph:")
            print(json.dumps(counts, indent=2))
            imported_any = True

        if args.import_semantic_jsonl:
            counts = lexicon.import_jsonl(
                args.import_semantic_jsonl,
                dataset=args.lexicon_source,
                license_name=args.lexicon_license,
                source_url=args.lexicon_source_url,
            )
            print("Imported custom semantic dictionary:")
            print(json.dumps(counts, indent=2))
            imported_any = True

        if args.semantic_inspect:
            analysis = lexicon.analyze(" ".join(args.query))
            print(json.dumps(analysis.diagnostics(), indent=2, ensure_ascii=False))
            return 0

        if imported_any and not args.query and not args.interactive:
            return 0

        if args.mathematical:
            model_path = Path(args.syntax_model)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Mathematical syntax model not found: {model_path}. "
                    "Train it first with: py -3 tools/train_mathematical_core.py --download --epochs 5"
                )
            if lexicon.counts()["concepts"] == 0:
                raise ValueError(
                    f"No dictionary concepts found in {args.lexicon_db}. "
                    "Import Open English WordNet first with scripts/import-open-english-wordnet.cmd"
                )
            engine: MCRLanguageEngine | MathematicalComprehensionEngine = MathematicalComprehensionEngine(
                syntax_model_path=str(model_path),
                lexicon=lexicon,
            )
        elif args.neural:
            engine = NeuralMCRLanguageEngine(
                store,
                base_url=args.model_url,
                model=args.model_name,
                api_key=args.model_api_key,
                timeout=args.model_timeout,
                deep_reasoning=args.think,
                semantic_network=lexicon,
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
        if isinstance(engine, MathematicalComprehensionEngine):
            _print_understanding(engine, utterance, debug=args.debug)
        else:
            _print_answer(engine, utterance, debug=args.debug)
        return 0
    except (OSError, ValueError) as exc:
        print(f"MCR-L error: {exc}", file=sys.stderr)
        return 2
    finally:
        if lexicon is not None:
            lexicon.close()
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
