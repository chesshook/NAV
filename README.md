# MCR-L
## Mathematical Cognitive Resonance for Language

MCR-L is an experimental, CPU-first, symbolic/structured language-reasoning prototype inspired by the recurrent situation model in SCENT-X MCR-2.

**It is not a trained foundation model and does not claim ChatGPT-level generality.** The first release tests a narrower hypothesis: a local system can parse a bounded set of intents, activate explicit semantic signals, retrieve indexed evidence, score structured answer plans, and realize answers with grammar templates—without loading a neural language model or requiring a GPU.

## Goals

- Run on ordinary CPU-only laptops with the Python standard library.
- Keep the language/knowledge store on disk and retrieve indexed records rather than load a multi-gigabyte corpus into RAM.
- Separate parsing, state estimation, evidence retrieval, answer planning, verification, and wording.
- Abstain when evidence is missing rather than invent facts.
- Make every score and signal inspectable.
- Provide a documented import path for larger JSONL corpora.

## Important limits

This architecture avoids neural-network inference and GPU requirements in this prototype; it does not eliminate computation. It uses explicit rules and indexed retrieval, so performance and coverage are bounded by the grammar, concepts, and knowledge supplied. A 2–3 GB text corpus is not automatically a knowledge base: it needs provenance, chunking, metadata, indexing, and quality control. The project does not claim that mathematical formulas alone solve open-domain language understanding.

## Quick start

Requires Python 3.10+; no third-party packages.

```bash
python -m mcrl "What is MCR-L?"
python -m mcrl --interactive
python -m pip install -e .  # optional: install the mcrl command
python -m unittest discover -s tests -v
```

On Windows, the same commands work in PowerShell or CMD when `python` is on PATH.

## Architecture

1. **Lexical parser** — normalizes text and extracts known terms, entities, and intent cues.
2. **Situation state** — updates a recurrent vector using a bounded sigmoid recurrence and an explicit signed relationship matrix.
3. **Retriever** — uses SQLite FTS5 when available; a deterministic in-memory fallback is included for the tiny starter knowledge base.
4. **Planner** — scores evidence-backed answer plans against intent, coverage, support, contradiction, and uncertainty.
5. **Verifier** — rejects plans below a confidence threshold or lacking support.
6. **Realizer** — fills controlled templates from the selected plan; it does not predict the next token.

## Add knowledge

Create UTF-8 JSON Lines records, one per line:

```json
{"id":"ram-001","title":"What is RAM?","text":"RAM is short-term working memory used by a computer while programs run.","answer":"RAM is the computer's short-term working memory. It holds data and instructions that active programs need quickly.","domain":"computing","source":"Your vetted reference","keywords":["RAM","memory"]}
```

Import them:

```bash
python -m mcrl --db data/knowledge.sqlite --import-jsonl path/to/records.jsonl
python -m mcrl --db data/knowledge.sqlite "What is RAM?"
```

The SQLite database is built locally and can be large without being fully loaded into memory. Do not ingest copyrighted books or personal data unless you have the rights and a lawful basis to do so. Store source and license metadata.

## Project status

MCR-L 0.1 is a deliberately small research baseline. It should be evaluated against a fixed question set before adding more mechanisms. See `docs/ROADMAP.md` and `docs/FORMULAS.md`.

## License

MIT. See `LICENSE`.
