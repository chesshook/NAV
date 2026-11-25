# MCR-L roadmap

## 0.1 — CPU-first baseline (current)
- [x] Explicit 16-dimensional semantic feature parser.
- [x] Recurrent bounded activation state.
- [x] Inspectable signed connection matrix.
- [x] SQLite store with optional FTS5.
- [x] Streaming JSONL importer.
- [x] Evidence-aware answer-plan scoring and abstention.
- [x] Controlled answer realization from retrieved records.
- [x] Deterministic handling for basic greetings, thanks, farewells, and identity questions.
- [x] Unit tests and CPU benchmark harness.
- [ ] Persist per-conversation state and feedback weights.
- [ ] CI validation on Windows, Linux, and macOS.

## 0.2 — Better semantic coverage
- Add a controlled ontology and multiword concept aliases.
- Add morphology, spelling normalization, entity spans, and coreference.
- Add relationship extraction with explicit provenance.
- Add separate lexical retrieval and semantic-graph traversal.
- Add contradiction detection and source/date/confidence handling.
- Build a labelled evaluation set with paraphrases and adversarial negatives.

## 0.3 — Compositional reasoning
- Represent facts as typed relations rather than only prose passages.
- Add deterministic rule execution for arithmetic, units, dates, and constraints.
- Build multi-hop answer plans with traceable evidence paths.
- Add counterexample generation and plan verification.
- Add domain-specific grammar/realization rules.

## 0.4 — Learning and calibration
- Learn feature weights from a reviewed dataset, with train/validation/test splits.
- Calibrate confidence against observed correctness; do not call raw ranking scores probabilities.
- Evaluate memory updates for drift, poisoning, and catastrophic interference.
- Persist state with explicit versioning and rollback.

## 1.0 — Only after evidence
- Broader domain coverage and multilingual support.
- Stronger semantic parsing and reference resolution.
- Independent evaluation against rule-based, retrieval-based, and small CPU neural baselines.
- Publish latency, memory, energy, accuracy, abstention, and failure-rate measurements.

## Acceptance gates
A version is not promoted because it sounds intelligent. It must pass:
1. Held-out paraphrase tests.
2. Unsupported-question abstention tests.
3. Contradiction and negation tests.
4. Evidence attribution checks.
5. Reproducible CPU/RAM/latency benchmarks.
6. Regression tests across each supported language/domain.

A GPU-free runtime is a hardware property, not an intelligence benchmark.
