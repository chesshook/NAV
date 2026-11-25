# MCR-L roadmap

## 0.2 — Semantic-composition foundation
- [x] Typed meaning graph with nodes and semantic-role edges.
- [x] Bounded discourse state for participant roles and recent topics.
- [x] Counterfactual node-ablation sensitivity and normalized importance weights.
- [x] Response meaning selection before lexical realization.
- [x] Concept-local lexical candidate ranking.
- [x] Grammar AST and deterministic linearization for an initial speech-act slice.
- [x] Back-parsing checks for response meaning preservation.
- [x] Separate evidence-gated factual lookup path.
- [x] Clarification requests for bare wh fragments.
- [x] Regression tests for paraphrase equivalence, negation, temporal/comparative node extraction, evidence gating, and social composition.

## 0.3 — Conversational and epistemic grounding (current branch)
- [x] Distinguish salutation extensions such as “Hello there” from presence questions such as “Are u there?”
- [x] Represent presence checks with asker and target roles; compose a direct presence confirmation.
- [x] Store atomic subject-predicate-object facts with aliases, tags, priority, confidence metadata, epistemic status, source, and source URL.
- [x] Stream fact JSONL into SQLite with FTS5 and a fallback path.
- [x] Parse common topic-request forms such as “Tell me about X” and “What do you know about X?”
- [x] Compose casual, limited answers from selected fact records.
- [x] Admit missing knowledge without inserting fabricated filler facts.
- [x] Resolve a simple topic follow-up such as “Is it big?” against the previous topic.
- [x] Seed a small, NASA-sourced Moon fact set for appearance, surface color, reflected sunlight, natural-satellite status, and relative size.
- [x] Test presence, paraphrases, property selection, sourced partial answers, topic follow-ups, and unknown-topic abstention.
- [x] Add a CLI import path for custom fact JSONL.
- [ ] Broaden the ontology and fact coverage to more everyday subjects with provenance.
- [ ] Separate user-facing conversational style from fact confidence and source evidence.
- [ ] Improve source conflict handling and duplicate/contradictory fact detection.

## 0.4 — General compositional language
- [ ] Typed event frames for subjects, objects, instruments, locations, recipients, causes, and experiencers.
- [ ] Logical operators for conjunction, disjunction, implication, quantifiers, and conditionals.
- [ ] Correct negation scope, tense, aspect, and modality.
- [ ] Comparatives, quantities, and unit-aware constraints.
- [ ] Robust coreference and discourse relation parsing across multiple turns.
- [ ] Grammar constructions that compose clauses, not only fixed social turns.
- [ ] Meaning-preservation evaluation over held-out paraphrases and adversarial minimal pairs.

## 0.5 — Learning and scale
- [ ] Curated, licensed lexical/grammar/ontology database.
- [ ] Corpus compiler that extracts candidate atomic facts with source spans; human review before facts are promoted as reliable.
- [ ] Learn lexical and structural weights from reviewed examples with train/validation/test splits.
- [ ] Confidence calibration against measured answer correctness.
- [ ] Contradiction detection, provenance ranking, and source-age policies.
- [ ] Multi-hop reasoning over typed facts.
- [ ] CPU/RAM/latency/energy benchmarks on a multi-gigabyte indexed knowledge store.
- [ ] Independent comparison against rule-based, retrieval, and small CPU neural baselines.

## Acceptance gates

A milestone is not promoted because a few examples sound natural. It must pass:
1. Held-out paraphrase tests where unseen surface forms map to equivalent meaning graphs.
2. Negation-scope, temporal-order, role-reversal, and comparative-relation tests.
3. Unsupported-topic abstention tests.
4. Meaning-preservation tests that catch omitted or reversed roles.
5. Evidence attribution and contradiction checks.
6. Reproducible CPU/RAM/latency/energy measurements.
7. Regression tests across all supported grammar constructions.

The long-term objective remains CPU-first, meaning-first language composition. MCR-L 0.3 is a narrow practical milestone, not a completed general language AI.
