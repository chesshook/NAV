# MCR-L roadmap

## 0.2 — Semantic-composition foundation (current branch)
- [x] Typed meaning graph with nodes and semantic-role edges.
- [x] Bounded discourse state for speaker/addressee roles and recent referents.
- [x] Counterfactual node-ablation sensitivity and normalized importance weights.
- [x] Response meaning selection before lexical realization.
- [x] Concept-local lexical candidate ranking.
- [x] Grammar AST and deterministic linearization for an initial speech-act slice.
- [x] Back-parsing checks for response meaning preservation.
- [x] Separate evidence-gated factual lookup path.
- [x] Regression tests for paraphrase equivalence, negation, temporal/comparative node extraction, evidence gating, and social composition.
- [ ] Broader grammatical parsing and semantic-role assignment for ordinary clauses.
- [ ] Persistent discourse state with explicit privacy/reset controls.
- [ ] Better morphology, spelling normalization, multiword lexemes, and ambiguity handling.

## 0.3 — Compositional operators
- [ ] Typed event frames with subject, object, instrument, location, cause, and recipient roles.
- [ ] Logical operators for conjunction, disjunction, implication, quantifiers, and conditionals.
- [ ] Negation scope and temporal ordering that can be reasoned over, not merely detected.
- [ ] Comparatives and quantities with unit-aware constraints.
- [ ] Reference resolution across multiple turns and explicit correction handling.
- [ ] Domain ontology and concept-to-concept relation traversal.
- [ ] Evidence attribution and contradiction detection.

## 0.4 — Knowledge and learning
- [ ] Curated and licensed concept/lexeme/grammar database with provenance.
- [ ] Streaming corpus compiler that builds indexed semantic records rather than treating raw prose as intelligence.
- [ ] Reviewed semantic examples with train/validation/test splits.
- [ ] Learn lexical and structural weights from held-out examples; do not hand-tune against the test set.
- [ ] Confidence calibration against measured accuracy.
- [ ] Evaluate memory updates for drift, poisoning, and catastrophic interference.

## 0.5 — Broader language coverage
- [ ] Clause composition and discourse relations across supported domains.
- [ ] Multi-hop evidence-backed reasoning over typed relations.
- [ ] More grammar constructions and varied surface realizations.
- [ ] Independent evaluation against rule-based, retrieval, and small CPU neural baselines.
- [ ] Multilingual support only after the English pipeline is measurable.

## Acceptance gates
A milestone is not promoted because a few examples sound natural. It must pass:
1. Held-out paraphrase tests where unseen surface forms map to the same meaning.
2. Negation-scope, temporal-order, and comparative-relation tests.
3. Unsupported-question abstention tests.
4. Meaning-preservation tests that catch omitted or reversed roles.
5. Evidence attribution and contradiction checks.
6. Reproducible CPU/RAM/latency/energy measurements.
7. Regression tests across all supported grammar constructions.

The long-term objective remains CPU-first semantic composition. The current implementation is only the initial, narrow vertical slice needed to test that design.
