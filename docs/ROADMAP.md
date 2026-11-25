# MCR-L roadmap

## 0.2 — Semantic-composition foundation
- [x] Typed meaning graph with nodes and semantic-role edges.
- [x] Counterfactual node-ablation sensitivity.
- [x] Response meaning selection before lexical realization.
- [x] Concept-local lexical candidate ranking.
- [x] Grammar AST and output verification for a bounded speech-act slice.

## 0.3 — Conversational and epistemic prototype
- [x] Greeting and presence-check constructions.
- [x] Atomic sourced facts and import tooling.
- [x] Limited topic responses and simple follow-up topic resolution.
- [x] Honest response path for unsupported topics.

## 0.4 — Topic-neutral dialogue and capability routing (current branch)
- [x] Remove automatic loading of the Moon fact file and demo document file.
- [x] Require explicit opt-in for demo documents and explicit import for user facts.
- [x] Keep generic dialogue words out of the topic memory.
- [x] Use only explicit current topics or valid question anaphors to resolve discourse references.
- [x] Constrain fact selection to the explicit subject/alias and requested property.
- [x] Never replace a missing distance/color/size fact with an unrelated fact about the same subject.
- [x] Recognize appreciation such as “that's great” and respond conversationally.
- [x] Recognize “don't tell me about X” as a topic-boundary request.
- [x] Add safe arithmetic independent of the knowledge store.
- [x] Add regression cases for topic isolation, calculations, acknowledgements, boundaries, and property-specific abstention.
- [ ] Broaden conversational constructions using held-out paraphrase data, not only authored patterns.
- [ ] Improve natural topic phrase extraction for relation questions with multiple entities.
- [ ] Add general event frames, tense/aspect, modality, and robust negation scope.

## 0.5 — Pragmatic interaction and general composition
- [x] Add a separate pragmatic interpretation layer before topic extraction and response selection.
- [x] Preserve multiple speech acts in a compound conversational turn.
- [x] Distinguish assistant identity, user identity, capability checks, and compliments.
- [x] Prevent social turns and identity/capability questions from creating false factual topics.
- [x] Add direct verb-form addition extraction to the safe arithmetic route.
- [x] Add regression tests for the observed v0.4 failures.
- [ ] Learn grammar and lexical relations from reviewed examples with train/validation/test splits.
- [ ] Compose clauses with subject, object, experiencer, cause, location, time, and instrument roles.
- [ ] Represent conjunction, disjunction, conditionals, quantifiers, and comparisons.
- [ ] Resolve references across multiple turns with explicit ambiguity handling.
- [ ] Build meaning plans for open-ended statements, questions, requests, corrections, and disagreement.
- [ ] Evaluate outputs by semantic equivalence and pragmatic appropriateness, not only lexical checks.

## 0.6 — Knowledge and reasoning (optional layer)
- [ ] Curate user-selected and licensed atomic facts with provenance and source dates.
- [ ] Add subject/predicate/object indexing and source conflict detection.
- [ ] Require full subject and property match before claim selection.
- [ ] Extract candidate facts from text with source spans; require review before promotion.
- [ ] Add multi-hop reasoning only after single-hop evidence selection is reliable.
- [ ] Calibrate confidence against measured correctness and freshness.

## Acceptance gates

A milestone is promoted only after:
1. Held-out paraphrases map to equivalent semantic structures.
2. Unrelated new topics do not inherit prior discourse topics.
3. Role reversals, negation, tense, and property contrasts are distinguished.
4. Arithmetic and other safe capabilities do not consult unrelated domain memory.
5. Unknown-topic and unknown-property questions abstain with the correct subject.
6. Source-attributed facts cannot cross-contaminate subjects or requested predicates.
7. Tests run across Python 3.10–3.13.
8. CPU, RAM, latency, and broader language coverage are measured separately.

The intended architecture remains meaning-first and CPU-first. MCR-L 0.5 is still a bounded prototype: the pragmatic layer repairs concrete conversational distinctions but does not yet provide broad learned language understanding, a persistent user model, general common-sense reasoning, or open-ended natural-language generation. Those remain explicit research milestones.
