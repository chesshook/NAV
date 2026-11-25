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

## 0.4 — Topic-neutral dialogue and capability routing
- [x] Remove automatic loading of demo knowledge.
- [x] Require explicit opt-in for demo documents and user facts.
- [x] Keep generic dialogue words out of topic memory.
- [x] Constrain fact selection to explicit subjects and requested properties.
- [x] Add safe arithmetic independent of the knowledge store.

## 0.5 — Pragmatic interaction and general composition
- [x] Add a separate pragmatic interpretation layer before topic extraction.
- [x] Preserve multiple speech acts in compound conversational turns.
- [x] Distinguish assistant identity, user identity, capability checks, and compliments.
- [ ] Learn grammar and lexical relations from reviewed examples with clean data splits.
- [ ] Compose clauses with subject, object, experiencer, cause, location, time, and instrument roles.
- [ ] Represent conjunction, disjunction, conditionals, quantifiers, and comparisons.

## 0.6 — Mathematical interpretation field
- [x] Score pragmatic interpretation hypotheses from weighted semantic evidence.
- [x] Expose candidate support, structural constraints, and cue sensitivity.
- [x] Track a bounded dialogue situation and recent user/assistant messages.
- [ ] Calibrate scores against measured correctness.
- [ ] Evaluate ambiguous multi-turn reference resolution on held-out conversations.

## 0.7 — Compositional interaction planning
- [x] Compile explicit speech-output directives into payload-bearing plans.
- [x] Give explicit output commands priority over prohibited embedded questions.
- [x] Treat assistant-directed action verbs as interaction requests.
- [x] Distinguish symbolic expression from physical capability.
- [x] Route standalone reactions and history questions to interaction handling.
- [x] Store bounded session transcripts.
- [ ] Broaden clause decomposition across coordination, contrast, conditionals, and embedded clauses.
- [ ] Evaluate action intent, speech-act scope, negation, and reference resolution on held-out data.

## 0.8 — Learned language foundation
- [x] Add a local OpenAI-compatible inference adapter using the Python standard library.
- [x] Define a typed semantic-frame schema with dialogue acts, clauses, references, knowledge need, ambiguity, and response goal.
- [x] Interpret each new utterance with a bounded recent conversation transcript.
- [x] Separate semantic interpretation from natural-language realization.
- [x] Add an optional independent deliberation-plan pass before realization.
- [x] Fail closed when the configured model is unavailable; do not silently use the old keyword parser.
- [x] Keep deterministic safe arithmetic as a separate capability.
- [ ] Run the new multi-turn evaluation set against actual local model weights.
- [ ] Measure semantic-frame accuracy, reference resolution, hallucination rate, latency, and peak RAM on target PCs.
- [ ] Curate a licensed, human-reviewed semantic conversation corpus with conversation-level train/validation/test splits.
- [ ] Compare prompt-only baseline, parameter-efficient fine-tuning, and smaller distilled student models.
- [ ] Calibrate uncertainty from held-out results.

## 0.9 — Evidence-aware world knowledge (planned)
- [ ] Retrieve relevant facts only after the semantic frame identifies a knowledge need.
- [ ] Store source provenance and dates; distinguish model memory from verified current evidence.
- [ ] Detect contradictions and answer only what the evidence supports.
- [ ] Add multi-hop reasoning only after single-hop evidence selection is reliable.

## Acceptance gates

A milestone is promoted only after:
1. Semantic interpretations generalize to unseen paraphrases and informal writing.
2. Context-dependent replies resolve the correct previous proposition.
3. Role reversals, negation, tense, and property contrasts are distinguished.
4. Requests without a necessary subject trigger a useful clarification.
5. Model knowledge is not presented as externally verified fact.
6. Unknown or ambiguous cases are handled with calibrated uncertainty.
7. Tests run across Python 3.10–3.13.
8. CPU, RAM, latency, and language coverage are measured independently.

MCR-L remains a research prototype. The symbolic engine is inspectable but limited; a
pretrained model adds learned linguistic competence but does not guarantee correctness.
The objective is measurable generalization, not a claim of perfect understanding.
