# MCR-L roadmap

## 0.2–0.7 — Symbolic compositional prototype
- [x] Typed meaning graph, pragmatic hypotheses, interaction planning, bounded session history.
- [x] Sourced facts, abstention, deterministic arithmetic.
- [x] Exposed limitations of hand-written phrase inventory through transcript regressions.

## 0.8 — Learned language adapter
- [x] Local pretrained model adapter with structured interpretation and response realization.
- [x] Optional separate deliberation plan.
- [x] Explicitly remains a Qwen-backed broad language mode, not independent mathematical comprehension.

## 0.9 — Mathematical lexical-semantic network
- [x] WordNet LMF importer and custom streaming JSONL concept import.
- [x] Context-sensitive word-sense candidates, graph relations, phrase activations, unknown-word retention.
- [x] Pass lexical evidence into the v0.8 model.
- [x] Batch relation-subgraph loading.
- [ ] Learn sense scores and relation weights from reviewed examples and calibrate them.

## 1.0 — Independent mathematical comprehension foundation
- [x] Separate mathematical mode that does not call a language model.
- [x] Stream CoNLL-U Universal Dependencies data.
- [x] Train sparse averaged-perceptron POS tagging on English EWT.
- [x] Train arc-standard dependency transitions using gold dependency trees.
- [x] Use dictionary POS hints at inference.
- [x] Pass all syntactic tokens, including unknown words and punctuation, into the parser.
- [x] Compose a meaning structure with predicates, grammatical roles, polarity, modality, time, and communicative structure.
- [x] Add training/dev/held-out-test metrics for POS accuracy, UAS, LAS, and parse completeness.
- [ ] Measure actual parser metrics on target data after the user trains the model.
- [ ] Evaluate open-ended interviews with frozen, unseen prompts and independent human labels.
- [ ] Expand syntax training with additional licensed genres and language varieties.
- [ ] Train verb frames, thematic roles, semantic scope, event relations, and discourse reference.
- [ ] Replace overlap heuristics with learned word-sense disambiguation and relation propagation.
- [ ] Calibrate parser/sense uncertainty against observed correctness.

## 1.1 — Independent sentence/discourse semantics (planned)
- [ ] Generalize semantic roles beyond syntactic subject/object candidates.
- [ ] Represent negation scope, quantifiers, conditionals, modality, comparison, and event time compositionally.
- [ ] Learn coreference and ellipsis resolution across multi-turn dialogue.
- [ ] Add world/event grounding and distinguish claims, questions, hypotheticals, quotations, and instructions.
- [ ] Support unknown lexical terms without losing compositional syntax.
- [ ] Hold out complete conversations and domains for evaluation.

## 1.2 — Conversational teaching (planned; intentionally deferred)
- [ ] Accept user teaching as candidate concept/relation/policy updates.
- [ ] Resolve what the correction refers to before changing persistent memory.
- [ ] Store source, speaker, context, confidence, and version provenance.
- [ ] Separate learning facts from learning behaviour and response style.
- [ ] Require review/confirmation for uncertain or contradictory updates.

## Acceptance gates

1. Syntax and semantic structures generalize to unseen paraphrases and informal input.
2. Unknown words remain parseable and explicitly unresolved.
3. Role reversals, negation, tense, modality, and question/declarative contrasts are distinguished.
4. Ambiguity and parse failures are surfaced rather than hidden.
5. Held-out data is separated by sentence/conversation and not used for tuning.
6. CPU time, RAM, model size, and per-utterance latency are reported.
7. Human interview labels agree with the meaning structure at a measured rate.

MCR-L is a research prototype. The objective is to grow an independent, inspectable understanding system by measured generalization, not by adding answer patterns or claiming completeness prematurely.
