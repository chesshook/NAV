# MCR-L
## Mathematical Cognitive Resonance for Language

MCR-L 0.2 is an experimental CPU-first semantic-composition engine inspired by the explicit situation signals and recurrent-state ideas in SCENT-X MCR-2.

This branch is a **research prototype**, not a general-purpose conversational model. It does not load a neural language model, does not require a GPU, and uses only the Python standard library. Its current grammar and semantic ontology are deliberately bounded.

## What changed from 0.1

MCR-L 0.1 was retrieval-led: it searched for a stored answer and calculated sensitivity after selecting that answer. That was the wrong foundation for the intended experiment.

MCR-L 0.2 separates two different tasks:

- **Language composition:** parse an utterance into a typed meaning graph; measure the counterfactual importance of its nodes; solve for a response meaning; choose words from concept-local lexical candidates; build a grammar structure; and back-parse the result to verify meaning preservation.
- **Factual answering:** use the local index only to find possible evidence, then require multi-term coverage and source metadata before returning a stored factual answer.

The language-composition path does not retrieve a canned sentence for each input.

## Active response pipeline

\[
\text{utterance}\rightarrow G(V,E,A)\rightarrow S_i,w_i
\rightarrow R^*\rightarrow \text{syntax AST}\rightarrow \text{verification}
\]

1. **Meaning graph** — typed nodes represent speech acts, events, participants, polarity, affect, temporal relations, comparisons, and topics. Edges represent roles such as speaker, addressee, event, negation, and time scope.
2. **Discourse state** — keeps bounded session-level speaker/addressee roles and recent referents.
3. **Counterfactual sensitivity** — removes each meaning node in turn, measures the loss in graph utility, and normalizes the losses. High-sensitivity meaning must be preserved.
4. **Response meaning solver** — scores a small set of pragmatic response operators against the weighted meaning, before selecting any output words.
5. **Lexical compiler** — considers only words attached to the selected concept, scoring semantic precision, grammar fit, register, ambiguity, and lexical reuse.
6. **Grammar engine** — builds a small syntax AST and linearizes it into a sentence.
7. **Meaning validator** — parses the generated sentence again and checks whether key semantic requirements survived.
8. **Evidence path** — factual questions use SQLite FTS5, where available, only as a candidate lookup accelerator. The candidate must pass a separate coverage/source gate.

## First supported composition slice

The initial rules cover basic greetings, gratitude, farewells, wellbeing questions, identity questions, and positive meeting pleasantries. For example:

    you> Nice to meet u MCR
    MCR-L: Nice to meet you too!

The response is assembled from:
- positive meeting meaning,
- user as speaker,
- assistant as addressee,
- reciprocal social intent,
- a grammar construction for an adjective + infinitive + object + reciprocal marker.

This is a narrow, explicit grammar—not proof that the engine understands arbitrary English. The parser also extracts negation, temporal, and comparative nodes, but extraction alone does not mean the system can yet reason correctly about every such relation.

## Quick start

Requires Python 3.10+; no third-party packages.

    python -m mcrl --interactive
    python -m mcrl --interactive --debug
    python -m mcrl "Nice to meet u MCR" --debug
    python -m unittest discover -s tests -v

On Windows, run these from the repository root in PowerShell or CMD.

## Add sourced factual knowledge

Create UTF-8 JSON Lines records, one per line:

    {"id":"ram-001","title":"What is RAM?","text":"RAM is short-term working memory used by a computer while programs run.","answer":"RAM is the computer's short-term working memory. It holds data and instructions that active programs need quickly.","domain":"computing","source":"Your vetted reference","keywords":["RAM","memory"]}

Import them:

    python -m mcrl --db data/knowledge.sqlite --import-jsonl path/to/records.jsonl
    python -m mcrl --db data/knowledge.sqlite "What is RAM?"

JSONL ingestion is streamed in batches. The SQLite database can be larger than available RAM because it is not loaded as a single in-memory object. A 2–3 GB text collection is not automatically a useful semantic knowledge base: it needs clean records, ontology links, provenance, rights/licence metadata, and evaluation.

## Measurement and limitations

- The printed semantic-composition score is a **meaning-preservation ratio**, not a probability of intelligence or truth.
- The factual path score is a ranking heuristic, not a calibrated probability.
- There is no broad open-domain semantic parser, learned ontology, multilingual grammar, multi-hop fact reasoner, or validated online learning in this milestone.
- The implementation does not eliminate computation or brute force in every possible form. It avoids vocabulary-wide next-token generation; concept-local lexical choice is still a finite candidate selection.
- Hand-authored concepts/rules are the current coverage ceiling. Unknown constructions should abstain rather than be disguised as understanding.

See docs/FORMULAS.md for the active equations and docs/ROADMAP.md for acceptance gates.
