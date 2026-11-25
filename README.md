# MCR-L
## Mathematical Cognitive Resonance for Language

MCR-L 0.3 is an experimental CPU-first semantic-composition engine inspired by the explicit situation signals and recurrent-state ideas in SCENT-X MCR-2.

This branch is a bounded research prototype, not a general-purpose conversational model. It uses the Python standard library, requires no dedicated GPU, and does not load a neural language model. Its grammar and curated fact base are deliberately small.

## What the engine does

MCR-L separates three different jobs:

1. Understand the form of a message: compose typed meaning nodes for speech acts, events, participants, polarity, modifiers, time, comparisons, and topic references.
2. Choose a response meaning: calculate counterfactual importance for represented meaning nodes, then select a pragmatic response plan before choosing output words.
3. Answer about the world: retrieve source-attributed atomic facts or vetted document records. Compose only what the local store supports, and admit when it has no reliable information.

The active response path is:

\[
\text{utterance}\rightarrow G(V,E,A)\rightarrow S_i,w_i
\rightarrow R^*\rightarrow \text{syntax AST}\rightarrow \text{verification}
\]

The factual topic path is:

\[
\text{topic request}\rightarrow \text{fact candidates}
\rightarrow \text{source/priority selection}
\rightarrow \text{epistemic response plan}\rightarrow \text{wording}
\]

## Examples

The current conversational grammar supports these examples:

    you> Hello there
    MCR-L: Hello there!

    you> Are u there?
    MCR-L: Yes, I am here!

    you> Nice to meet u MCR
    MCR-L: Nice to meet you too!

The topic responder currently has a small, sourced Moon knowledge set. For example:

    you> Tell me about the Moon
    MCR-L: Hmm, I only have a few sourced basics about the Moon so far,
    but here's what I do know: It often looks bright, almost white, from Earth;
    up close, much of its surface is made of dark-gray shades; and it shines
    by reflecting sunlight instead of making its own light. LOL.

    you> Is it big?
    MCR-L: Hmm, I only have a few sourced basics about the Moon so far,
    but here's what I do know: It is about one-quarter as wide as Earth. LOL.

Those statements come from individual NASA-attributed facts. The answer deliberately distinguishes the Moon's bright, almost-white appearance from Earth from its mostly gray surface up close. It also states that moonlight is reflected sunlight.

For a topic with no fact or vetted document in the local store, MCR-L says so instead of inventing an answer:

    you> Anything about Neptune?
    MCR-L: Hmm, I don't have reliable facts about Neptune in my local
    knowledge yet, so I don't want to make something up. LOL.

This is not evidence of open-domain understanding. It is a narrow, inspectable system for testing semantic composition and honest use of a small fact base.

## Active architecture

1. Meaning graph — typed nodes represent speech acts, events, participants, polarity, affect, time, comparisons, and topics. Edges represent semantic roles such as speaker, addressee, asker, target, agent, patient, negation, and event membership.
2. Discourse state — keeps a bounded session topic so a simple follow-up such as “Is it big?” can resolve “it” to the last topic.
3. Counterfactual sensitivity — removes each meaning node in turn, measures the loss in structural utility, and normalizes those losses.
4. Response meaning solver — chooses a small pragmatic response plan before output words are selected.
5. Lexical compiler and grammar AST — select concept-local wording and linearize a grammar structure. This is not vocabulary-wide next-token generation.
6. Meaning validator — re-parses supported outputs and checks task-specific meaning requirements.
7. Atomic fact store — SQLite stores subject-predicate-object records with aliases, confidence, epistemic status, priority, source, URL, and tags. FTS5 is used when available, with a SQL fallback.
8. Topic responder — selects supported facts for the type of question (for example, color versus size) and composes a limited answer. If no source-backed fact exists, it admits that.
9. Document evidence path — longer factual explanations can still use the existing disk-backed document index and strict term-coverage/source checks.

## Quick start

Requires Python 3.10+; no third-party packages.

    git switch semantic-composition-v0.3
    python -m mcrl --interactive
    python -m mcrl --interactive --debug
    python -m mcrl "Tell me about the Moon" --debug
    python -m unittest discover -s tests -v

On Windows, run these commands from the repository root in PowerShell or CMD.

## Add sourced knowledge

### Add a document record

Document JSONL remains useful for longer vetted passages:

    {"id":"ram-001","title":"What is RAM?","text":"RAM is short-term working memory used by a computer while programs run.","answer":"RAM is the computer's short-term working memory.","domain":"computing","source":"Your vetted reference","licence":"CC BY 4.0","keywords":["RAM","memory"]}

Import it:

    python -m mcrl --db data/knowledge.sqlite --import-jsonl path/to/records.jsonl

### Add an atomic semantic fact

Facts should be small claims with explicit provenance. Each JSONL line is an object:

    {"id":"moon-radius","subject":"Moon","aliases":["lunar","Earth's Moon"],"predicate":"radius","object":"The Moon's mean radius is about 1,737 kilometres.","confidence":1.0,"epistemic_status":"established","priority":50,"source":"Your trusted reference","source_url":"https://example.org/source","tags":["astronomy","size"]}

Required fields are id, subject, predicate, object, source, and source_url. Confidence must be between 0 and 1. Epistemic status must be established, qualified, or tentative. Keep source and confidence labels honest; they are metadata, not automatically calibrated probabilities.

Import facts:

    python -m mcrl --db data/knowledge.sqlite --import-facts-jsonl path/to/facts.jsonl

Both imports are streamed in batches. The database is disk-backed and is not loaded as a single in-memory object. The built-in mcrl/data/facts.jsonl Moon facts are loaded idempotently when the engine starts.

### Current built-in fact sources

- NASA Science, Moonlight — What Color is the Moon? https://science.nasa.gov/moon/moonlight/
- NASA Science, Five Things to Know about the Moon https://science.nasa.gov/solar-system/moon/five-things-to-know-about-the-moon/

The current built-in Moon set covers apparent color, surface color, reflected sunlight, natural-satellite status, and approximate size relative to Earth.

## Measurement and limitations

- The printed meaning-preservation value is a ratio of passed, hand-authored checks. It is not a probability of truth or intelligence.
- The printed fact-support value is the minimum confidence metadata of the selected fact records. It is not statistically calibrated.
- The facts are manually curated and sourced; the system does not autonomously research or verify new facts at runtime.
- The parser recognizes a bounded set of constructions. It does not yet provide broad English syntax, general semantic role labeling, arbitrary coreference, or full negation/tense scope.
- Topic selection uses an explicit alias/tag index and small question-property mappings. It does not infer all possible relations from raw text.
- Unknown topics should abstain rather than receive plausible-sounding filler.
- A 2–3 GB text collection is not automatically a useful semantic knowledge base: it needs segmentation, atomic facts, provenance, licences, ontology links, indexing, and evaluation.
- The implementation avoids vocabulary-wide next-token generation; concept-local lexical choice and fact retrieval are still finite computations.

The objective remains CPU-first semantic composition. The current milestone is a narrow vertical slice, not a completed general language AI.
