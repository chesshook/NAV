
# MCR-L
## Mathematical Cognitive Resonance for Language

MCR-L 0.9 adds a mathematical lexical-semantic network to the learned-language
foundation. Its purpose is to represent word senses and relations explicitly, retain
ambiguity, and provide reusable lexical evidence for understanding complete
utterances. It is still a research prototype, not a claim of perfect understanding.

## Core architecture

The intended flow is:

    every word and phrase
        -> lexical normalization and morphology
        -> candidate dictionary senses
        -> context-sensitive mathematical activation
        -> typed semantic relation graph
        -> sentence and discourse composition
        -> intent / response planning
        -> natural-language realization

Every lexical item is retained. Known words activate candidate senses; multiword
lexical entries become separate phrase nodes; unknown words remain unresolved nodes
with character n-grams instead of being dropped or assigned an invented meaning.

**The dictionary is not the sentence interpreter.** It describes possible lexical
meanings. A separate compositional stage must work out grammar, who did what, who is
being addressed, negation, modality, time, intent, references, and the speaker's
commitment. A word's dictionary definition is not automatically what the speaker
means in context.

The v0.8 local pretrained model remains the current broad sentence/discourse
interpreter and language realizer. In v0.9 the mathematical network runs first and
supplies dictionary sense candidates and relation evidence to it. This is a deliberate
hybrid foundation—not yet a fully independent language engine.

## Import a real semantic dictionary

The first supported lexical backbone is Open English WordNet 2025+. It is a graph of
word forms, senses, glosses, examples, and typed relations, not merely a flat word list.
The project's 2025+ release reports 161,875 word forms, 120,564 synsets, and 419,226
relations. The project publishes it under CC BY 4.0; preserve attribution and source
notices. See the
[official repository](https://github.com/globalwordnet/english-wordnet) and
[Princeton WordNet license information](https://wordnet.princeton.edu/license-and-commercial-use).

From the repository root on Windows:

    scripts/import-open-english-wordnet.cmd

This downloads and imports the LMF dictionary into data/semantic_network.sqlite.
The importer streams the XML twice into SQLite and does not load the full dictionary
into memory.

Inspect how a word is represented in context:

    py -3 -m mcrl --lexicon-db data/semantic_network.sqlite --semantic-inspect "I deposited money at the bank."

The output includes the original surface words, normalization candidates, alternative
senses, their support values, unknown words, and recognized multiword expressions.
For a single ambiguous word such as "bank", multiple senses should remain available
when context is insufficient.

## Run contextual conversation

Start llama.cpp in one terminal:

    scripts/start-local-model.cmd

Then start MCR-L in another:

    py -3 -m mcrl --neural --interactive --debug

Now each utterance is encoded by the lexical-semantic network before the contextual
sentence interpreter and response realizer run.

The optional deliberation stage remains available:

    py -3 -m mcrl --neural --think --interactive --debug

It is separate response planning; it does not make the semantic network itself more
accurate, and it is not a reproduction of DeepSeek-R1 training.

## Build a larger dictionary

A 2–5 GB storage budget can later hold much more than a dictionary: definitions,
examples, syntactic frames, domain terminology, event relations, reviewed discourse
examples, and user-taught concepts. MCR-L supports streaming custom JSONL import:

    py -3 -m mcrl --lexicon-db data/semantic_network.sqlite --import-semantic-jsonl my_lexicon.jsonl --lexicon-source "My dataset" --lexicon-license "CC BY 4.0" --lexicon-source-url "https://example.org"

Each line is one concept record:

    {"id":"vehicle.n.01","pos":"n","definition":"A means of transport.","examples":["The vehicle stopped."],"lemmas":["vehicle","motor vehicle"],"relations":[{"type":"hypernym","target":"artifact.n.01"}]}

Large files are read in batches. Only import data whose license permits the intended
use, preserve provenance, deduplicate records, and distinguish dictionary claims from
verified world facts. More bytes alone do not create understanding.

## Mathematical scoring

The current sense score combines context compatibility and graph relation support:

    lexical_support = 0.50 * local_gloss_overlap
                    + 0.25 * current_utterance_overlap
                    + 0.10 * prior_dialogue_overlap
                    + 0.15 * evidence_density

    total_support = 0.72 * lexical_support
                  + 0.28 * typed_relation_support

The support margin is used to retain ambiguity rather than forcing an arbitrary sense.
These are engineering support values, not calibrated probabilities. Definitions and
graph links are lexical evidence, not sentence-role assertions or proof that an
utterance is factually true.

See docs/MATHEMATICAL_SEMANTIC_NETWORK.md for the full design, equations, import format,
and next research stages.

## Testing

    py -3 -m unittest discover -s tests -v

The tests cover LMF import, JSONL import, ambiguous word senses, context-driven
disambiguation, phrase concepts, unknown-word retention, and integration with the
neural dialogue path. Tests do not prove complete understanding; interview-style
held-out evaluation is still required.

## The actual long-term goal

MCR-L should learn to interpret arbitrary conversation before it is taught how to
behave. The next research stage is to develop and evaluate sentence/discourse
composition over licensed, reviewed examples. Once it can robustly represent what
someone said—including ambiguity and uncertainty—we can build chat-based teaching
that updates its concept and behaviour memory with explicit provenance.

The lexical network is one foundation layer. Broad comprehension also requires
syntax, compositional semantics, discourse, grounded knowledge, world models,
pragmatics, and learned language realization. We will measure each capability rather
than claim that a large dictionary alone solves them.
