# MCR-L 0.9 — mathematical lexical-semantic network

## Objective

MCR-L 0.9 introduces a persistent mathematical lexical-semantic graph. Each lexical
word in an utterance is represented, looked up against candidate lexical senses,
scored against local and conversation context, and connected to typed concept
relations. Unknown words are retained as unresolved nodes with character n-grams;
they are not silently discarded or assigned a guessed definition.

The graph is the lexical foundation. It does not, by itself, understand every sentence.
It must not confuse a word's dictionary meaning with the role that word plays in a
particular utterance, the speaker's intention, or whether the utterance is true.

## Architecture

The current pipeline is:

1. **Lexical normalization.** Preserve the original surface form, map common informal
   spellings to canonical forms, and generate conservative morphology candidates.
2. **Sense retrieval.** Retrieve all candidate synsets/senses for every recognized
   lexical form. A word with several senses keeps several candidate nodes.
3. **Context activation.** Compare each sense gloss, examples, and lemma against the
   nearby words and the complete utterance. The most recent dialogue is a weaker
   secondary context, never a replacement for the current sentence.
4. **Relation propagation.** Use both concept-to-concept synset edges and
   sense-to-sense lexical edges (hypernym, meronym/holonym, antonym, derivation,
   entailment, cause, and other WN-LMF relations) to add contextual support.
5. **Ambiguity preservation.** Select a sense only when one candidate is sufficiently
   better supported. Otherwise retain multiple candidates and mark the word ambiguous.
6. **Phrase activation.** Check contiguous two-to-five-word sequences against
   multiword lexical entries. Phrase senses are separate nodes from component words.
7. **Composition.** Pass the lexical field to the clause/discourse interpreter. That
   stage still has to determine grammar, agent/patient roles, negation scope, modality,
   intent, reference, and conversational purpose.
8. **Response.** The existing learned-language stage realizes a response from the
   semantic interpretation. The lexical network supplies evidence; it does not answer
   a question by itself.

Unknown terms keep a character n-gram representation. This gives later learning
systems a stable unresolved symbol, not a fabricated meaning.

## Mathematical activation

For a candidate synset s and an occurrence at word position i, let G(s) be the terms
in its definition, examples, and lemma. Let C_local(i) be the terms in the surrounding
five-word window and C_turn be the current utterance's content terms plus a lower
weight dialogue context.

The lexical support used in this prototype separates the current sentence from
historical conversation:

\[
\begin{aligned}
L(s\mid i)=&
0.50\frac{|G(s)\cap C_{local}(i)|}{\max(1,|C_{local}(i)|)}\\
&+0.25\frac{|G(s)\cap C_{turn}|}{\max(1,|C_{turn}|)}\\
&+0.10\frac{|G(s)\cap C_{history}|}{\max(1,|C_{history}|)}\\
&+0.15\frac{|G(s)\cap(C_{local}(i)\cup C_{turn}\cup C_{history})|}{\max(1,|G(s)|)}
\end{aligned}
\]

For typed relation r, the network assigns a relation weight w(r). A neighboring
candidate t contributes only in proportion to its own lexical support q(t):

\[
R(s_i)=\max_{j\ne i,\,t\in S_j,\,r\in E}
w(r)\,(0.20+0.80q(t))\,\mathbf{1}[(s_i,t,r)\in E]
\]

The combined engineering support is:

\[
A(s_i)=0.72L(s_i)+0.28R(s_i)
\]

The candidate margin is the difference between the best and second-best support.
A small margin is recorded as ambiguity instead of forcing a single sense.

These are **heuristic support scores, not calibrated probabilities**. They are
inspectable and testable; they must later be learned or calibrated against labelled
word-sense and sentence-meaning examples. A relation in the dictionary is lexical
evidence only—it is not proof that two words are connected by that relation in the
sentence.

## Dictionary format and sources

### Open English WordNet

The Windows helper downloads the official Open English WordNet 2025+ LMF file and
imports it as a streaming two-pass operation:

    scripts/import-open-english-wordnet.cmd

The data is stored in:

    data/semantic_network.sqlite

The importer accepts uncompressed XML/LMF, gzip-compressed XML/LMF, and ZIP archives.
It streams lexical entries and sense-to-sense relations first, then concept definitions,
examples, synsets, and synset-to-synset relations; it does not load the complete
dictionary into RAM.

The Open English WordNet project reports for its 2025+ release:
- 161,875 word forms;
- 120,564 synsets;
- 419,226 relations.

The 2025 core edition (without the selected proper-noun extension) reports 135,969
words, 107,519 synsets, and 355,064 relations. The project publishes the resource under
CC BY 4.0; preserve attribution and upstream notices. Its network distinguishes
specific senses and typed semantic relations, rather than merely listing synonyms.

Sources:
- https://github.com/globalwordnet/english-wordnet
- https://en-word.net/
- https://wordnet.princeton.edu/license-and-commercial-use

The WordNet resource is much smaller than 2–5 GB. It is a high-quality seed graph,
not a complete model of language, facts, events, social intent, or common sense.

### Adding larger dictionaries and experience corpora

MCR-L also supports streaming custom JSONL. One record describes a concept:

    {"id":"vehicle.n.01","pos":"n","definition":"A means of transport.","examples":["The vehicle stopped."],"lemmas":["vehicle","motor vehicle"],"relations":[{"type":"hypernym","target":"artifact.n.01"}]}

Import with source attribution and license metadata:

    py -3 -m mcrl --lexicon-db data/semantic_network.sqlite --import-semantic-jsonl my_lexicon.jsonl --lexicon-source "Dataset name" --lexicon-license "Dataset license" --lexicon-source-url "https://example.org"

JSONL is streamed line by line in batches; a multi-gigabyte file is not read into RAM.
A 2–5 GB target is reasonable as a storage budget for a *combined* resource of
lexical entries, definitions, usage examples, syntactic frames, domain terminology,
conversation contexts, and reviewed semantic relations. It is not useful to fill that
budget with duplicated definitions or unlicensed copied text. Every source needs
provenance, a license, deduplication, and quality control.

## Inspect the lexical field

After importing WordNet:

    py -3 -m mcrl --lexicon-db data/semantic_network.sqlite --semantic-inspect "I deposited money at the bank."

This prints every word's original form, normalization candidates, candidate senses,
support values, unresolved subword units, and recognized multiword concepts.

The network can also be used during conversation:

    py -3 -m mcrl --neural --interactive --debug

The lexical graph runs before neural semantic composition. Its candidate definitions
and relations are supplied as lexical evidence to the contextual interpreter. The
model is still responsible for interpreting the clause and discourse as a whole.

## What is not solved yet

This milestone does not claim complete language understanding. It provides a reusable
lexical graph and one mathematical sense-activation method. It does not yet provide:

- broad syntactic parsing and unrestricted semantic role assignment;
- reliable scope for nested negation, quantifiers, modality, and conditionals;
- full reference resolution over long conversations;
- grounded models of physical events, causality, space, time, and social situations;
- confidence calibration on a held-out test set;
- independent natural-language generation without the v0.8 model layer;
- automatic self-learning from arbitrary conversations.

The next phase should train and evaluate sentence composition over a broad licensed
corpus: sentence-to-meaning graph examples, multi-turn reference labels, corrections,
and contrastive pairs. After the system demonstrates robust generalization on
unseen interview-style conversations, an explicit chat-teaching mechanism can safely
add reviewed concepts, relations, and behavioural examples to persistent memory.

## Design constraint: understanding before teaching

The planned learning loop is:

    user utterance
        -> lexical network
        -> sentence/discourse meaning graph
        -> ambiguity and consistency checks
        -> learner state / concept memory
        -> response policy
        -> language realization

A teaching correction should update a concept, relation, or policy only after MCR-L can
represent what is being corrected and preserve provenance. It must not silently treat
every conversational sentence as a verified fact or instruction.
