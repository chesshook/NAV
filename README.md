# MCR-L
## Mathematical Cognitive Resonance for Language

MCR-L 0.7 is an experimental CPU-first, meaning-first conversational system. It uses typed semantic representations, an explicit pragmatic interpretation layer, compound dialogue-act planning, a small grammar engine, optional sourced facts, and safe arithmetic. It uses the Python standard library and does not require a dedicated GPU.

It is **not** a general-purpose language model and does not claim complete or human-level understanding. This milestone focuses on a foundation: the current message must determine what the system is responding to. A stale topic or an unrelated database hit must never hijack the answer.

## Design separation

MCR-L separates four responsibilities:

1. **Dialogue understanding:** infer the communicative purpose of the current utterance, retain multiple moves in one turn, and distinguish questions about the assistant from questions about the user.
2. **Response framing:** decide whether to answer, acknowledge, clarify, calculate, ask a follow-up, or admit that relevant knowledge is missing.
3. **Optional world knowledge:** use only explicitly imported, source-attributed documents or atomic facts. No subject domain is silently preloaded.
4. **Surface realization:** build a response from the selected response meaning, lexical choices, and a grammar construction.

The interactional layer also distinguishes compliments, capability checks, identity requests, and compound turns such as a greeting followed by a wellbeing question. These are separate pragmatic signals; they are not promoted to factual topics.

## Mathematical interpretation field

MCR-L does not frame a reply by repeatedly selecting a likely next token. It first builds an explicit interpretation and response meaning, then realizes that meaning through grammar. The v0.6 pragmatic layer evaluates competing semantic hypotheses using weighted role and construction evidence.

For a candidate interpretation h, its support is the weighted sum of observed cues divided by total cue weight, minus an explicit contradiction penalty. Counterfactual sensitivity is the decrease in support when one cue is removed. The resulting cue impacts are normalized to show which evidence carries the interpretation. These values are engineering support measures, **not calibrated probabilities**.

The bounded dialogue situation model tracks recent user moves, conversational goal, participant roles, active topic, interaction tone, and the assistant action selected. It stores the actual utterances and responses for a limited session recap; it does not infer a real-world identity or silently create personal facts.

## Compositional interaction planning

Before a message can reach factual retrieval, MCR-L checks whether it expresses an operation on the conversation itself. This layer produces an explicit plan with operation, payload, constraints, supporting cues, and precedence.

Examples:

    you> can u smile for me?
    MCR-L: 😊

    you> wowow
    MCR-L: Wowow! What's going on?

    you> dont tell me who are you, just say Helo
    MCR-L: Helo

    you> what we talked till now?
    MCR-L: [recap built from the actual user and assistant turns in this session]

An explicit output directive has priority over an embedded question that the user has prohibited. Action verbs such as "smile" are treated as actions requested of the assistant, not automatically converted into factual subjects. Session recaps are built from recorded turns and clearly limit themselves to what was actually said.

This is still a bounded compositional grammar. It is more flexible than one-answer-per-question templates, but it is not unrestricted language understanding. A genuinely GPT-scale conversational capability requires broad learned language representations and training data; the symbolic engine alone cannot manufacture that competence.

The dialogue path remains:

\[
\text{utterance}\rightarrow G(V,E,A)\rightarrow S_i,w_i
\rightarrow R^*\rightarrow \text{syntax AST}\rightarrow \text{verification}
\]

A separate safe arithmetic route handles explicit numeric expressions without consulting subject knowledge.

## Examples

The current bounded conversational grammar includes:

    you> Hello there
    MCR-L: Hello there!

    you> Are u there?
    MCR-L: Yes, I am here!

    you> Nice to meet u MCR
    MCR-L: Nice to meet you too!

    you> thats great
    MCR-L: Glad it helps! What should we tackle next?

    you> dont tell me anything about moon
    MCR-L: Got it — I won’t bring up Moon unless you ask.

    you> Hi! How is it going?
    MCR-L: Hi! I'm here and ready to help—thanks for asking. How are things going with you?

    you> do you know who I am?
    MCR-L: I only know what you've shared with me in this conversation.
    I can't identify you independently or see private details.

    you> tell me your name
    MCR-L: I'm MCR-L, a CPU-first semantic-composition prototype.

A supported arithmetic expression is evaluated independently:

    you> Can u tell me the value of sum of 4 and 5?
    MCR-L: 4 + 5 = 9.

The evaluator accepts only numeric literals and a short whitelist of arithmetic AST operations. It never uses Python eval or exec.

## Knowledge is optional, not the personality

A clean MCR-L engine starts with **zero domain facts and zero demo documents**. It does not automatically load the Moon sample or project/computing demo documents.

With no source-backed knowledge imported:

    you> What do you know about the Moon?
    MCR-L: Hmm, I don't know enough about Moon to give you a reliable answer yet.
    I don't want to bluff or make something up. Ask me about a specific angle
    and I'll tell you what I can support.

A property-specific question must not be answered with an unrelated fact about the same subject:

    you> How far is the Moon from Earth?
    MCR-L: Hmm, I don't have a reliable distance fact for Moon Earth yet.
    I don't want to bluff or swap in an unrelated fact.

This is the intended behavior when MCR-L has no knowledge of the requested detail. The engine does not need a Moon fact to understand that the question asks about a distance.

### Import knowledge only when deliberately requested

A JSONL fact file can be imported explicitly:

    python -m mcrl --db data/knowledge.sqlite --import-facts-jsonl facts.jsonl

A fact record contains an id, subject, predicate, and object; optional aliases and tags; confidence metadata and epistemic status; and a source name and URL.

For example:

    {"id":"example-001","subject":"Example topic","aliases":["example"],"predicate":"definition","object":"A source-backed sentence about the topic.","confidence":0.95,"epistemic_status":"qualified","priority":20,"source":"A vetted source","source_url":"https://example.org/source","tags":["definition"]}

The knowledge store filters candidates by the explicit subject/alias and, for property questions, by an allowed predicate. A color question cannot be answered by returning a size fact. A distance question cannot be answered by returning an unrelated color fact. If no matching record exists, the engine admits that.

The small mcrl/data/facts.jsonl and mcrl/data/seed.jsonl files are optional demonstration material. They are not loaded automatically. Demo documents can be loaded only with --load-demo-documents.

## Topic isolation and dialogue state

- The last topic changes only when the parser creates an explicit topic target or a correctly resolved anaphor.
- Generic words such as “great”, “tell”, “value”, or “sum” are not saved as topics.
- A new explicit topic replaces the previous one.
- A follow-up such as “Is it big?” can use the previous topic because it contains a pronoun in a question construction.
- A new arithmetic request bypasses topic retrieval entirely.
- Topic requests are checked against the current utterance; an old topic is never added just because it was recently discussed.

## Safe arithmetic

Supported examples include:

    python -m mcrl "What is the sum of 4 and 5?"
    python -m mcrl "What is 12 * 7?"

The calculator parses a numeric expression into a Python AST and evaluates only approved constants, unary operators, and arithmetic operators. It applies magnitude and exponent bounds. Unsupported expressions are not executed.

## Quick start

Requires Python 3.10+; no third-party packages.

    git switch semantic-composition-v0.7
    py -3 -m mcrl --interactive
    py -3 -m mcrl --interactive --debug
    py -3 -m mcrl "Can u tell me the value of sum of 4 and 5?"
    py -3 -m unittest discover -s tests -v

On Windows, run these commands from the repository root in PowerShell or CMD.

### Optional demo material

To explicitly load the small project/computing document set for one run:

    py -3 -m mcrl --load-demo-documents --interactive

To persist and use custom documents:

    py -3 -m mcrl --db data/knowledge.sqlite --import-jsonl records.jsonl
    py -3 -m mcrl --db data/knowledge.sqlite --interactive

To persist and use custom atomic facts:

    py -3 -m mcrl --db data/knowledge.sqlite --import-facts-jsonl facts.jsonl
    py -3 -m mcrl --db data/knowledge.sqlite --interactive

Both JSONL import paths stream in batches into SQLite. The database is disk-backed and is not loaded as one large in-memory object.

## Measurement and limitations

- The response-preservation value is a ratio of explicit verification checks. It is not a probability of truth or intelligence.
- Pragmatic candidate scores are weighted constraint-support values, not calibrated probabilities. Counterfactual sensitivity reports relative cue contribution within a candidate.
- The situation model is bounded, session-local interaction state, not a persistent human-like memory or user profile.
- Explicit commands, assistant-directed action requests, standalone exclamations, and conversation-history questions are compiled into interaction plans before fact lookup.
- The source-confidence-metadata value is the minimum confidence metadata of selected facts. It is not statistically calibrated.
- The arithmetic evaluator is a bounded capability, not general mathematical reasoning.
- The parser recognizes a limited set of English constructions. It does not yet provide broad syntax, learned general semantic role labeling, unrestricted coreference, or full negation/tense scope.
- The response planner has a finite set of supported dialogue acts. Unsupported constructions should trigger an honest clarification or limitation, not an invented interpretation.
- The system does not autonomously research, learn, or verify facts from arbitrary raw text.
- No fixed rule system can honestly promise 101% understanding of every possible utterance. The acceptance target is measurable generalization to held-out paraphrases, role reversals, question types, discourse contexts, and unknown-topic cases.

The aim remains CPU-first, meaning-first language composition: understand the current utterance, determine what response it calls for, use evidence only when it is relevant, and be explicit about what is not known. This milestone is a measurable symbolic step toward that goal, not a claim of GPT-level breadth or human-like consciousness. The remaining route to broad conversational competence is to learn compositional language representations from large, diverse, reviewed corpora and evaluate genuine held-out generalization—not to keep growing phrase lists.
