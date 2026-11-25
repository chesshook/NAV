# MCR-L 0.7: compositional interaction and grounded session memory

## 1. Current-utterance semantic graph

An utterance is mapped to:

\[
G=(V,E,A)
\]

- V: nodes for speech acts, events, participants, polarity, affect, temporal/comparative operators, and explicit topic targets.
- E: semantic-role edges such as speaker, addressee, asker, target, agent, patient, has-event, and negates.
- A: source surface forms and parser confidence metadata.

Common chat contractions are normalized before parsing. Topic extraction happens after dialogue construction analysis, so grammatical words and social markers are not mistaken for topic nouns.

## 2. Topic isolation

The engine may read a prior topic only when the current utterance contains an anaphor in a question-like construction and no new explicit topic has been extracted:

\[
T_t =
\begin{cases}
T_{\mathrm{explicit}}, & \text{if current topic is explicit}\\
T_{t-1}, & \text{if current construction contains a resolvable anaphor}\\
\varnothing, & \text{otherwise}
\end{cases}
\]

Ordinary content tokens do not update discourse topic state. Only explicit topic.target or correctly resolved topic.resolved nodes can do so. New unrelated questions therefore cannot inherit an old subject.

## 3. Counterfactual sensitivity

For every semantic node \(v_i\), the current implementation computes structural utility loss after removing the node and incident edges:

\[
S_i=\max(0,U(G)-U(G\setminus v_i))
\]

and normalizes:

\[
w_i=\frac{S_i}{\sum_jS_j+\epsilon}
\]

These are explicit engineering weights, not learned measurements of human attention. The response planner evaluates pragmatic response candidates before lexical realization.

## 4. Dialogue response planning and pragmatics

A single user turn can perform more than one conversational move. The graph therefore retains a primary speech act and any secondary acts rather than overwriting them. For example:

\[
\text{Hi! How is it going?}
\rightarrow
\{\text{social.greeting},\text{social.wellbeing}\}
\]

A separate pragmatic interpreter distinguishes:
- assistant identity requests from user-identity questions;
- ability questions from missing-topic clarification;
- compliments from factual topic mentions;
- greetings and other social functions that coexist in one utterance.

This prevents words such as "am" or "going" from being mistaken for world topics. User-identity answers are bounded by the current conversation; the system does not claim independent knowledge of the person.

The current dialogue solver has explicit operators for:
- greetings and greeting reciprocation;
- meeting pleasantries, gratitude, and farewell;
- wellbeing and presence checks;
- incomplete-question clarification;
- positive acknowledgement and empathetic acknowledgement;
- compliment acknowledgement;
- supported and unsupported capability checks;
- assistant and user identity questions;
- respecting a request to avoid a topic.

Each operator defines required semantic effects; a grammar AST realizes the selected response meaning. This remains a bounded pragmatic recognizer, not a general language learner.

## 4.1 Interactional pragmatics

A single user turn can perform multiple moves. The graph retains the primary speech act and secondary acts rather than discarding all but the first. For example:

\[
\text{Hi! How is it going?}
\rightarrow
\{\text{social.greeting},\text{social.wellbeing}\}
\]

The separate pragmatic interpreter distinguishes:
- assistant identity questions from user-identity questions;
- ability checks from incomplete-topic clarification;
- compliments from factual topic mentions;
- social moves that coexist in one utterance.

This prevents words such as "am" and "going" from being mistaken for factual topics. Answers about the user are bounded by what the user shared in the current conversation; the system does not claim independent identity knowledge.

The response planner now handles compound greeting/wellbeing turns, compliments, identity questions, and supported or unsupported capability checks. These remain bounded constructions, not broad learned language understanding.

## 5. Safe arithmetic capability

Natural-language arithmetic is first reduced to a small expression grammar. The expression is parsed into an AST. Only numeric constants, bounded unary operations, and whitelisted arithmetic operators are accepted. There is no call to eval, exec, attributes, function calls, imports, or arbitrary Python code. Magnitude, exponent, and division-by-zero checks are enforced.

Arithmetic returns before fact retrieval or topic-state updates. This prevents a calculation such as “sum of 4 and 5” from being confused with the last subject discussed.

## 6. Optional knowledge retrieval

The knowledge store has separate document and atomic-fact tables. Neither is automatically seeded by the engine.

For a fact request:
1. The parser must provide an explicit current topic or a legitimate resolved anaphor.
2. Candidate facts are searched.
3. The candidate's subject or alias must contain the full normalized topic term set.
4. When the question requests a property, the predicate must belong to that property class.
5. If no exact topic/property fact exists, the system must abstain from factual assertion.

For document fallback, every topic term must match the candidate's title/tags as well as its indexed content. Property-specific questions do not use a generic paragraph as fallback.

The subject check is important: FTS is a candidate generator, not an authority that can assign a topic to the question. The current utterance supplies the topic; fact records cannot override it.

## 7. Honest unknown response

For a topic with no relevant source:
- retain the topic named by the current request;
- optionally name the missing property;
- admit that the local system lacks reliable information;
- do not add a plausible-sounding filler claim;
- do not copy a fact from a previously discussed subject.

This is a capability boundary, not a factual answer. Its status is shown as knowledge-match=none, not a false confidence score.

## 8. What is not achieved

The system still does not have a broad, learned language parser, an open-domain world model, general factual reasoning, robust compositional logic, or universal paraphrase understanding. The rules are bounded and must be tested against held-out language. A CPU-only design is a deployment property, not proof of intelligence.


## 9. Weighted interpretation hypotheses

The pragmatic interpreter evaluates a small set of candidate semantic hypotheses independently. Each candidate is grounded in typed cues such as a participant reference, an interrogative construction, an ability frame, a polarity marker, or a positive attribution.

For hypothesis h:

[
C(h)=operatorname{clip}_{[0,1]}
left(
rac{sum_i alpha_i x_i}{sum_i alpha_i}
-lambda K(h)
ight)
]

- (x_iin[0,1]) is the observed activation of cue i.
- (alpha_igeq 0) is the declared importance of that cue for the candidate.
- (K(h)) is an explicit contradiction penalty.
- (lambda) controls the penalty strength.

Candidate generation also applies hard structural gates. For example, a compliment hypothesis requires a positive attribution aimed at the addressee and is rejected when the clause is negated. An identity question has separate targets for the user and the assistant.

The score is **constraint support**, not a calibrated probability, certainty, or general intelligence measure.

## 10. Counterfactual cue sensitivity

Each cue is tested by removing its activation while holding the other evidence fixed:

[
Delta_i(h)=max(0,C(h)-C(hmid x_ileftarrow 0))
]

The normalized sensitivity distribution is:

[
s_i(h)=rac{Delta_i(h)}{sum_jDelta_j(h)+epsilon}
]

This answers a concrete question: which observed semantic cue made this interpretation stronger? It does not claim the cue is universally important or that the interpretation is correct in every context. A weak margin between competing identity hypotheses is treated as ambiguity rather than silently forcing one target.

## 11. Dialogue situation state

A bounded session state records:
- the ordered user moves within the latest turn;
- the current conversational goal;
- speaker and addressee roles;
- the active explicit topic;
- interaction tone;
- the assistant action and outcome;
- a short, bounded history of recent turns.

Non-topic moves do not erase the current topic. A social or identity question cannot overwrite an unrelated active referent, and a participant question cannot create a factual topic such as the token "am". The situation record contains only what the parser explicitly represented; it does not infer personal identity.

## 12. Response path and generation boundary

The v0.6 path is:

[
	ext{utterance}
ightarrow 	ext{typed cues}
ightarrow 	ext{candidate interpretations}
ightarrow 	ext{constraint support and sensitivity}
ightarrow 	ext{meaning graph}
ightarrow 	ext{response plan}
ightarrow 	ext{grammar realization}
ightarrow 	ext{verification}
]

The response path is meaning-first rather than autoregressive next-token selection. The current lexicon and parser are still bounded, and the surface grammar still has finite constructions. This is a deliberate research boundary: the milestone removes token-by-token response framing for its supported constructions; it does not solve open-domain language understanding or unrestricted sentence generation.


## 13. Interaction plan as an operation on the dialogue

Some user messages do not request a fact or a conventional speech-act response. They request an operation on the interaction itself:
- emit a user-supplied string;
- perform a symbolic expression;
- explain a physical capability boundary;
- summarize the current session;
- acknowledge a standalone interjection.

The interaction planner compiles these into:

[
P=(o,p,a,mathcal{C},mathcal{E},q)
]

where (o) is the operation, (p) the payload, (a) the requested action, (mathcal{C}) the constraints, (mathcal{E}) the structural evidence, and (q) the precedence.

Candidate operations are resolved before factual retrieval. Explicit user-supplied speech output has higher precedence than an embedded question whose disclosure was explicitly prohibited. Thus a turn such as "don't tell me who you are, just say Hello" is interpreted as a negative constraint on identity disclosure plus a positive speech-output directive. The requested payload is emitted, not answered as an identity question.

An assistant-directed action frame such as "Can you smile for me?" is interpreted as an action predicate with an addressee and beneficiary. It is not turned into a subject entity called "smile". When the action can only be represented symbolically, the response states or shows that limitation rather than claiming physical embodiment.

## 14. Evidence-bounded conversation recap

Each bounded turn record stores the actual user utterance, assistant response, inferred user moves, conversational goal, topic, and action outcome. A recap is composed from these records:

[
H_t = [(u_1,r_1),ldots,(u_t,r_t)]
]

[
operatorname{recap}(H_t)=operatorname{linearize}
left(
operatorname{select}_{k}(H_t)
ight)
]

The recap selects only the most recent recorded turns within the configured history limit. It does not use the world-knowledge database and does not invent unrecorded facts. The explicit transcript-style fallback is intentionally evidence-bounded.

## 15. Architecture boundary

The current engine has two kinds of mathematical structure:
1. explicit semantic constraint scoring and counterfactual sensitivity;
2. symbolic composition of interaction operations and bounded conversational history.

Neither one is, by itself, a learned general language model. A system with broad, GPT-like conversational competence requires learned representations from diverse language data, broad compositional coverage, and an evaluator that tests unseen paraphrases and contexts. No hand-authored rule count or low latency can substitute for that training and evaluation. The next research stage must focus on data-driven abstraction rather than expanding the phrase inventory.
