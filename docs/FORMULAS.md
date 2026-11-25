# MCR-L 0.5: active architecture and safeguards

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
