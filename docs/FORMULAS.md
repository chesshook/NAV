# MCR-L 0.3: active mathematical specification

This document describes implemented mechanisms in the semantic-composition branch. It distinguishes hand-authored engineering assumptions from learned or empirically established facts.

## 1. Typed meaning graph

An utterance is mapped to:

\[
G=(V,E,A)
\]

- \(V\): typed nodes for speech acts, events, participants, polarity, affect, time, comparison, and topics.
- \(E\): semantic-role edges such as speaker, addressee, asker, target, agent, patient, has-event, and negates.
- \(A\): node attributes, including the source surface form and parser confidence.

For example, “Are u there?” should create a request.presence_check act, a presence_check event, a user asker, and an assistant target. The system does not derive this from the word “there” alone: it needs a second-person participant and a question-like construction.

## 2. Counterfactual sensitivity

The implementation computes structural utility \(U(G)\) from node kinds and relation edges. For every node \(v_i\), it removes that node and its incident edges:

\[
S_i=\max(0,U(G)-U(G\setminus v_i))
\]

Then:

\[
w_i=\frac{S_i}{\sum_j S_j+\epsilon}
\]

If every loss is zero, the implementation assigns a uniform distribution over the nodes. These weights express the importance of represented graph components under hand-authored utility weights; they are not learned truths about human meaning.

## 3. Response-meaning selection

The planner evaluates response meanings before the grammar engine chooses words:

\[
J(R)=\sum_k W_k P_k(R)+T(R)+B(R)-M(R)
\]

where:
- \(W_k\): aggregated sensitivity for semantic kind \(k\);
- \(P_k(R)\): whether response plan \(R\) preserves that kind;
- \(T(R)\): pragmatic/task fit;
- \(B(R)\): a small candidate utility term;
- \(M(R)\): missing-requirement penalty.

The current supported candidates include reciprocal greeting, reciprocal meeting pleasantry, gratitude acknowledgement, farewell, wellbeing, identity, clarification, and presence confirmation. This is a deliberately finite response-operator set.

## 4. Concept-local lexical selection

The lexical compiler ranks candidates attached to a selected concept, rather than ranking the entire vocabulary:

\[
L^*=\arg\max_{\ell\in Lex(c)}
(0.42P+0.22G+0.20R-0.16A+0.30Q)
\]

\(P\) is semantic precision, \(G\) grammar fit, \(R\) register fit, \(A\) ambiguity penalty, and \(Q\) lexical reuse preference. Candidate values are hand-authored in the current prototype.

## 5. Grammar and meaning verification

The grammar engine constructs a small syntax AST and linearizes it. The resulting text is parsed again. For each supported response operator, required checks are evaluated:

\[
V(Y,R)=\frac{\text{passed required checks}}{\text{number of required checks}}
\]

The CLI's meaning-preservation score is this ratio. It is not a global semantic similarity measure; it is only as strong as the parser and the required checks for that response operator.

## 6. Atomic sourced facts

A semantic fact is stored as:

\[
f=(s,p,o,c,e,r,u,t)
\]

where:
- \(s\): subject;
- \(p\): predicate;
- \(o\): object/value;
- \(c\): manually supplied confidence metadata in \([0,1]\);
- \(e\): epistemic status (established, qualified, tentative);
- \(r\): source/attribution;
- \(u\): source URL;
- \(t\): aliases and tags.

SQLite stores the records on disk. FTS5, when available, generates candidates from subject, aliases, predicate, object, and tags. Candidate priority is explicit metadata, not a learned relevance model.

## 7. Epistemic topic response

The topic responder selects fact predicates according to the user's query properties. For example, a Moon color question favors apparent_color_from_earth and surface_color_up_close; a size follow-up favors diameter_relative_to_earth.

The wording uses a cautious lead-in because the local store is small. Each factual clause corresponds to a selected fact. If no fact and no adequate sourced document record are available, the system explicitly says it lacks reliable local knowledge and does not invent a filler claim.

The CLI's fact-support score is:

\[
C_{\text{answer}}=\min_{f\in F_{\text{selected}}} c_f
\]

It is a simple metadata floor, not calibrated factual probability. The presence of a URL does not alone guarantee source correctness; fact curation remains a human responsibility.

## 8. What remains unsolved

The formulas do not produce concepts, grammar, grounding, or truth criteria by themselves. Broader work is still needed on syntax, morphology, scope, anaphora, logical operators, relation learning, contradiction detection, source freshness, confidence calibration, and held-out evaluation.

A GPU-free runtime is a hardware property—not an accuracy or intelligence benchmark.
