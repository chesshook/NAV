# MCR-L 0.2: active mathematical specification

This document describes the semantic-composition branch. It distinguishes implemented formulas from research goals. It does not claim general intelligence.

## 1. Typed meaning graph

An utterance is mapped to:

\[
G=(V,E,A)
\]

- \(V\): typed meaning nodes (speech act, event, participant, polarity, affect, time, comparison, topic).
- \(E\): typed role edges (speaker, addressee, has-event, negates, temporal-scope, comparison).
- \(A\): node attributes such as surface form and parser confidence.

The graph is the working representation. The original word sequence is retained as evidence for lexicalization and factual lookup, not used as the sole semantic representation.

## 2. Counterfactual sensitivity

The implementation computes a structural utility \(U(G)\) from represented node kinds and relation edges. For each node \(v_i\), it removes that node and its incident edges:

\[
S_i=\max(0,U(G)-U(G\setminus v_i))
\]

Then it normalizes:

\[
w_i=\frac{S_i}{\sum_j S_j+\epsilon}
\]

where \(\epsilon\) prevents division by zero. If all losses are zero, the implementation assigns a uniform distribution over nodes.

These weights indicate which represented components are structurally important to the current graph. They are not learned truths about human meaning; the category and edge weights are explicit engineering hypotheses.

## 3. Response-meaning selection

The response planner constructs semantic response candidates before the grammar engine selects words. Its candidate score is:

\[
J(R)=\sum_k W_k P_k(R)+T(R)+B(R)-M(R)
\]

where:
- \(W_k\) is the aggregated sensitivity for a semantic kind;
- \(P_k(R)\) is whether candidate \(R\) preserves that kind;
- \(T(R)\) is task/pragmatic fit;
- \(B(R)\) is a small candidate utility bias;
- \(M(R)\) is a missing-requirement penalty.

For a positive meeting pleasantry, a reciprocal-meeting response preserves the event, positive affect, participant roles, and social act. A generic acknowledgement preserves fewer of those details and should therefore score lower.

## 4. Concept-local lexical selection

The lexical compiler does not rank every word in the vocabulary. It ranks candidates attached to one concept:

\[
L^*=\arg\max_{\ell\in Lex(c)}
(0.42P+0.22G+0.20R-0.16A+0.30Q)
\]

where:
- \(P\): semantic precision;
- \(G\): grammar fit;
- \(R\): register fit;
- \(A\): ambiguity penalty;
- \(Q\): lexical reuse bonus when the input's modifier is a suitable output choice.

The candidate set for the positive-meeting modifier is currently a short hand-authored list (nice, glad, pleased, delighted, happy). It is not a trained language model.

## 5. Grammar and verification

The grammar engine builds a small syntax AST from semantic roles and linearizes it. The result is parsed again. For each response plan, required semantic checks are evaluated:

\[
V(Y,R)=\frac{\text{passed required checks}}{\text{number of required checks}}
\]

A response is emitted as successful only when every current required check passes. This verifier is limited by the parser and check list; it cannot guarantee that a sentence is correct in every broader linguistic context.

## 6. Separate factual evidence score

Factual questions use the disk-backed knowledge store only as a candidate generator. For candidate record \(d\):

\[
E(d)=0.72C+0.18T+0.10S
\]

- \(C\): fraction of distinct query content terms appearing in record title/text/answer/keywords/tags;
- \(T\): fraction of query content terms appearing in the title;
- \(S\): 1 when a source is recorded, otherwise 0.

The candidate must also have at least 0.70 term coverage, a non-empty source, a non-empty answer, and an overall score at or above the configured threshold. This is a heuristic gate, not a probability calibration or full entailment checker.

## 7. Legacy 0.1 mechanisms

The 0.1 recurrent feature vector, evidence-plan score, and finite-difference score sensitivity are retained in older modules for comparison, but they are not on the 0.2 active answer path. The current active sensitivity is counterfactual ablation of meaning-graph nodes.

## 8. What remains unsolved

The formulas do not produce concepts, grammar, grounding, or truth criteria by themselves. This branch still needs wider lexical coverage, morphology, robust syntactic role assignment, entity linking, reference resolution, compositional operators, typed factual relations, contradiction checks, and held-out evaluation.

A GPU-free runtime is a hardware constraint—not an accuracy or intelligence benchmark.
