# MCR-L 0.1: mathematical specification

This document describes the implemented prototype, not a claim of general intelligence.

## 1. Semantic feature vector

The parser maps an input string to a bounded vector:

\[
x_t \in [0,1]^N, \quad N=16
\]

The dimensions are: question, define, explain, compare, how-to, entity, action, causality, temporal, negation, constraint, uncertainty, evidence, risk, context, follow-up.

The current parser uses explicit English cue rules. It is intentionally inspectable and has limited coverage.

## 2. Recurrent situation update

The implemented recurrence is:

\[
a_{t+1}=\rho a_t+(1-\rho)\sigma(b+g_x x_t+g_w W^T a_t)
\]

with:

- \(\rho=0.56\), state retention;
- \(g_x=3.45\), input gain;
- \(g_w=1.15\), recurrent gain;
- \(b=-1.45\), baseline bias;
- \(\sigma(z)=1/(1+e^{-z})\);
- activations clipped to \([0,1]\).

The sparse signed matrix \(W\) contains hand-authored starter relationships. These are hypotheses to test, not discovered universal language laws.

## 3. Evidence retrieval

The knowledge store is SQLite. When FTS5 exists, a query is issued against a full-text index. Otherwise, the prototype uses SQL LIKE matching. JSONL ingestion is streaming and batched.

The current implementation is lexical retrieval, not vector/embedding search. It does not claim to understand arbitrary paraphrases.

## 4. Answer-plan score

Each retrieved record receives:

\[
J(p)=0.40C+0.22T+0.14I+0.12E+0.12S
\]

where:

- \(C\): fraction of query content terms found in the record;
- \(T\): fraction of query content terms found in the record title;
- \(I\): intent compatibility;
- \(E\): whether a source is recorded;
- \(S\): mean recurrent activation for the record's tagged semantic features.

The score is a ranking heuristic, not a calibrated probability. The CLI's confidence value must not be interpreted as a statistical probability that the answer is true.

## 5. Sensitivity

For a selected plan score function \(J(a)\), the prototype computes finite differences:

\[
d_i=\frac{|J(a)-J(a-\epsilon e_i)|}{\epsilon}
\]

and normalizes the values to sum to one when their sum is nonzero. This reports which activations influence the current plan score. It does not prove that those signals are causally important in the real world.

## 6. Feedback plasticity

Only an explicit /good or /bad command invokes the bounded update:

\[
\Delta W_{ij}=\eta r(a_i^- - 0.38)(a_j^+ - 0.38)-0.0002|W_{ij}|
\]

Weights are clipped to \([-0.48,1.18]\). The update is a toy online adaptation mechanism. In 0.1, weights live only in the current process and are not persisted or trained on a validated dataset.

## 7. What this does not solve

The equations do not themselves supply the semantic features, facts, grounding, or evaluation criteria. Those are supplied by the parser, the curated database, source metadata, and tests. This release is a reproducible baseline for experimentation, not an established alternative to foundation models.
