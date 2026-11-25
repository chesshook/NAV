"""MCR-L 0.2 engine: meaning -> sensitivity -> response plan -> grammar -> verify.

Knowledge retrieval is a separate, evidence-gated path for factual questions. It is
never used to answer social turns or to select arbitrary words in a sentence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .discourse_state import DiscourseState
from .grammar_engine import GrammarEngine
from .knowledge import KnowledgeStore
from .meaning_solver import solve_response
from .meaning_validator import validate_response
from .semantic_parser import parse_meaning
from .sensitivity import calculate_sensitivity


@dataclass
class Answer:
    text: str
    confidence: float
    abstained: bool
    selected_title: str = ""
    source: str = ""
    diagnostics: dict = field(default_factory=dict)


class MCRLanguageEngine:
    def __init__(self, store: KnowledgeStore, minimum_evidence: float = 0.68) -> None:
        self.store = store
        self.minimum_evidence = minimum_evidence
        self.discourse = DiscourseState()
        self.grammar = GrammarEngine()

    @staticmethod
    def _record_terms(record: dict) -> set[str]:
        fields = [
            record.get("title", ""),
            record.get("text", ""),
            record.get("answer", ""),
            " ".join(record.get("keywords", [])),
            " ".join(record.get("tags", [])),
        ]
        return set(re.findall(r"[a-z0-9]+", " ".join(fields).lower()))

    def _factual_candidate(self, graph):
        query_terms = set(graph.lexical_tokens)
        if len(query_terms) < 2:
            return None, 0.0, {"reason": "not enough distinct content terms"}

        records = self.store.search(" ".join(sorted(query_terms)), limit=80)
        ranked = []
        for record in records:
            record_terms = self._record_terms(record)
            title_terms = set(re.findall(r"[a-z0-9]+", record.get("title", "").lower()))
            coverage = len(query_terms & record_terms) / max(1, len(query_terms))
            title_coverage = len(query_terms & title_terms) / max(1, len(query_terms))
            source_quality = 1.0 if record.get("source") else 0.0
            score = 0.72 * coverage + 0.18 * title_coverage + 0.10 * source_quality
            ranked.append((score, coverage, record))

        if not ranked:
            return None, 0.0, {"reason": "no indexed candidates", "candidates": 0}
        ranked.sort(key=lambda item: item[0], reverse=True)
        score, coverage, record = ranked[0]
        diagnostics = {
            "candidates": len(ranked),
            "term_coverage": round(coverage, 4),
            "title_coverage": round(
                len(query_terms & set(re.findall(r"[a-z0-9]+", record.get("title", "").lower())))
                / max(1, len(query_terms)),
                4,
            ),
            "source_present": bool(record.get("source")),
            "score": round(score, 4),
        }
        if coverage < 0.70 or score < self.minimum_evidence or not record.get("source") or not record.get("answer"):
            return None, score, diagnostics
        return record, score, diagnostics

    def answer(self, utterance: str) -> Answer:
        graph = parse_meaning(utterance, self.discourse)
        sensitivity = calculate_sensitivity(graph)

        # The response planner sees meaning and its counterfactual importance
        # before a single output word is selected.
        response = solve_response(graph, sensitivity)
        if response is not None:
            text, ast = self.grammar.realize(response, graph)
            validation = validate_response(response, graph, text)
            self.discourse.observe_graph(graph)
            if not validation.valid:
                return Answer(
                    text="I formed a response plan, but its wording did not preserve enough of the intended meaning. I am abstaining rather than presenting it as correct.",
                    confidence=validation.preservation,
                    abstained=True,
                    diagnostics={
                        "mode": "semantic_composition",
                        "graph": graph.summary(),
                        "sensitivity": sensitivity.diagnostics(),
                        "response_plan": response.__dict__,
                        "grammar_ast": ast.__dict__,
                        "verification": validation.diagnostics(),
                        "discourse": self.discourse.snapshot(),
                    },
                )
            return Answer(
                text=text,
                confidence=validation.preservation,
                abstained=False,
                selected_title="Composed response meaning: " + response.act,
                source="MCR-L semantic composition and grammar rules",
                diagnostics={
                    "mode": "semantic_composition",
                    "graph": graph.summary(),
                    "sensitivity": sensitivity.diagnostics(),
                    "response_plan": {
                        "act": response.act,
                        "predicate": response.predicate,
                        "affect": response.affect,
                        "recipient": response.recipient,
                        "reciprocal": response.reciprocal,
                        "requirements": response.requirements,
                        "utility": round(response.utility, 5),
                    },
                    "grammar_ast": ast.__dict__,
                    "verification": validation.diagnostics(),
                    "discourse": self.discourse.snapshot(),
                },
            )

        # Factual answers remain on a separate evidence path. The candidate
        # index is only a lookup accelerator; the full term-coverage gate below
        # prevents a single accidental token hit from selecting an answer.
        if graph.intent in {"define", "explain", "question"}:
            record, score, evidence = self._factual_candidate(graph)
            self.discourse.observe_graph(graph)
            if record is not None:
                return Answer(
                    text=record["answer"].strip(),
                    confidence=score,
                    abstained=False,
                    selected_title=record["title"],
                    source=record["source"],
                    diagnostics={
                        "mode": "evidence_backed_retrieval",
                        "graph": graph.summary(),
                        "sensitivity": sensitivity.diagnostics(),
                        "evidence": evidence,
                        "discourse": self.discourse.snapshot(),
                    },
                )
            return Answer(
                text="I can represent the question, but my local knowledge index does not contain enough matching, sourced evidence to answer it reliably.",
                confidence=score,
                abstained=True,
                diagnostics={
                    "mode": "evidence_gate",
                    "graph": graph.summary(),
                    "sensitivity": sensitivity.diagnostics(),
                    "evidence": evidence,
                    "discourse": self.discourse.snapshot(),
                },
            )

        self.discourse.observe_graph(graph)
        return Answer(
            text=(
                "I have represented parts of that message, but this build does not yet "
                "have a validated composition rule for its speech act. I will not fill "
                "the gap with a guessed or retrieved sentence."
            ),
            confidence=0.0,
            abstained=True,
            diagnostics={
                "mode": "unsupported_semantic_structure",
                "graph": graph.summary(),
                "sensitivity": sensitivity.diagnostics(),
                "discourse": self.discourse.snapshot(),
            },
        )
