"""MCR-L 0.3: meaning -> sensitivity -> plan -> realization -> verification.

Conversational acts use semantic composition. Topic questions use atomic, sourced
facts and a separate epistemic response composer. Longer factual explanations can
still use the document index, with a strict evidence gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .discourse_state import DiscourseState
from .grammar_engine import GrammarEngine
from .knowledge import KnowledgeStore
from .meaning_solver import solve_response
from .meaning_validator import validate_response
from .semantic_parser import parse_meaning
from .sensitivity import calculate_sensitivity
from .topic_responder import compose_topic_answer


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
        self.minimum_score = minimum_evidence
        self.last_plan = None
        self.discourse = DiscourseState()
        self.grammar = GrammarEngine()
        self._load_builtin_facts()

    def _load_builtin_facts(self) -> None:
        """Load the small, source-attributed fact seed idempotently."""
        fact_file = Path(__file__).resolve().parent / "data" / "facts.jsonl"
        if fact_file.is_file():
            self.store.import_facts_jsonl(str(fact_file))

    @staticmethod
    def _record_terms(record: dict) -> set[str]:
        fields = [
            record.get("title", ""),
            record.get("text", ""),
            record.get("answer", ""),
            " ".join(record.get("keywords", [])),
            " ".join(record.get("tags", [])),
        ]
        return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", " ".join(fields).lower()))

    @staticmethod
    def _topic_terms(graph) -> list[str]:
        values = [
            node.value for node in graph.nodes
            if node.kind == "topic"
            or (node.kind == "entity" and node.value.lower() not in {"mcr", "mcr-l", "assistant", "chatgpt"})
        ]
        values.extend(graph.lexical_tokens)
        result: list[str] = []
        for value in values:
            for term in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", str(value).lower()):
                if len(term) > 1 and term not in result:
                    result.append(term)
        return result

    @staticmethod
    def _topic_label(graph, facts: list[dict], topic_terms: list[str]) -> str:
        if facts:
            return str(facts[0]["subject"])
        nodes = [node.value for node in graph.nodes if node.kind in {"topic", "entity"}]
        ignored = {"mcr", "mcr-l", "assistant", "chatgpt"}
        for value in nodes:
            if value.lower() not in ignored:
                return value
        return topic_terms[0] if topic_terms else ""

    def _factual_candidate(self, graph):
        query_terms = set(graph.lexical_tokens)
        if len(query_terms) < 1:
            return None, 0.0, {"reason": "not enough distinct content terms"}

        records = self.store.search(" ".join(sorted(query_terms)), limit=80)
        ranked = []
        for record in records:
            record_terms = self._record_terms(record)
            title_terms = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", record.get("title", "").lower()))
            coverage = len(query_terms & record_terms) / max(1, len(query_terms))
            title_coverage = len(query_terms & title_terms) / max(1, len(query_terms))
            if len(query_terms) == 1 and title_coverage < 1.0:
                continue
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
                len(query_terms & set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", record.get("title", "").lower())))
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

        # Presence, greetings, gratitude, identity and other supported pragmatic
        # acts are solved from the input meaning, not from document retrieval.
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
                    "intent": graph.speech_act.split(".")[-1],
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

        query_terms = self._topic_terms(graph)
        factual_intent = graph.intent in {
            "topic_information", "define", "explain", "question",
        }
        if factual_intent and query_terms:
            facts = self.store.search_facts(query_terms, limit=40)
            topic = self._topic_label(graph, facts, query_terms)
            topic_terms = set(graph.lexical_tokens) | set(query_terms)
            topic_terms -= {"tell", "know", "about", "anything", "something", "facts"}
            if facts:
                composed = compose_topic_answer(topic, facts, topic_terms)
                self.discourse.observe_graph(graph)
                self.discourse.remember_topic(topic)
                source_names = sorted({fact["source"] for fact in composed.selected_facts})
                return Answer(
                    text=composed.text,
                    confidence=composed.support_score,
                    abstained=False,
                    selected_title=f"Composed partial knowledge about {composed.topic}",
                    source="; ".join(source_names),
                    diagnostics={
                        "mode": "fact_composition",
                        "graph": graph.summary(),
                        "sensitivity": sensitivity.diagnostics(),
                        "topic_answer": composed.diagnostics(),
                        "discourse": self.discourse.snapshot(),
                    },
                )

            record, score, evidence = self._factual_candidate(graph)
            self.discourse.observe_graph(graph)
            self.discourse.remember_topic(topic)
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

            composed = compose_topic_answer(topic, (), topic_terms)
            return Answer(
                text=composed.text,
                confidence=0.0,
                abstained=True,
                selected_title=f"No sourced facts indexed for {composed.topic}",
                source="Local fact store has no matching entry",
                diagnostics={
                    "mode": "unknown_topic",
                    "graph": graph.summary(),
                    "sensitivity": sensitivity.diagnostics(),
                    "topic_answer": composed.diagnostics(),
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
