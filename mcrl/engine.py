"""MCR-L 0.4: dialogue competence, safe capabilities, and optional grounded knowledge.

The current utterance determines the route. Arithmetic is solved independently;
dialogue acts are composed from meaning; optional fact lookup requires an explicit
current topic plus a matching property. No knowledge domain is preloaded by default.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .calculator import try_calculate
from .discourse_state import DiscourseState
from .grammar_engine import GrammarEngine
from .knowledge import KnowledgeStore
from .meaning_solver import solve_response
from .meaning_validator import validate_response
from .semantic_parser import parse_meaning
from .sensitivity import calculate_sensitivity
from .topic_responder import compose_topic_answer
from .pragmatics import analyze_pragmatics
from .situation_model import DialogueSituation


@dataclass
class Answer:
    text: str
    confidence: float
    abstained: bool
    selected_title: str = ""
    source: str = ""
    diagnostics: dict = field(default_factory=dict)


PROPERTY_CUES = {
    "distance": {"distance", "far", "away", "near"},
    "color": {"color", "colour", "white", "gray", "grey"},
    "size": {"size", "big", "large", "wide", "diameter", "small"},
    "population": {"population"},
    "capital": {"capital"},
    "definition": {"meaning", "definition"},
}


class MCRLanguageEngine:
    def __init__(self, store: KnowledgeStore, minimum_evidence: float = 0.68) -> None:
        self.store = store
        self.minimum_evidence = minimum_evidence
        self.minimum_score = minimum_evidence  # compatibility name
        self.last_plan = None
        self.discourse = DiscourseState()
        self.situation = DialogueSituation()
        self.grammar = GrammarEngine()
        # Deliberately do not load mcrl/data/facts.jsonl here. Knowledge is
        # optional input supplied by the operator, not hard-coded personality.

    @staticmethod
    def _topic_from_graph(graph) -> str:
        nodes = [
            node.value for node in graph.by_kind("topic")
            if node.concept in {"topic.target", "topic.resolved"}
        ]
        return nodes[0] if nodes else ""

    @staticmethod
    def _requested_property(graph) -> str | None:
        words = set(graph.tokens)
        for property_name, cues in PROPERTY_CUES.items():
            if words & cues:
                return property_name
        if {"how", "far"} <= words:
            return "distance"
        return None

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

    def _factual_document_candidate(self, topic: str):
        """Optional document fallback with exact topic coverage, never query-word OR."""
        topic_terms = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", topic.lower()))
        if not topic_terms:
            return None, 0.0, {"reason": "no explicit topic"}
        records = self.store.search(" ".join(sorted(topic_terms)), limit=80)
        ranked = []
        for record in records:
            title_and_tags = set(
                re.findall(
                    r"[a-z0-9]+(?:-[a-z0-9]+)*",
                    " ".join(
                        [
                            record.get("title", ""),
                            " ".join(record.get("keywords", [])),
                            " ".join(record.get("tags", [])),
                        ]
                    ).lower(),
                )
            )
            all_terms = self._record_terms(record)
            title_coverage = len(topic_terms & title_and_tags) / len(topic_terms)
            coverage = len(topic_terms & all_terms) / len(topic_terms)
            # Every topic term must be present; one stray word is not a subject match.
            if coverage < 1.0 or title_coverage < 1.0:
                continue
            if not record.get("answer") or not record.get("source"):
                continue
            score = 0.90 * title_coverage + 0.10 * (1.0 if record.get("source") else 0.0)
            ranked.append((score, record))
        if not ranked:
            return None, 0.0, {"reason": "no exact-topic sourced document", "topic_terms": sorted(topic_terms)}
        ranked.sort(key=lambda row: row[0], reverse=True)
        score, record = ranked[0]
        return record, score, {
            "reason": "exact-topic document",
            "topic_terms": sorted(topic_terms),
            "score": round(score, 4),
            "source_present": True,
        }

    def _answer_dialogue(self, graph, sensitivity) -> Answer | None:
        response = solve_response(graph, sensitivity)
        if response is None:
            return None
        text, ast = self.grammar.realize(response, graph)
        validation = validate_response(response, graph, text)
        if response.act == "acknowledge_topic_avoidance":
            self.discourse.avoid_topic(str(response.recipient))
        self.discourse.observe_graph(graph)
        self.situation.observe(
            graph,
            response_act=response.act,
            outcome="answered" if validation.valid else "response_validation_failed",
        )
        pragmatic_diagnostics = analyze_pragmatics(tuple(graph.tokens)).diagnostics()
        if not validation.valid:
            return Answer(
                text="I understood the broad conversational intent, but my response wording failed its meaning check. I should not present it as a successful answer.",
                confidence=validation.preservation,
                abstained=True,
                diagnostics={
                    "mode": "dialogue_composition",
                    "graph": graph.summary(),
                    "response_plan": response.__dict__,
                    "grammar_ast": ast.__dict__,
                    "verification": validation.diagnostics(),
                    "discourse": self.discourse.snapshot(),
                    "pragmatic_interpretation": pragmatic_diagnostics,
                    "situation": self.situation.snapshot(),
                },
            )
        return Answer(
            text=text,
            confidence=validation.preservation,
            abstained=False,
            selected_title=f"Composed dialogue act: {response.act}",
            source="MCR-L semantic response planner and grammar",
            diagnostics={
                "mode": "dialogue_composition",
                "intent": graph.speech_act,
                "graph": graph.summary(),
                "sensitivity": sensitivity.diagnostics(),
                "response_plan": {
                    "act": response.act,
                    "recipient": response.recipient,
                    "requirements": response.requirements,
                    "utility": round(response.utility, 5),
                    "score_components": dict(response.diagnostics or {}),
                },
                "grammar_ast": ast.__dict__,
                "verification": validation.diagnostics(),
                "discourse": self.discourse.snapshot(),
                "pragmatic_interpretation": pragmatic_diagnostics,
                "situation": self.situation.snapshot(),
            },
        )

    def answer(self, utterance: str) -> Answer:
        # Deterministic capabilities are tried before open-topic fallback. They
        # never query the knowledge base and therefore cannot inherit an old topic.
        calculation = try_calculate(utterance)
        if calculation is not None:
            self.situation.record_action("bounded_arithmetic", "completed")
            return Answer(
                text=calculation.text(),
                confidence=1.0,
                abstained=False,
                selected_title="Safe arithmetic evaluation",
                source="MCR-L bounded arithmetic evaluator",
                diagnostics={
                    "mode": "calculation",
                    "expression": calculation.expression,
                    "operation": calculation.operation,
                    "result": calculation.result,
                    "situation": self.situation.snapshot(),
                },
            )

        graph = parse_meaning(utterance, self.discourse)
        sensitivity = calculate_sensitivity(graph)

        dialogue_answer = self._answer_dialogue(graph, sensitivity)
        if dialogue_answer is not None:
            return dialogue_answer

        topic = self._topic_from_graph(graph)
        requested_property = self._requested_property(graph)
        factual_intent = graph.intent in {
            "topic_information", "define", "explain", "question",
        }

        if factual_intent:
            if not topic:
                self.discourse.observe_graph(graph)
                self.situation.observe(graph, "ask_for_subject", "clarification_requested")
                return Answer(
                    text="I can tell you're asking something, but I can't identify the subject yet. What thing are you asking about?",
                    confidence=0.0,
                    abstained=False,
                    selected_title="Clarification: missing topic",
                    diagnostics={
                        "mode": "topic_clarification",
                        "graph": graph.summary(),
                        "discourse": self.discourse.snapshot(),
                    },
                )

            # Only an explicit or correctly resolved current-turn topic can
            # reach the fact store. A previous topic never supplies a new subject.
            topic_terms = re.findall(r"[a-z0-9]+", topic.lower())
            allowed_predicates = None
            if requested_property:
                # If no stored fact supports this property, do not substitute
                # another fact about the same subject.
                from .topic_responder import PROPERTY_PREDICATES
                allowed_predicates = PROPERTY_PREDICATES.get(requested_property, set())
            facts = self.store.search_facts(
                topic_terms,
                subject=topic,
                predicates=allowed_predicates,
                limit=40,
            ) if topic_terms else []

            if facts:
                composed = compose_topic_answer(
                    topic,
                    facts,
                    query_terms=graph.tokens,
                    requested_property=requested_property,
                )
                self.discourse.observe_graph(graph)
                self.discourse.remember_topic(topic)
                self.situation.observe(graph, "answer_source_backed_facts", "answered")
                sources = sorted({fact["source"] for fact in composed.selected_facts})
                return Answer(
                    text=composed.text,
                    confidence=composed.support_score,
                    abstained=False,
                    selected_title=f"Source-backed partial answer about {composed.topic}",
                    source="; ".join(sources),
                    diagnostics={
                        "mode": "fact_composition",
                        "graph": graph.summary(),
                        "sensitivity": sensitivity.diagnostics(),
                        "topic_answer": composed.diagnostics(),
                        "discourse": self.discourse.snapshot(),
                    },
                )

            # Optional prose records must match every explicit topic term in
            # their title/tags. Property-specific questions do not use generic
            # paragraphs because they may answer the wrong attribute.
            record, score, evidence = (
                (None, 0.0, {"reason": "property-specific fact unavailable"})
                if requested_property
                else self._factual_document_candidate(topic)
            )
            if record is not None:
                self.discourse.observe_graph(graph)
                self.discourse.remember_topic(topic)
                self.situation.observe(graph, "answer_source_backed_document", "answered")
                return Answer(
                    text=record["answer"].strip(),
                    confidence=score,
                    abstained=False,
                    selected_title=record["title"],
                    source=record["source"],
                    diagnostics={
                        "mode": "evidence_backed_document",
                        "graph": graph.summary(),
                        "sensitivity": sensitivity.diagnostics(),
                        "evidence": evidence,
                        "discourse": self.discourse.snapshot(),
                    },
                )

            composed = compose_topic_answer(
                topic,
                (),
                query_terms=graph.tokens,
                requested_property=requested_property,
            )
            self.discourse.observe_graph(graph)
            self.discourse.remember_topic(topic)
            self.situation.observe(graph, "admit_missing_evidence", "abstained")
            return Answer(
                text=composed.text,
                confidence=0.0,
                abstained=True,
                selected_title=f"No matching evidence for {composed.topic}",
                source="No matching, topic-specific source in the local store",
                diagnostics={
                    "mode": "unknown_topic_or_property",
                    "graph": graph.summary(),
                    "sensitivity": sensitivity.diagnostics(),
                    "topic_answer": composed.diagnostics(),
                    "evidence": evidence,
                    "discourse": self.discourse.snapshot(),
                },
            )

        if graph.speech_act == "request.avoid_topic":
            # This should have been handled by solve_response; defensive fallback.
            topic = topic or "that topic"
            self.discourse.avoid_topic(topic)
            self.discourse.observe_graph(graph)
            self.situation.observe(graph, "respect_topic_boundary", "acknowledged")
            return Answer(
                text=f"Got it — I won't bring up {topic} unless you ask.",
                confidence=1.0,
                abstained=False,
                selected_title="Acknowledged topic boundary",
                diagnostics={"mode": "topic_boundary", "discourse": self.discourse.snapshot()},
            )

        self.discourse.observe_graph(graph)
        if graph.speech_act == "assertion":
            self.situation.observe(graph, "continue_conversation", "continued")
            return Answer(
                text="Got you. I may be missing some context, though—what would you like to explore about that?",
                confidence=0.0,
                abstained=False,
                selected_title="Conversational continuation",
                diagnostics={
                    "mode": "conversation_continuation",
                    "graph": graph.summary(),
                    "sensitivity": sensitivity.diagnostics(),
                    "discourse": self.discourse.snapshot(),
                },
            )

        self.situation.observe(graph, "request_clarification", "clarification_requested")
        return Answer(
            text="I'm not fully following what you're asking yet. Could you say a little more about what you mean?",
            confidence=0.0,
            abstained=True,
            selected_title="Clarification needed",
            diagnostics={
                "mode": "unsupported_semantic_structure",
                "graph": graph.summary(),
                "sensitivity": sensitivity.diagnostics(),
                "discourse": self.discourse.snapshot(),
            },
        )
