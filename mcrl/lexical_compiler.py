"""Concept-local lexical selection; never scores the entire vocabulary.

The compiler only considers words attached to a meaning node and ranks them by
semantic precision, grammar fit, register, and ambiguity.
"""
from __future__ import annotations

from dataclasses import dataclass

from .meaning_graph import SemanticGraph


@dataclass(frozen=True)
class LexicalCandidate:
    surface: str
    semantic_precision: float
    grammar_fit: float
    casual_fit: float
    formal_fit: float
    ambiguity: float


_MEETING_MODIFIERS = (
    LexicalCandidate("nice", 0.94, 1.00, 0.98, 0.76, 0.03),
    LexicalCandidate("glad", 0.93, 0.98, 0.94, 0.88, 0.02),
    LexicalCandidate("pleased", 0.96, 0.96, 0.62, 1.00, 0.02),
    LexicalCandidate("delighted", 0.91, 0.90, 0.56, 0.94, 0.04),
    LexicalCandidate("happy", 0.88, 0.96, 0.92, 0.82, 0.04),
)


def _score(candidate: LexicalCandidate, register: str, preferred: str) -> float:
    register_fit = candidate.casual_fit if register == "casual" else candidate.formal_fit
    reuse = 0.30 if preferred and candidate.surface == preferred else 0.0
    return (
        0.42 * candidate.semantic_precision
        + 0.22 * candidate.grammar_fit
        + 0.20 * register_fit
        - 0.16 * candidate.ambiguity
        + reuse
    )


class LexicalCompiler:
    def choose_meeting_modifier(self, graph: SemanticGraph) -> str:
        source = graph.by_kind("affect")
        preferred = source[0].surface.lower() if source else ""
        register = "casual" if any(word in {"u", "ya", "yo", "thx"} for word in graph.tokens) else "formal"
        best = max(_MEETING_MODIFIERS, key=lambda item: _score(item, register, preferred))
        return best.surface.capitalize()

    @staticmethod
    def choose_greeting(graph: SemanticGraph) -> str:
        # A greeting is lexicalized from the recognized speech act, while the
        # wording remains a local choice rather than a corpus retrieval.
        for candidate in ("hello", "hi", "hey"):
            if candidate in graph.tokens:
                return candidate.capitalize()
        return "Hello"

    @staticmethod
    def choose_farewell(graph: SemanticGraph) -> str:
        for candidate in ("goodbye", "bye", "farewell"):
            if candidate in graph.tokens:
                return candidate.capitalize()
        return "Goodbye"
