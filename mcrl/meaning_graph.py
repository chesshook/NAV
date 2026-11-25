"""Typed meaning graphs used as the language engine's internal representation.

A graph separates concepts (nodes) from semantic roles (edges). It is deliberately
small and inspectable; it is not a claim of unrestricted natural-language parsing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MeaningNode:
    id: int
    concept: str
    kind: str
    value: str
    confidence: float = 1.0
    surface: str = ""


@dataclass(frozen=True)
class MeaningEdge:
    source: int
    target: int
    role: str
    confidence: float = 1.0


@dataclass(frozen=True)
class SemanticGraph:
    original: str
    normalized: str
    tokens: tuple[str, ...]
    nodes: tuple[MeaningNode, ...]
    edges: tuple[MeaningEdge, ...]
    lexical_tokens: tuple[str, ...]
    intent: str = "statement"

    def by_kind(self, kind: str) -> tuple[MeaningNode, ...]:
        return tuple(node for node in self.nodes if node.kind == kind)

    def by_concept(self, concept: str) -> tuple[MeaningNode, ...]:
        return tuple(node for node in self.nodes if node.concept == concept)

    def first(self, kind: str, default: str = "") -> str:
        found = self.by_kind(kind)
        return found[0].value if found else default

    def has(self, concept: str) -> bool:
        return any(node.concept == concept for node in self.nodes)

    def edge_targets(self, role: str) -> tuple[MeaningNode, ...]:
        ids = {edge.target for edge in self.edges if edge.role == role}
        return tuple(node for node in self.nodes if node.id in ids)

    @property
    def speech_act(self) -> str:
        return self.first("speech_act", "statement")

    @property
    def predicate(self) -> str:
        return self.first("event", "")

    @property
    def polarity(self) -> str:
        return self.first("polarity", "positive")

    @property
    def temporal_relation(self) -> str:
        return self.first("temporal", "unspecified")

    @property
    def comparison(self) -> str:
        return self.first("comparison", "none")

    def without_node(self, node_id: int) -> "SemanticGraph":
        """Return a counterfactual graph with one meaning node ablated."""
        return SemanticGraph(
            original=self.original,
            normalized=self.normalized,
            tokens=self.tokens,
            nodes=tuple(node for node in self.nodes if node.id != node_id),
            edges=tuple(
                edge for edge in self.edges
                if edge.source != node_id and edge.target != node_id
            ),
            lexical_tokens=self.lexical_tokens,
            intent=self.intent,
        )

    def summary(self) -> dict:
        return {
            "speech_act": self.speech_act,
            "intent": self.intent,
            "nodes": [
                {
                    "id": node.id,
                    "concept": node.concept,
                    "kind": node.kind,
                    "value": node.value,
                    "surface": node.surface,
                    "confidence": round(node.confidence, 3),
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "role": edge.role,
                    "confidence": round(edge.confidence, 3),
                }
                for edge in self.edges
            ],
        }


class MeaningGraphBuilder:
    """Convenience builder which assigns stable, local node identifiers."""

    def __init__(self) -> None:
        self.nodes: list[MeaningNode] = []
        self.edges: list[MeaningEdge] = []

    def node(
        self,
        concept: str,
        kind: str,
        value: str,
        confidence: float = 1.0,
        surface: str = "",
    ) -> int:
        node_id = len(self.nodes)
        self.nodes.append(
            MeaningNode(
                id=node_id,
                concept=concept,
                kind=kind,
                value=value,
                confidence=max(0.0, min(1.0, float(confidence))),
                surface=surface,
            )
        )
        return node_id

    def edge(self, source: int, target: int, role: str, confidence: float = 1.0) -> None:
        if source < 0 or target < 0 or source >= len(self.nodes) or target >= len(self.nodes):
            raise IndexError("meaning edge references an unknown node")
        self.edges.append(
            MeaningEdge(source, target, role, max(0.0, min(1.0, float(confidence))))
        )

    def build(
        self,
        original: str,
        normalized: str,
        tokens: Iterable[str],
        lexical_tokens: Iterable[str],
        intent: str = "statement",
    ) -> SemanticGraph:
        return SemanticGraph(
            original=original,
            normalized=normalized,
            tokens=tuple(tokens),
            nodes=tuple(self.nodes),
            edges=tuple(self.edges),
            lexical_tokens=tuple(lexical_tokens),
            intent=intent,
        )
