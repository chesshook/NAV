"""Counterfactual meaning sensitivity.

For each semantic node, remove it, recompute structural utility, and measure
the loss. The normalized loss controls what the response must preserve.
"""
from __future__ import annotations

from dataclasses import dataclass

from .meaning_graph import SemanticGraph


_KIND_VALUE = {
    "speech_act": 0.34,
    "event": 0.27,
    "predicate": 0.22,
    "participant": 0.10,
    "polarity": 0.12,
    "temporal": 0.09,
    "comparison": 0.10,
    "affect": 0.10,
    "operator": 0.08,
    "entity": 0.07,
    "topic": 0.04,
    "speech_content": 0.16,
}


def semantic_utility(graph: SemanticGraph) -> float:
    """Evaluate how much task-relevant structure remains in a graph."""
    if not graph.nodes:
        return 0.0

    total = 0.0
    seen_kinds: set[str] = set()
    for node in graph.nodes:
        if node.kind == "participant":
            # Each participant can carry a different role; both matter.
            total += _KIND_VALUE["participant"] * node.confidence
        elif node.kind not in seen_kinds:
            total += _KIND_VALUE.get(node.kind, 0.03) * node.confidence
            seen_kinds.add(node.kind)

    role_bonus = {
        "has_event": 0.08,
        "speaker": 0.07,
        "addressee": 0.13,
        "negates": 0.08,
        "temporal_scope": 0.06,
        "compares": 0.06,
        "reciprocated_by": 0.08,
    }
    total += sum(role_bonus.get(edge.role, 0.02) * edge.confidence for edge in graph.edges)
    return total


@dataclass(frozen=True)
class SensitivityResult:
    baseline_utility: float
    node_weights: dict[int, float]
    node_losses: dict[int, float]
    kind_weights: dict[str, float]

    def diagnostics(self) -> dict:
        return {
            "baseline_utility": round(self.baseline_utility, 5),
            "node_weights": {str(k): round(v, 5) for k, v in self.node_weights.items()},
            "node_losses": {str(k): round(v, 5) for k, v in self.node_losses.items()},
            "kind_weights": {k: round(v, 5) for k, v in self.kind_weights.items()},
        }


def calculate_sensitivity(graph: SemanticGraph) -> SensitivityResult:
    """Calculate normalized counterfactual impact for every graph node."""
    baseline = semantic_utility(graph)
    losses: dict[int, float] = {}
    for node in graph.nodes:
        without = graph.without_node(node.id)
        loss = max(0.0, baseline - semantic_utility(without))
        losses[node.id] = loss

    total = sum(losses.values())
    if total <= 1e-12 and graph.nodes:
        # Degenerate graphs still get a defined distribution.
        node_weights = {node.id: 1.0 / len(graph.nodes) for node in graph.nodes}
    else:
        node_weights = {
            node_id: loss / total if total else 0.0
            for node_id, loss in losses.items()
        }

    kind_weights: dict[str, float] = {}
    by_id = {node.id: node for node in graph.nodes}
    for node_id, weight in node_weights.items():
        kind = by_id[node_id].kind
        kind_weights[kind] = kind_weights.get(kind, 0.0) + weight

    return SensitivityResult(
        baseline_utility=baseline,
        node_weights=node_weights,
        node_losses=losses,
        kind_weights=kind_weights,
    )
