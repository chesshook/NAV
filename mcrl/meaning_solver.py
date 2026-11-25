"""Solve for a response meaning before any sentence is written."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .meaning_graph import SemanticGraph
from .sensitivity import SensitivityResult


@dataclass(frozen=True)
class ResponseMeaning:
    act: str
    predicate: str = ""
    affect: str = ""
    recipient: str = "user"
    reciprocal: bool = False
    style: str = "neutral"
    requirements: tuple[str, ...] = ()
    source_modifier: str = ""
    utility: float = 0.0
    diagnostics: Mapping[str, float] | None = None


def _source_modifier(graph: SemanticGraph) -> str:
    modifiers = graph.by_kind("affect")
    return modifiers[0].surface.lower() if modifiers else ""


def _candidate_components(
    graph: SemanticGraph,
    sensitivity: SensitivityResult,
    candidate: ResponseMeaning,
) -> dict[str, float]:
    """Expose the distinct terms used to score a response-level meaning."""
    weights = sensitivity.kind_weights
    preservation = {
        "event": 1.0 if candidate.predicate else 0.0,
        "affect": 1.0 if candidate.affect else 0.0,
        "participant": 1.0 if candidate.recipient else 0.0,
        "speech_act": 1.0 if candidate.act else 0.0,
    }
    semantic_preservation = sum(
        weights.get(kind, 0.0) * value
        for kind, value in preservation.items()
    )
    task_fit = 0.20 if candidate.act in {
        "reciprocate_meeting", "reciprocate_greeting", "acknowledge_gratitude",
        "reciprocate_farewell", "answer_wellbeing", "identify_self", "confirm_presence",
        "acknowledge_positive_statement", "acknowledge_negative_statement",
        "acknowledge_topic_avoidance", "greet_and_answer_wellbeing",
        "acknowledge_compliment", "answer_user_identity", "answer_capability",
    } else 0.0
    missing_requirement_penalty = sum(
        0.035 for requirement in candidate.requirements
        if requirement == "recipient" and not candidate.recipient
    )
    declared_plan_utility = candidate.utility
    total = (
        semantic_preservation + task_fit + declared_plan_utility
        - missing_requirement_penalty
    )
    return {
        "semantic_preservation": semantic_preservation,
        "pragmatic_task_fit": task_fit,
        "declared_plan_utility": declared_plan_utility,
        "missing_requirement_penalty": missing_requirement_penalty,
        "total": total,
    }


def _candidate_score(
    graph: SemanticGraph,
    sensitivity: SensitivityResult,
    candidate: ResponseMeaning,
) -> float:
    """Compatibility wrapper for the scalar response utility."""
    return _candidate_components(graph, sensitivity, candidate)["total"]


def solve_response(
    graph: SemanticGraph,
    sensitivity: SensitivityResult,
) -> ResponseMeaning | None:
    """Choose a response-level meaning graph by weighted preservation utility."""
    acts = set(graph.speech_acts)
    if {"social.greeting", "social.wellbeing"} <= acts:
        candidates = (
            ResponseMeaning(
                act="greet_and_answer_wellbeing",
                recipient="you",
                requirements=("return_greeting", "answer_wellbeing", "invite_reciprocity"),
                utility=0.28,
            ),
        )
    elif graph.speech_act == "social.compliment":
        candidates = (
            ResponseMeaning(
                act="acknowledge_compliment",
                affect="positive",
                recipient="you",
                requirements=("acknowledge_kindness", "avoid_self_aggrandizement", "continue_naturally"),
                utility=0.25,
            ),
        )
    elif graph.speech_act == "request.user_identity":
        candidates = (
            ResponseMeaning(
                act="answer_user_identity",
                recipient="you",
                requirements=("state_context_limit", "do_not_claim_external_identity"),
                utility=0.26,
            ),
        )
    elif graph.speech_act == "request.capability":
        capability_nodes = graph.by_kind("capability")
        capability = capability_nodes[0].value if capability_nodes else ""
        candidates = (
            ResponseMeaning(
                act="answer_capability",
                predicate=capability,
                recipient="you",
                requirements=("answer_ability_directly", "state_real_capability_boundary"),
                utility=0.26,
            ),
        )
    elif graph.speech_act == "social.pleasantry":
        candidates = (
            ResponseMeaning(
                act="reciprocate_meeting",
                predicate="meet",
                affect="positive",
                recipient="you",
                reciprocal=True,
                requirements=("acknowledge", "meeting", "positive_tone", "recipient", "reciprocity"),
                source_modifier=_source_modifier(graph),
                utility=0.22,
            ),
            ResponseMeaning(
                act="generic_acknowledgement",
                affect="positive",
                recipient="you",
                requirements=("acknowledge", "positive_tone", "recipient"),
                source_modifier=_source_modifier(graph),
                utility=-0.02,
            ),
        )
    elif graph.speech_act == "social.greeting":
        candidates = (
            ResponseMeaning(
                act="reciprocate_greeting",
                recipient="you",
                requirements=("acknowledge", "greeting", "recipient"),
                utility=0.20,
            ),
        )
    elif graph.speech_act == "social.gratitude":
        candidates = (
            ResponseMeaning(
                act="acknowledge_gratitude",
                recipient="you",
                requirements=("acknowledge", "gratitude"),
                utility=0.20,
            ),
        )
    elif graph.speech_act == "social.farewell":
        candidates = (
            ResponseMeaning(
                act="reciprocate_farewell",
                recipient="you",
                requirements=("acknowledge", "farewell"),
                utility=0.20,
            ),
        )
    elif graph.speech_act == "social.wellbeing":
        candidates = (
            ResponseMeaning(
                act="answer_wellbeing",
                recipient="you",
                requirements=("answer_question", "status"),
                utility=0.20,
            ),
        )
    elif graph.speech_act == "request.clarification":
        candidates = (
            ResponseMeaning(
                act="ask_clarification",
                recipient="you",
                requirements=("acknowledge_incomplete_question", "invite_specific_topic"),
                utility=0.20,
            ),
        )
    elif graph.speech_act == "social.positive_acknowledgement":
        candidates = (
            ResponseMeaning(
                act="acknowledge_positive_statement",
                affect="positive",
                recipient="you",
                requirements=("acknowledge_positive_feedback", "maintain_conversation"),
                utility=0.22,
            ),
        )
    elif graph.speech_act == "social.empathetic_acknowledgement":
        candidates = (
            ResponseMeaning(
                act="acknowledge_negative_statement",
                affect="negative",
                recipient="you",
                requirements=("acknowledge_negative_affect", "invite_context_without_assuming"),
                utility=0.22,
            ),
        )
    elif graph.speech_act == "request.avoid_topic":
        topic_nodes = graph.by_kind("topic")
        topic = topic_nodes[0].value if topic_nodes else "that topic"
        candidates = (
            ResponseMeaning(
                act="acknowledge_topic_avoidance",
                recipient=topic,
                requirements=("acknowledge_preference", "do_not_repeat_topic"),
                utility=0.24,
            ),
        )
    elif graph.speech_act == "request.presence_check":
        candidates = (
            ResponseMeaning(
                act="confirm_presence",
                predicate="be_present",
                recipient="you",
                requirements=("answer_presence_question", "state_current_availability"),
                utility=0.24,
            ),
        )
    elif graph.speech_act == "request.identity":
        candidates = (
            ResponseMeaning(
                act="identify_self",
                recipient="you",
                requirements=("identify_system", "state_capability", "state_limit"),
                utility=0.20,
            ),
        )
    else:
        return None

    ranked = [
        (_candidate_components(graph, sensitivity, candidate), candidate)
        for candidate in candidates
    ]
    ranked.sort(key=lambda pair: pair[0]["total"], reverse=True)
    score_components, selected = ranked[0]
    score = score_components["total"]
    return ResponseMeaning(
        act=selected.act,
        predicate=selected.predicate,
        affect=selected.affect,
        recipient=selected.recipient,
        reciprocal=selected.reciprocal,
        style=selected.style,
        requirements=selected.requirements,
        source_modifier=selected.source_modifier,
        utility=score,
        diagnostics={
            "selected_plan_utility": score,
            "candidates_considered": float(len(candidates)),
            "semantic_preservation": score_components["semantic_preservation"],
            "pragmatic_task_fit": score_components["pragmatic_task_fit"],
            "declared_plan_utility": score_components["declared_plan_utility"],
            "missing_requirement_penalty": score_components["missing_requirement_penalty"],
            "input_speech_act_weight": sensitivity.kind_weights.get("speech_act", 0.0),
            "input_event_weight": sensitivity.kind_weights.get("event", 0.0),
            "input_participant_weight": sensitivity.kind_weights.get("participant", 0.0),
        },
    )
