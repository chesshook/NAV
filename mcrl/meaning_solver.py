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


def _candidate_score(
    graph: SemanticGraph,
    sensitivity: SensitivityResult,
    candidate: ResponseMeaning,
) -> float:
    weights = sensitivity.kind_weights
    preservation = {
        "event": 1.0 if candidate.predicate else 0.0,
        "affect": 1.0 if candidate.affect else 0.0,
        "participant": 1.0 if candidate.recipient else 0.0,
        "speech_act": 1.0 if candidate.act else 0.0,
    }
    preserved = sum(weights.get(kind, 0.0) * value for kind, value in preservation.items())
    task_fit = 0.20 if candidate.act in {
        "reciprocate_meeting", "reciprocate_greeting", "acknowledge_gratitude",
        "reciprocate_farewell", "answer_wellbeing", "identify_self", "confirm_presence",
    } else 0.0
    missing_requirements = sum(
        0.035 for requirement in candidate.requirements
        if requirement == "recipient" and not candidate.recipient
    )
    # This utility is a response-plan objective, not next-token probability.
    return preserved + task_fit + candidate.utility - missing_requirements


def solve_response(
    graph: SemanticGraph,
    sensitivity: SensitivityResult,
) -> ResponseMeaning | None:
    """Choose a response-level meaning graph by weighted preservation utility."""
    if graph.speech_act == "social.pleasantry":
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

    ranked = sorted(
        ((_candidate_score(graph, sensitivity, candidate), candidate) for candidate in candidates),
        key=lambda pair: pair[0],
        reverse=True,
    )
    score, selected = ranked[0]
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
            "input_speech_act_weight": sensitivity.kind_weights.get("speech_act", 0.0),
            "input_event_weight": sensitivity.kind_weights.get("event", 0.0),
            "input_participant_weight": sensitivity.kind_weights.get("participant", 0.0),
        },
    )
