"""Back-parse generated wording and check preservation of response meaning."""
from __future__ import annotations

from dataclasses import dataclass

from .meaning_graph import SemanticGraph
from .meaning_solver import ResponseMeaning
from .semantic_parser import parse_meaning


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    preservation: float
    checks: dict[str, bool]
    reparsed_speech_act: str

    def diagnostics(self) -> dict:
        return {
            "valid": self.valid,
            "preservation": round(self.preservation, 4),
            "checks": self.checks,
            "reparsed_speech_act": self.reparsed_speech_act,
        }


def validate_response(
    response: ResponseMeaning,
    source: SemanticGraph,
    text: str,
) -> ValidationResult:
    reparsed = parse_meaning(text)
    checks: dict[str, bool]

    if response.act == "reciprocate_meeting":
        checks = {
            "positive_tone": bool(reparsed.by_kind("affect")),
            "meeting_event": reparsed.predicate == "meeting",
            "recipient": bool(reparsed.edge_targets("addressee")),
            "reciprocity": reparsed.has("operator.reciprocity"),
            "social_pleasantry": reparsed.speech_act == "social.pleasantry",
        }
    elif response.act == "reciprocate_greeting":
        checks = {"greeting": reparsed.speech_act == "social.greeting"}
    elif response.act == "acknowledge_gratitude":
        checks = {"acknowledgement": "welcome" in reparsed.tokens or "you're" in text.lower()}
    elif response.act == "reciprocate_farewell":
        checks = {"farewell": reparsed.speech_act == "social.farewell"}
    elif response.act == "answer_wellbeing":
        lower = text.lower()
        checks = {"status_answer": "running" in lower, "honest_nonhuman_status": "feelings" in lower}
    elif response.act == "ask_clarification":
        lower = text.lower()
        checks = {
            "clarification_question": text.rstrip().endswith("?"),
            "asks_for_missing_intent": "clarify" in lower or "explain" in lower or "know" in lower,
        }
    elif response.act == "identify_self":
        lower = text.lower()
        checks = {"system_identity": "mcr-l" in lower, "scope_limit": "not a general-purpose" in lower}
    else:
        checks = {"nonempty": bool(text.strip())}

    passed = sum(1 for value in checks.values() if value)
    preservation = passed / max(1, len(checks))
    return ValidationResult(
        valid=all(checks.values()),
        preservation=preservation,
        checks=checks,
        reparsed_speech_act=reparsed.speech_act,
    )
