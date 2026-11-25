"""Controlled language realization from a selected, evidence-backed plan."""
from __future__ import annotations

from .planner import CandidatePlan


def realize(plan: CandidatePlan) -> str:
    record = plan.record
    answer = (record.get("answer") or record.get("text") or "").strip()
    if not answer:
        return ""
    intent = record.get("intents", [])
    # A template is selected from the user's intent, but the factual payload
    # always comes from the selected indexed record.
    if "define" in intent and not answer.endswith((".", "!", "?")):
        answer += "."
    if plan.score < 0.58:
        return "The closest matching indexed reference says: " + answer
    return answer
