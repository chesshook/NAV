"""Evidence-aware scoring of structured answer plans."""
from __future__ import annotations

from dataclasses import dataclass
from .semantic import SemanticFrame, tokenize, FEATURE_NAMES

INTENT_COMPATIBILITY = {
    "define": {"define", "question", "general"},
    "explain": {"explain", "question", "general"},
    "how_to": {"how_to", "question", "general"},
    "compare": {"compare", "question", "general"},
    "advice": {"advice", "question", "general"},
    "question": {"question", "define", "explain", "how_to", "compare", "advice", "general"},
    "general": {"general", "define", "explain", "how_to", "question"},
}


@dataclass(frozen=True)
class CandidatePlan:
    record: dict
    score: float
    coverage: float
    title_match: float
    intent_fit: float
    evidence_quality: float
    situation_fit: float
    matched_terms: tuple[str, ...]

    def diagnostics(self) -> dict:
        return {
            "title": self.record["title"],
            "score": round(self.score, 4),
            "coverage": round(self.coverage, 4),
            "title_match": round(self.title_match, 4),
            "intent_fit": round(self.intent_fit, 4),
            "evidence_quality": round(self.evidence_quality, 4),
            "situation_fit": round(self.situation_fit, 4),
            "matched_terms": list(self.matched_terms),
        }


def _record_terms(record: dict) -> set[str]:
    text = " ".join([
        record.get("title", ""),
        record.get("text", ""),
        record.get("answer", ""),
        " ".join(record.get("keywords", [])),
    ])
    return set(tokenize(text))


def score_plan(frame: SemanticFrame, record: dict, activation: tuple[float, ...]) -> CandidatePlan:
    query_terms = set(frame.content_tokens)
    record_terms = _record_terms(record)
    matched = tuple(sorted(query_terms & record_terms))
    coverage = len(matched) / max(1, len(query_terms))

    title_terms = set(tokenize(record.get("title", "")))
    title_match = len(query_terms & title_terms) / max(1, len(query_terms))

    compatible = INTENT_COMPATIBILITY.get(frame.intent, {"general", "question"})
    record_intents = set(record.get("intents", []))
    intent_fit = 1.0 if compatible & record_intents else (0.55 if not record_intents else 0.25)
    evidence_quality = 1.0 if record.get("source") else 0.25

    tag_indices = [FEATURE_NAMES.index(tag) for tag in record.get("tags", []) if tag in FEATURE_NAMES]
    situation_fit = (
        sum(activation[index] for index in tag_indices) / len(tag_indices)
        if tag_indices else 0.35
    )

    score = (
        0.40 * coverage
        + 0.22 * title_match
        + 0.14 * intent_fit
        + 0.12 * evidence_quality
        + 0.12 * situation_fit
    )
    return CandidatePlan(
        record=record,
        score=max(0.0, min(1.0, score)),
        coverage=coverage,
        title_match=title_match,
        intent_fit=intent_fit,
        evidence_quality=evidence_quality,
        situation_fit=situation_fit,
        matched_terms=matched,
    )


def build_plans(frame: SemanticFrame, records: list[dict], activation: tuple[float, ...]) -> list[CandidatePlan]:
    plans = [score_plan(frame, record, activation) for record in records]
    plans.sort(key=lambda item: item.score, reverse=True)
    return plans
