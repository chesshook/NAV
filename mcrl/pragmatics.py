"""Mathematical pragmatic interpretation for MCR-L.

This module does not predict a next token. It evaluates explicit semantic
hypotheses against role, construction, and polarity evidence. Scores express
constraint support, not calibrated probabilities. Counterfactual ablation
measures which evidence actually changes each hypothesis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


SECOND_PERSON = {"you", "your", "yours"}
COPULAS = {"am", "is", "are", "was", "were"}
POSITIVE_WORDS = {
    "nice", "glad", "pleased", "happy", "great", "good", "wonderful",
    "delighted", "awesome", "amazing", "cool", "perfect", "fantastic",
    "brilliant", "smart", "helpful", "kind", "hero", "superhero", "genius",
    "legend", "super",
}
NEGATION_WORDS = {"not", "never", "no", "cannot", "without"}
CAPABILITY_ALIASES = {
    "sum": "arithmetic", "add": "arithmetic", "addition": "arithmetic",
    "calculate": "arithmetic", "compute": "arithmetic", "subtract": "arithmetic",
    "multiply": "arithmetic", "divide": "arithmetic", "math": "arithmetic",
    "code": "code", "program": "code", "programming": "code",
    "translate": "translation", "summarize": "summarization",
}
CAPABILITY_FILLER = {
    "please", "me", "you", "your", "u", "the", "a", "an", "this",
    "that", "it", "something", "anything", "for", "with", "to",
}
NON_CAPABILITY_REQUESTS = {
    "tell", "explain", "show", "give", "say", "help", "make", "build",
    "create", "fix", "compare", "describe", "find", "search",
}


@dataclass(frozen=True)
class InterpretationCandidate:
    """One explainable hypothesis in the pragmatic interpretation field."""

    label: str
    target: str
    score: float
    evidence: Mapping[str, float]
    sensitivity: Mapping[str, float]
    constraints: tuple[str, ...] = ()


def _contains(words: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    width = len(phrase)
    return any(words[i:i + width] == phrase for i in range(len(words) - width + 1))


def _weighted_candidate(
    label: str,
    target: str,
    evidence: Mapping[str, float],
    weights: Mapping[str, float],
    constraints: tuple[str, ...],
    contradiction: float = 0.0,
) -> InterpretationCandidate:
    """Score a hypothesis by weighted support, then ablate each active cue.

    The denominator is fixed for a hypothesis, so ablation has a clear meaning:
    how much score would disappear if one observed cue were absent?
    """
    denominator = sum(weights.values()) or 1.0
    numerator = sum(weights[name] * max(0.0, min(1.0, evidence.get(name, 0.0)))
                    for name in weights)
    support = numerator / denominator
    score = max(0.0, min(1.0, support - 0.45 * contradiction))

    raw_sensitivity = {
        name: weights[name] * max(0.0, min(1.0, evidence.get(name, 0.0))) / denominator
        for name in weights
    }
    total_sensitivity = sum(raw_sensitivity.values()) or 1.0
    sensitivity = {
        name: value / total_sensitivity for name, value in raw_sensitivity.items()
        if value > 0.0
    }
    return InterpretationCandidate(
        label=label,
        target=target,
        score=round(score, 6),
        evidence=dict(evidence),
        sensitivity={name: round(value, 6) for name, value in sensitivity.items()},
        constraints=constraints,
    )


def _identity_patterns(words: tuple[str, ...]) -> tuple[dict[str, float], dict[str, float]]:
    user = {
        "direct_question": float(_contains(words, ("who", "am", "i"))),
        "embedded_question": float(_contains(words, ("know", "who", "i", "am"))),
        "first_person_reference": float(any(w in {"i", "me", "my", "mine"} for w in words)),
        "interrogative_frame": float("who" in words or "what" in words),
        "assistant_knowledge_frame": float("know" in words and "you" in words),
    }
    assistant = {
        "direct_question": float(
            _contains(words, ("who", "are", "you"))
            or _contains(words, ("what", "are", "you"))
        ),
        "name_request": float(
            "your" in words and "name" in words
            and any(w in {"tell", "say", "what", "give", "is"} for w in words)
        ),
        "self_reference": float(
            "yourself" in words
            and any(w in {"about", "describe", "introduce", "know"} for w in words)
        ),
        "addressee_reference": float(any(w in SECOND_PERSON for w in words)),
    }
    return user, assistant


def _capability_target(words: tuple[str, ...]) -> str:
    """Resolve the skill requested by an ability question, not arbitrary content."""
    marker = None
    for phrase in (("know", "how", "to"), ("able", "to")):
        width = len(phrase)
        marker = next(
            (i + width for i in range(len(words) - width + 1)
             if words[i:i + width] == phrase),
            None,
        )
        if marker is not None:
            break

    candidate_words: tuple[str, ...] = ()
    direct_modal = False
    if marker is not None:
        candidate_words = words[marker:]
    elif len(words) >= 3 and words[0] in {"can", "could"} and words[1] == "you":
        candidate_words = words[2:]
        direct_modal = True
    else:
        return ""

    for word in candidate_words:
        if word in CAPABILITY_FILLER:
            continue
        if direct_modal and word in NON_CAPABILITY_REQUESTS:
            return ""
        if word in CAPABILITY_ALIASES:
            return CAPABILITY_ALIASES[word]
        if direct_modal:
            return ""
        if word.isalpha() and len(word) > 2:
            return word
    return ""


@dataclass(frozen=True)
class PragmaticSignals:
    identity_target: str = ""
    capability_target: str = ""
    compliment: bool = False
    candidates: tuple[InterpretationCandidate, ...] = ()
    identity_margin: float = 0.0
    ambiguous_identity: bool = False

    @property
    def assistant_identity(self) -> bool:
        return self.identity_target == "assistant"

    @property
    def user_identity(self) -> bool:
        return self.identity_target == "user"

    def prevents_topic_extraction(self, *, greeting: bool, wellbeing: bool,
                                  gratitude: bool, farewell: bool) -> bool:
        return bool(
            self.identity_target or self.capability_target or self.compliment
            or greeting or wellbeing or gratitude or farewell
        )

    def diagnostics(self) -> dict:
        return {
            "method": "weighted_semantic_constraint_support",
            "score_semantics": "support score; not a calibrated probability",
            "identity_target": self.identity_target,
            "identity_margin": self.identity_margin,
            "ambiguous_identity": self.ambiguous_identity,
            "capability_target": self.capability_target,
            "compliment": self.compliment,
            "candidates": [
                {
                    "label": item.label,
                    "target": item.target,
                    "score": item.score,
                    "evidence": dict(item.evidence),
                    "counterfactual_sensitivity": dict(item.sensitivity),
                    "constraints": list(item.constraints),
                }
                for item in self.candidates
            ],
        }


def analyze_pragmatics(words: tuple[str, ...]) -> PragmaticSignals:
    """Generate and score competing pragmatic hypotheses without token guessing."""
    user_pattern, assistant_pattern = _identity_patterns(words)
    candidates: list[InterpretationCandidate] = []

    user_gate = bool(user_pattern["direct_question"] or user_pattern["embedded_question"])
    assistant_gate = bool(
        assistant_pattern["direct_question"]
        or assistant_pattern["name_request"]
        or assistant_pattern["self_reference"]
    )

    if user_gate:
        candidates.append(_weighted_candidate(
            "identity_question", "user",
            user_pattern,
            {
                "direct_question": 0.38,
                "embedded_question": 0.32,
                "first_person_reference": 0.18,
                "interrogative_frame": 0.07,
                "assistant_knowledge_frame": 0.05,
            },
            ("identity_target_must_be_user", "must_not_extract_am_as_topic"),
        ))
    if assistant_gate:
        candidates.append(_weighted_candidate(
            "identity_question", "assistant",
            assistant_pattern,
            {
                "direct_question": 0.38,
                "name_request": 0.32,
                "self_reference": 0.22,
                "addressee_reference": 0.08,
            },
            ("identity_target_must_be_assistant", "must_not_create_world_topic"),
        ))

    capability_target = _capability_target(words)
    if capability_target:
        capability_evidence = {
            "ability_construction": float(
                _contains(words, ("know", "how", "to"))
                or _contains(words, ("able", "to"))
                or (len(words) >= 3 and words[0] in {"can", "could"} and words[1] == "you")
            ),
            "assistant_addressee": float(any(w in SECOND_PERSON for w in words)),
            "skill_resolved": 1.0,
            "question_frame": float("?" in words or words[0] in {"do", "can", "could", "are"}),
        }
        candidates.append(_weighted_candidate(
            "capability_question", capability_target, capability_evidence,
            {
                "ability_construction": 0.42,
                "assistant_addressee": 0.18,
                "skill_resolved": 0.30,
                "question_frame": 0.10,
            },
            ("must_be_about_assistant_ability", "must_not_be_routed_as_fact_lookup"),
        ))

    positive_attribution = float(any(w in POSITIVE_WORDS for w in words))
    second_person = float(any(w in SECOND_PERSON for w in words))
    copular_attribution = float(any(w in COPULAS for w in words))
    negated = float(any(w in NEGATION_WORDS for w in words))
    if second_person and copular_attribution and positive_attribution and not negated:
        compliment_evidence = {
            "addressee_reference": second_person,
            "copular_attribution": copular_attribution,
            "positive_attribution": positive_attribution,
        }
        candidates.append(_weighted_candidate(
            "social_compliment", "assistant", compliment_evidence,
            {
                "addressee_reference": 0.30,
                "copular_attribution": 0.28,
                "positive_attribution": 0.42,
            },
            ("positive_quality_must_target_assistant", "negation_must_not_be_present"),
            contradiction=negated,
        ))

    identity_candidates = [
        candidate for candidate in candidates
        if candidate.label == "identity_question"
    ]
    identity_target = ""
    identity_margin = 0.0
    ambiguous_identity = False
    if identity_candidates:
        ranked = sorted(identity_candidates, key=lambda item: item.score, reverse=True)
        identity_margin = ranked[0].score - (ranked[1].score if len(ranked) > 1 else 0.0)
        if len(ranked) == 1 or identity_margin >= 0.12:
            identity_target = ranked[0].target
        else:
            ambiguous_identity = True

    capability_candidates = [
        candidate for candidate in candidates if candidate.label == "capability_question"
    ]
    compliment_candidates = [
        candidate for candidate in candidates if candidate.label == "social_compliment"
    ]
    selected_capability = capability_candidates[0].target if capability_candidates else ""
    compliment = bool(compliment_candidates and compliment_candidates[0].score >= 0.70)

    candidates.sort(key=lambda item: item.score, reverse=True)
    return PragmaticSignals(
        identity_target=identity_target,
        capability_target=selected_capability,
        compliment=compliment,
        candidates=tuple(candidates),
        identity_margin=round(identity_margin, 6),
        ambiguous_identity=ambiguous_identity,
    )
