"""Small, explicit semantic parser. This is a rule-based baseline, not an LLM."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.IGNORECASE)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "how", "i", "in", "is", "it", "me",
    "my", "of", "on", "or", "our", "please", "should", "so", "that", "the",
    "their", "them", "there", "these", "this", "to", "us", "was", "we", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you",
    "your",
}

INTENT_CUES = {
    "define": ("what is", "what are", "define", "meaning of", "explain the term"),
    "explain": ("why", "how does", "how do", "reason", "because", "cause", "explain"),
    "how_to": ("how to", "steps to", "how can i", "how do i", "instructions"),
    "compare": ("difference between", "compare", "versus", " vs ", "better than"),
    "advice": ("should i", "can i", "is it safe", "is it okay", "recommend"),
}

RISK_CUES = (
    "safe", "danger", "delete", "remove", "format", "password", "medical",
    "medicine", "legal", "electric", "fire", "weapon", "security",
)
CAUSAL_CUES = ("why", "because", "cause", "reason", "result", "therefore", "leads to")
TEMPORAL_CUES = ("before", "after", "when", "while", "during", "until", "since", "yesterday", "today", "tomorrow")
NEGATION_CUES = ("not", "never", "no ", "cannot", "can't", "isn't", "doesn't", "without", "neither")
FOLLOW_UP_CUES = ("it", "that", "those", "these", "also", "then", "what about", "and what", "continue")

FEATURE_NAMES = (
    "question", "define", "explain", "compare", "how_to", "entity", "action",
    "causality", "temporal", "negation", "constraint", "uncertainty",
    "evidence", "risk", "context", "follow_up", "social",
)

SOCIAL_PATTERNS = (
    ("greeting", re.compile(r"^(hi|hello|hey|howdy|yo|good morning|good afternoon|good evening)[!.?]*$")),
    ("wellbeing", re.compile(r"^(how are you|how's it going|how are things)[!.?]*$")),
    ("thanks", re.compile(r"^(thanks|thank you|thx|many thanks|i appreciate it)[!.?]*$")),
    ("farewell", re.compile(r"^(bye|goodbye|see you|good night|see you later)[!.?]*$")),
    ("identity", re.compile(r"^(who are you|what are you|what's your name|what is your name|are you an ai|are you a bot)[!.?]*$")),
)

SOCIAL_RESPONSES = {
    "greeting": "Hi! I'm MCR-L 0.1. I can answer from my local knowledge index, explain what I know, and tell you when I don't have enough evidence. What would you like to explore?",
    "wellbeing": "I'm running and ready to help. I'm software, so I don't have feelings, but thanks for asking. What shall we work on?",
    "thanks": "You're welcome! What would you like to explore next?",
    "farewell": "Goodbye! Your local MCR-L session can be started again whenever you're ready.",
    "identity": "I'm MCR-L 0.1, an experimental CPU-first, rule-and-retrieval language reasoning prototype. I am not yet a general-purpose neural language model.",
}


def classify_social(text: str) -> str | None:
    """Recognize complete, common social turns before factual retrieval."""
    normalized = " ".join((text or "").lower().split())
    for intent, pattern in SOCIAL_PATTERNS:
        if pattern.fullmatch(normalized):
            return intent
    return None


def social_response(intent: str) -> str:
    """Return a deterministic response for a recognized social intent."""
    return SOCIAL_RESPONSES[intent]


@dataclass(frozen=True)
class SemanticFrame:
    original: str
    normalized: str
    tokens: tuple[str, ...]
    content_tokens: tuple[str, ...]
    intent: str
    features: tuple[float, ...]
    entities: tuple[str, ...] = ()


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(t.lower() for t in TOKEN_RE.findall(text or ""))


def _contains_any(text: str, cues: Iterable[str]) -> bool:
    padded = f" {text.lower()} "
    return any(cue in padded for cue in cues)


def _intent(text: str) -> str:
    social = classify_social(text)
    if social:
        return social
    padded = f" {text.lower()} "
    for intent, cues in INTENT_CUES.items():
        if any(cue in padded for cue in cues):
            return intent
    if "?" in text:
        return "question"
    return "general"


def parse_semantics(text: str) -> SemanticFrame:
    original = (text or "").strip()
    normalized = " ".join((text or "").lower().split())
    words = tokenize(normalized)
    content = tuple(w for w in words if w not in STOPWORDS and len(w) > 1)
    intent = _intent(normalized)

    values = {name: 0.0 for name in FEATURE_NAMES}
    values["question"] = float("?" in original or intent != "general")
    values["define"] = float(intent == "define")
    values["explain"] = float(intent == "explain")
    values["compare"] = float(intent == "compare")
    values["how_to"] = float(intent == "how_to")
    values["advice"] = float(intent == "advice") if "advice" in values else 0.0
    values["causality"] = float(_contains_any(normalized, CAUSAL_CUES))
    values["temporal"] = float(_contains_any(normalized, TEMPORAL_CUES))
    values["negation"] = float(_contains_any(normalized, NEGATION_CUES))
    values["risk"] = float(_contains_any(normalized, RISK_CUES))
    values["uncertainty"] = float(intent in {"advice", "explain"} or not content)
    values["constraint"] = float(any(c in normalized for c in ("only", "must", "without", "under", "at least", "exactly")))
    values["context"] = min(1.0, len(content) / 12.0)
    values["follow_up"] = float(normalized.startswith(FOLLOW_UP_CUES) or any(normalized.startswith(c) for c in FOLLOW_UP_CUES))
    values["entity"] = min(1.0, len(content) / 5.0)
    values["action"] = float(any(w in content for w in ("build", "make", "create", "install", "run", "delete", "fix", "use", "open")))
    # Evidence is set after retrieval; it must not be inferred from the question itself.
    values["evidence"] = 0.0
    values["social"] = float(intent in {"greeting", "wellbeing", "thanks", "farewell", "identity"})

    features = tuple(max(0.0, min(1.0, values[name])) for name in FEATURE_NAMES)
    return SemanticFrame(
        original=original,
        normalized=normalized,
        tokens=words,
        content_tokens=content,
        intent=intent,
        features=features,
    )


def with_evidence(frame: SemanticFrame, evidence_strength: float) -> SemanticFrame:
    values = list(frame.features)
    index = FEATURE_NAMES.index("evidence")
    values[index] = max(0.0, min(1.0, float(evidence_strength)))
    return SemanticFrame(
        original=frame.original,
        normalized=frame.normalized,
        tokens=frame.tokens,
        content_tokens=frame.content_tokens,
        intent=frame.intent,
        features=tuple(values),
        entities=frame.entities,
    )
