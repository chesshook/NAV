"""Pragmatic interpretation of conversational moves and participant intent.

This layer is deliberately separate from topic extraction and factual retrieval.
It identifies interactional intent (who is being asked about, capability checks,
compliments, and co-occurring social moves) before the engine chooses a response.
"""
from __future__ import annotations

from dataclasses import dataclass


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


def _contains(words: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    width = len(phrase)
    return any(words[i:i + width] == phrase for i in range(len(words) - width + 1))


def _identity_target(words: tuple[str, ...]) -> str:
    # Questions about the assistant and questions about the user are distinct
    # pragmatic acts. Do not let "am" leak out as a factual topic.
    user_patterns = (
        ("who", "am", "i"),
        ("know", "who", "i", "am"),
        ("what", "do", "you", "know", "about", "me"),
        ("do", "you", "remember", "me"),
    )
    if any(_contains(words, pattern) for pattern in user_patterns):
        return "user"

    assistant_patterns = (
        ("who", "are", "you"),
        ("what", "are", "you"),
        ("what", "is", "your", "name"),
        ("tell", "me", "your", "name"),
        ("say", "your", "name"),
    )
    if any(_contains(words, pattern) for pattern in assistant_patterns):
        return "assistant"
    if "yourself" in words and (
        "about" in words or "describe" in words or "introduce" in words
    ):
        return "assistant"
    return ""


def _capability_target(words: tuple[str, ...]) -> str:
    # "Do you know how to X?" and "Are you able to X?" are ability checks.
    # A short "Can you sum?" is also an ability check, while "Can you tell
    # me about X?" remains a content request rather than a capability check.
    marker = None
    for phrase in (("know", "how", "to"), ("able", "to")):
        width = len(phrase)
        marker = next(
            (i + width for i in range(len(words) - width + 1) if words[i:i + width] == phrase),
            None,
        )
        if marker is not None:
            break

    candidate_words: tuple[str, ...] = ()
    if marker is not None:
        candidate_words = words[marker:]
    elif len(words) >= 3 and words[0] in {"can", "could"} and words[1] == "you":
        candidate_words = words[2:]
    else:
        return ""

    for word in candidate_words:
        if word in CAPABILITY_ALIASES:
            return CAPABILITY_ALIASES[word]
        if word in CAPABILITY_FILLER:
            continue
        # Keep a bounded, normalized name for unsupported skills too. The
        # response layer can then say clearly that the capability is absent.
        if word.isalpha() and len(word) > 2:
            return word
    return ""


@dataclass(frozen=True)
class PragmaticSignals:
    identity_target: str = ""
    capability_target: str = ""
    compliment: bool = False

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


def analyze_pragmatics(words: tuple[str, ...]) -> PragmaticSignals:
    identity_target = _identity_target(words)
    capability_target = _capability_target(words)
    compliment = (
        any(word in SECOND_PERSON for word in words)
        and any(word in COPULAS for word in words)
        and any(word in POSITIVE_WORDS for word in words)
        and not any(word in NEGATION_WORDS for word in words)
    )
    return PragmaticSignals(
        identity_target=identity_target,
        capability_target=capability_target,
        compliment=compliment,
    )
