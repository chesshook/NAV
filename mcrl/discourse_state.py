"""Small discourse memory for grammatical reference resolution."""
from __future__ import annotations

from dataclasses import dataclass, field

ASSISTANT_NAMES = {"mcr", "mcrl", "assistant", "chatgpt"}
SECOND_PERSON = {"you", "u", "ya", "your", "yours"}
FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our"}
ANAPHORS = {"it", "this", "that", "they", "them", "those", "these"}


@dataclass
class DiscourseState:
    """Tracks turn roles and a small set of recent discourse referents.

    This is bounded session memory, not a learned world model.
    """

    turn_count: int = 0
    last_topic: str = ""
    last_speech_act: str = ""
    recent_entities: list[str] = field(default_factory=list)
    user_aliases: set[str] = field(default_factory=lambda: {"user"})
    assistant_aliases: set[str] = field(default_factory=lambda: set(ASSISTANT_NAMES))

    def has_assistant_reference(self, words: tuple[str, ...]) -> bool:
        return any(word in self.assistant_aliases or word in SECOND_PERSON for word in words)

    def resolve_pronoun(self, word: str) -> str | None:
        lowered = (word or "").lower()
        if lowered in SECOND_PERSON or lowered in self.assistant_aliases:
            return "assistant"
        if lowered in FIRST_PERSON or lowered in self.user_aliases:
            return "user"
        if lowered in ANAPHORS:
            return self.last_topic or None
        return None

    def observe_parse(self, words: tuple[str, ...], entities: tuple[str, ...]) -> None:
        self.turn_count += 1
        for entity in entities:
            cleaned = str(entity).strip()
            if cleaned and cleaned.lower() not in self.assistant_aliases and cleaned.lower() not in self.user_aliases:
                self.last_topic = cleaned
                self.recent_entities.append(cleaned)
        self.recent_entities = self.recent_entities[-12:]

    def observe_graph(self, graph) -> None:
        self.last_speech_act = graph.speech_act
        topics = [node.value for node in graph.nodes if node.kind in {"topic", "entity"}]
        self.observe_parse(graph.tokens, tuple(topics))

    def snapshot(self) -> dict:
        return {
            "turn_count": self.turn_count,
            "last_topic": self.last_topic,
            "last_speech_act": self.last_speech_act,
            "recent_entities": self.recent_entities[-8:],
        }
