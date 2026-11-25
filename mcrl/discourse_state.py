"""Bounded dialogue state with explicit topic updates and avoid-topic memory."""
from __future__ import annotations

from dataclasses import dataclass, field

ASSISTANT_NAMES = {"mcr", "mcrl", "assistant", "chatgpt"}
SECOND_PERSON = {"you", "your", "yours"}
FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our"}


@dataclass
class DiscourseState:
    """Session-local conversational references; no global topic leakage."""

    turn_count: int = 0
    last_topic: str = ""
    last_speech_act: str = ""
    recent_entities: list[str] = field(default_factory=list)
    avoided_topics: set[str] = field(default_factory=set)
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
        return self.last_topic or None

    def remember_topic(self, topic: str) -> None:
        cleaned = " ".join(str(topic or "").split()).strip()
        if not cleaned:
            return
        self.last_topic = cleaned
        if cleaned not in self.recent_entities:
            self.recent_entities.append(cleaned)
        self.recent_entities = self.recent_entities[-12:]

    def avoid_topic(self, topic: str) -> None:
        cleaned = " ".join(str(topic or "").lower().split()).strip()
        if cleaned:
            self.avoided_topics.add(cleaned)

    def topic_is_avoided(self, topic: str) -> bool:
        normalized = " ".join(str(topic or "").lower().split())
        return bool(normalized and normalized in self.avoided_topics)

    def observe_graph(self, graph) -> None:
        """Advance turn state, but only explicit topic nodes may replace context."""
        self.turn_count += 1
        self.last_speech_act = "+".join(graph.speech_acts) or graph.speech_act
        resolved = [
            node.value for node in graph.nodes
            if node.concept in {"topic.target", "topic.resolved"}
        ]
        if resolved and graph.speech_act not in {"request.avoid_topic"}:
            self.remember_topic(resolved[0])

    def snapshot(self) -> dict:
        return {
            "turn_count": self.turn_count,
            "last_topic": self.last_topic,
            "last_speech_act": self.last_speech_act,
            "recent_entities": self.recent_entities[-8:],
            "avoided_topics": sorted(self.avoided_topics),
        }
