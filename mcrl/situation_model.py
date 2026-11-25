"""Compact, explicit model of the current conversational situation.

This is not a persistent profile or a source of guessed personal facts. It records
only the structure of the active interaction: moves, roles, declared topics, goals,
tone, and how MCR-L responded. It is bounded so ordinary CPU/memory costs stay
predictable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TurnRecord:
    index: int
    user_moves: tuple[str, ...]
    user_goal: str
    topic: str
    speaker: str
    addressee: str
    assistant_action: str
    outcome: str
    user_text: str = ""
    assistant_text: str = ""


def _goal_for(moves: tuple[str, ...], intent: str) -> str:
    joined = set(moves)
    if "request.user_identity" in joined:
        return "ask_about_user_identity"
    if "request.identity" in joined:
        return "ask_about_assistant_identity"
    if "request.capability" in joined:
        return "check_assistant_capability"
    if "request.avoid_topic" in joined:
        return "set_topic_boundary"
    if any(move in joined for move in ("request.topic_information", "request.information")):
        return "obtain_information"
    if any(move.startswith("social.") for move in moves):
        return "social_interaction"
    if intent == "instruction":
        return "complete_requested_action"
    return "continue_conversation"


@dataclass
class DialogueSituation:
    """A small, inspectable temporal state for interaction planning."""

    turn_count: int = 0
    active_goal: str = ""
    current_topic: str = ""
    last_user_moves: tuple[str, ...] = ()
    last_speaker: str = "user"
    last_addressee: str = "assistant"
    last_assistant_action: str = ""
    last_outcome: str = ""
    interaction_tone: str = "neutral"
    pending_question: bool = False
    recent_turns: list[TurnRecord] = field(default_factory=list)
    history_limit: int = 12

    def observe(
        self,
        graph: Any,
        assistant_action: str,
        outcome: str,
        user_text: str = "",
        assistant_text: str = "",
    ) -> None:
        """Commit one fully handled user turn to the situation state."""
        moves = tuple(getattr(graph, "speech_acts", ()) or (graph.speech_act,))
        self.turn_count += 1
        self.last_user_moves = moves
        self.active_goal = _goal_for(moves, getattr(graph, "intent", "statement"))

        topics = [
            node.value for node in getattr(graph, "nodes", ())
            if getattr(node, "concept", "") in {"topic.target", "topic.resolved"}
        ]
        if topics:
            self.current_topic = topics[0]
        elif any(move in {
            "request.identity", "request.user_identity", "request.capability",
            "request.conversation_summary", "request.perform_action", "request.action",
        } for move in moves):
            self.current_topic = ""
        # Ordinary social turns preserve a referent, while a new explicit task
        # changes the conversational focus even when it has no world-topic node.
        speaker_nodes = graph.edge_targets("speaker") if hasattr(graph, "edge_targets") else ()
        addressee_nodes = graph.edge_targets("addressee") if hasattr(graph, "edge_targets") else ()
        if speaker_nodes:
            self.last_speaker = speaker_nodes[0].value
        elif any(move.startswith("request.") for move in moves):
            self.last_speaker = "user"
        if addressee_nodes:
            self.last_addressee = addressee_nodes[0].value
        elif any(move.startswith("request.") for move in moves):
            self.last_addressee = "assistant"

        if "social.compliment" in moves or "social.gratitude" in moves:
            self.interaction_tone = "positive"
        elif any(getattr(node, "kind", "") == "affect"
                 and getattr(node, "value", "") == "negative"
                 for node in getattr(graph, "nodes", ())):
            self.interaction_tone = "negative"
        elif any(move in {"social.greeting", "social.wellbeing", "social.farewell"}
                 for move in moves):
            self.interaction_tone = "warm-neutral"

        self.last_assistant_action = assistant_action
        self.last_outcome = outcome
        # A record is committed only after a response action was selected; the
        # current user question is therefore no longer pending in this turn.
        self.pending_question = False
        self.recent_turns.append(TurnRecord(
            index=self.turn_count,
            user_moves=moves,
            user_goal=self.active_goal,
            topic=self.current_topic,
            speaker=self.last_speaker,
            addressee=self.last_addressee,
            assistant_action=assistant_action,
            outcome=outcome,
            user_text=" ".join(str(user_text or "").split()),
            assistant_text=" ".join(str(assistant_text or "").split()),
        ))
        self.recent_turns = self.recent_turns[-self.history_limit:]

    def record_action(
        self,
        action: str,
        outcome: str,
        user_text: str = "",
        assistant_text: str = "",
    ) -> None:
        """Record a handled non-linguistic capability, such as arithmetic."""
        self.turn_count += 1
        self.active_goal = "execute_capability"
        self.last_user_moves = ("request.action",)
        self.last_speaker = "user"
        self.last_addressee = "assistant"
        self.last_assistant_action = action
        self.last_outcome = outcome
        self.pending_question = False
        self.recent_turns.append(TurnRecord(
            index=self.turn_count,
            user_moves=self.last_user_moves,
            user_goal=self.active_goal,
            topic="",
            speaker="user",
            addressee="assistant",
            assistant_action=action,
            outcome=outcome,
            user_text=" ".join(str(user_text or "").split()),
            assistant_text=" ".join(str(assistant_text or "").split()),
        ))
        self.recent_turns = self.recent_turns[-self.history_limit:]

    def snapshot(self) -> dict:
        return {
            "turn_count": self.turn_count,
            "active_goal": self.active_goal,
            "current_topic": self.current_topic,
            "last_user_moves": list(self.last_user_moves),
            "last_speaker": self.last_speaker,
            "last_addressee": self.last_addressee,
            "last_assistant_action": self.last_assistant_action,
            "last_outcome": self.last_outcome,
            "interaction_tone": self.interaction_tone,
            "pending_question": self.pending_question,
            "recent_turns": [
                {
                    "index": turn.index,
                    "user_moves": list(turn.user_moves),
                    "user_goal": turn.user_goal,
                    "topic": turn.topic,
                    "speaker": turn.speaker,
                    "addressee": turn.addressee,
                    "assistant_action": turn.assistant_action,
                    "outcome": turn.outcome,
                    "user_text": turn.user_text,
                    "assistant_text": turn.assistant_text,
                }
                for turn in self.recent_turns
            ],
        }
