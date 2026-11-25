"""Compositional interaction planner for commands, actions, and chat history.

The planner turns clause structure into an explicit operation before factual
retrieval. It is intentionally separate from world knowledge: "smile" is an
action request, not a subject named Smile; a request to recap a chat is answered
from the session ledger, not from a knowledge database.

This module is a bounded grammar, not a general language model. Its diagnostics
explain the structural cues and precedence constraints used for each operation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[?!.,;:]")
INTERJECTIONS = {
    "wow", "wowow", "whoa", "woah", "omg", "yay", "bruh", "hmm", "huh",
    "damn", "oops", "yikes", "ah", "oh", "lol", "haha", "hehe", "lmao",
}
ACTION_OUTPUTS = {
    "smile": "😊",
    "wave": "👋",
    "laugh": "😄",
    "wink": "😉",
    "clap": "👏",
    "cheer": "🎉",
    "dance": "💃",
    "bow": "🙇",
}
ACTION_VERBS = set(ACTION_OUTPUTS) | {"sing", "whistle", "cry", "think", "joke"}
SPEECH_OUTPUT_RE = re.compile(
    r"(?is)\b(?:just\s+|only\s+)?"
    r"(?:say|reply\s+with|respond\s+with|answer\s+with)\s+"
    r"(?P<payload>.+?)\s*(?:[.!?]+)?$"
)
SPEECH_OUTPUT_PREFIXES = (
    ("just", "say"), ("only", "say"), ("say",),
    ("just", "reply", "with"), ("reply", "with"),
    ("just", "respond", "with"), ("respond", "with"),
    ("just", "answer", "with"), ("answer", "with"),
)
HISTORY_CUES = {
    "talk", "talked", "discuss", "discussed", "cover", "covered",
    "conversation", "chat", "recap", "summarize", "summary",
}
HISTORY_TIME_CUES = {"so", "far", "till", "until", "now", "before", "earlier"}
CONTRACTIONS = {
    "u": "you", "dont": "do not", "don't": "do not",
    "cant": "can not", "can't": "can not", "wont": "will not", "won't": "will not",
    "youre": "you are", "you're": "you are",
}


@dataclass(frozen=True)
class InteractionPlan:
    operation: str
    payload: str = ""
    action: str = ""
    constraints: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    precedence: int = 0
    support: float = 0.0

    def diagnostics(self) -> dict:
        return {
            "operation": self.operation,
            "payload": self.payload,
            "action": self.action,
            "constraints": list(self.constraints),
            "evidence": list(self.evidence),
            "precedence": self.precedence,
            "support": self.support,
            "score_semantics": "structural cue support, not calibrated probability",
        }


def normalize_words(text: str) -> tuple[str, ...]:
    tokens = [part.lower() for part in TOKEN_RE.findall(text or "")
              if part not in {"?", "!", ".", ",", ";", ":"}]
    expanded: list[str] = []
    for token in tokens:
        expanded.extend(CONTRACTIONS.get(token, token).split())
    return tuple(expanded)


def _has_phrase(words: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    width = len(phrase)
    return any(words[i:i + width] == phrase for i in range(len(words) - width + 1))


def _is_history_question(words: tuple[str, ...]) -> bool:
    has_history_object = bool(set(words) & HISTORY_CUES)
    has_collective_reference = bool(set(words) & {"we", "our", "us"})
    has_question = bool(set(words) & {"what", "which", "summarize", "recap"})
    has_time_scope = bool(set(words) & HISTORY_TIME_CUES) or "conversation" in words or "chat" in words
    return (
        has_history_object
        and has_question
        and (has_collective_reference or "conversation" in words or "chat" in words)
        and has_time_scope
    )


def _extract_speech_payload(text: str) -> str:
    match = SPEECH_OUTPUT_RE.search(text or "")
    if not match:
        return ""
    payload = match.group("payload").strip()
    # Remove only wrapper quotes and terminal command punctuation. Preserve the
    # user's requested words, casing, and internal punctuation.
    payload = payload.strip().strip('"“”\'‘’').strip()
    payload = re.sub(r"\s+(?:and\s+)?nothing\s+else$", "", payload, flags=re.I).strip()
    return payload.rstrip(" .!?;,")


def _is_explicit_output_command(text: str, words: tuple[str, ...]) -> bool:
    payload = _extract_speech_payload(text)
    if not payload:
        return False
    return any(_has_phrase(words, prefix) for prefix in SPEECH_OUTPUT_PREFIXES)


def _is_prohibited_identity_disclosure(words: tuple[str, ...]) -> bool:
    negated = _has_phrase(words, ("do", "not")) or _has_phrase(words, ("will", "not"))
    disclosure_verb = bool(set(words) & {"tell", "say", "explain", "describe", "reveal"})
    identity_content = (
        _has_phrase(words, ("who", "you", "are"))
        or _has_phrase(words, ("who", "are", "you"))
        or _has_phrase(words, ("what", "you", "are"))
        or _has_phrase(words, ("what", "are", "you"))
        or _has_phrase(words, ("your", "name"))
        or "yourself" in words
    )
    return negated and disclosure_verb and identity_content


def _requested_action(words: tuple[str, ...]) -> str:
    """Extract an action requested of the assistant, not a topic noun."""
    modal_frame = (
        _has_phrase(words, ("can", "you"))
        or _has_phrase(words, ("could", "you"))
        or _has_phrase(words, ("would", "you"))
        or _has_phrase(words, ("please",))
    )
    beneficiary = "for" in words and "me" in words
    for index, word in enumerate(words):
        if word not in ACTION_VERBS:
            continue
        prefix = words[:index]
        imperative = index == 0 or prefix in {
            ("please",), ("just",), ("please", "just"),
        }
        if modal_frame or beneficiary or imperative:
            return word
    return ""


def _is_standalone_interjection(words: tuple[str, ...]) -> bool:
    return bool(words) and len(words) <= 3 and all(word in INTERJECTIONS for word in words)


def plan_interaction(text: str) -> InteractionPlan | None:
    """Compile an utterance into one highest-priority conversational operation.

    Priority is structural: explicit requested speech output outranks a quoted
    or embedded identity question; then concrete assistant actions; then
    conversation-history queries; then standalone reactions. Unsupported
    content falls through to the semantic/factual engine.
    """
    words = normalize_words(text)
    if not words:
        return None

    if _is_explicit_output_command(text, words):
        payload = _extract_speech_payload(text)
        blocked_identity = _is_prohibited_identity_disclosure(words)
        constraints = ["emit_requested_content_verbatim", "do_not_answer_embedded_content"]
        evidence = ["explicit_speech_output_verb", "nonempty_user_supplied_payload"]
        if blocked_identity:
            constraints.append("respect_negated_identity_disclosure")
            evidence.append("negative_instruction_scopes_over_identity_disclosure")
        return InteractionPlan(
            operation="emit_user_requested_text",
            payload=payload,
            constraints=tuple(constraints),
            evidence=tuple(evidence),
            precedence=100,
            support=1.0,
        )

    action = _requested_action(words)
    if action:
        supported = action in ACTION_OUTPUTS
        return InteractionPlan(
            operation="perform_symbolic_action" if supported else "admit_action_limit",
            action=action,
            payload=ACTION_OUTPUTS.get(action, ""),
            constraints=(
                "interpret_verb_as_requested_action_not_factual_subject",
                "do_not_claim_physical_body_or_physical_action",
            ),
            evidence=tuple(
                cue for cue, present in (
                    ("assistant_directed_modal_or_benefactive_frame", True),
                    ("recognized_action_predicate", True),
                    ("beneficiary_for_user", bool("for" in words and "me" in words)),
                ) if present
            ),
            precedence=80,
            support=0.95 if supported else 0.82,
        )

    if _is_history_question(words):
        return InteractionPlan(
            operation="summarize_session_history",
            constraints=(
                "use_only_recorded_session_turns",
                "do_not_invent_unrecorded_topics",
                "distinguish_user_requests_from_assistant_responses",
            ),
            evidence=("collective_conversation_reference", "history_predicate", "temporal_scope"),
            precedence=70,
            support=0.95,
        )

    if _is_standalone_interjection(words):
        return InteractionPlan(
            operation="acknowledge_interjection",
            payload=" ".join(words),
            constraints=("treat_as_social_reaction_not_factual_topic",),
            evidence=("standalone_interjection",),
            precedence=40,
            support=0.90,
        )

    return None


def summarize_session(turns: Iterable[object], limit: int = 8) -> str:
    """Produce an evidence-bounded recap from actual session records."""
    records = [
        turn for turn in turns
        if getattr(turn, "user_text", "").strip()
    ][-max(1, limit):]
    if not records:
        return "We haven't exchanged any recorded messages yet. This session's conversation history is empty."

    lines = ["Here's the conversation I can see in this session:"]
    for index, turn in enumerate(records, start=1):
        user_text = " ".join(str(turn.user_text).split())
        assistant_text = " ".join(str(getattr(turn, "assistant_text", "")).split())
        if len(user_text) > 180:
            user_text = user_text[:177].rstrip() + "..."
        lines.append(f'{index}. You: "{user_text}"')
        if assistant_text:
            if len(assistant_text) > 180:
                assistant_text = assistant_text[:177].rstrip() + "..."
            lines.append(f'   Me: "{assistant_text}"')
    lines.append("That's a recap of the messages actually recorded here; I won't fill gaps with guesses.")
    return "\n".join(lines)
