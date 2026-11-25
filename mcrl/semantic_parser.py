"""Compositional parser for the first bounded MCR-L grammar.

The parser builds a typed graph from lexical concepts and their roles. It uses
multiword/lexical meaning plus small grammar rules, not whole-input reply lookup.
"""
from __future__ import annotations

import re
from typing import Any

from .meaning_graph import MeaningGraphBuilder, SemanticGraph

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[?!.,;:]")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i",
    "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "please", "so", "that", "the", "their", "them", "there", "these", "this",
    "to", "us", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",
}

POSITIVE_MODIFIERS = {
    "nice": "nice",
    "glad": "glad",
    "pleased": "pleased",
    "happy": "happy",
    "great": "great",
    "good": "good",
    "wonderful": "wonderful",
    "delighted": "delighted",
    "pleasure": "pleased",
}
MEETING_WORDS = {"meet", "meets", "meeting", "met"}
GREETING_WORDS = {"hi", "hello", "hey", "howdy", "yo"}
GRATITUDE_WORDS = {"thanks", "thank", "thx", "appreciate", "grateful"}
FAREWELL_WORDS = {"bye", "goodbye", "farewell", "later"}
NEGATION_WORDS = {"not", "never", "no", "cannot", "can't", "isn't", "aren't", "doesn't", "don't", "without"}
BEFORE_WORDS = {"before", "earlier", "previously"}
AFTER_WORDS = {"after", "later", "subsequently"}
PRESENT_WORDS = {"now", "today", "currently"}
MORE_WORDS = {"more", "greater", "higher", "faster", "stronger", "better"}
LESS_WORDS = {"less", "lower", "slower", "weaker", "worse"}
QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which"}
VERBS = {
    "build": "build", "built": "build", "make": "make", "made": "make",
    "create": "create", "created": "create", "fix": "fix", "fixed": "fix",
    "explain": "explain", "explains": "explain", "learn": "learn",
    "learned": "learn", "understand": "understand", "understood": "understand",
    "like": "like", "likes": "like", "love": "love", "loves": "love",
    "want": "want", "wants": "want", "need": "need", "needs": "need",
    "think": "think", "thinks": "think", "know": "know", "knows": "know",
    "work": "work", "works": "work", "run": "run", "runs": "run",
    "use": "use", "uses": "use", "compare": "compare", "compares": "compare",
    "help": "help", "helps": "help", "delete": "delete", "remove": "remove",
    "install": "install", "open": "open", "play": "play", "prefer": "prefer",
}
ASSISTANT_ALIASES = {"mcr", "mcrl", "assistant", "chatgpt"}
SECOND_PERSON = {"you", "u", "ya", "your", "yours"}
FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our"}
RECIPROCITY_WORDS = {"too", "also", "likewise", "either", "as-well"}


def _has(words: tuple[str, ...], choices: set[str]) -> bool:
    return any(word in choices for word in words)


def _contains_phrase(words: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    width = len(phrase)
    return any(words[i:i + width] == phrase for i in range(len(words) - width + 1))


def parse_meaning(text: str, discourse: Any | None = None) -> SemanticGraph:
    original = (text or "").strip()
    parts = TOKEN_RE.findall(original)
    words = tuple(part.lower() for part in parts if part not in {"?", "!", ".", ",", ";", ":"})
    normalized = " ".join(words)
    lexical_tokens = tuple(
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    )

    builder = MeaningGraphBuilder()
    node_ids: dict[str, int] = {}

    def add(concept: str, kind: str, value: str, surface: str = "", confidence: float = 1.0) -> int:
        key = f"{kind}:{value}"
        if key in node_ids:
            return node_ids[key]
        node_id = builder.node(concept, kind, value, confidence, surface)
        node_ids[key] = node_id
        return node_id

    modifiers = [(word, POSITIVE_MODIFIERS[word]) for word in words if word in POSITIVE_MODIFIERS]
    meeting = _has(words, MEETING_WORDS)
    greeting = _has(words, GREETING_WORDS)
    gratitude = _has(words, GRATITUDE_WORDS) or _contains_phrase(words, ("thank", "you"))
    farewell = _has(words, FAREWELL_WORDS) or _contains_phrase(words, ("see", "you"))
    wellbeing = _contains_phrase(words, ("how", "are", "you")) or _contains_phrase(words, ("how", "is", "it", "going"))
    identity = (
        _contains_phrase(words, ("who", "are", "you"))
        or _contains_phrase(words, ("what", "are", "you"))
        or _contains_phrase(words, ("what", "is", "your", "name"))
    )
    negated = _has(words, NEGATION_WORDS)
    temporal = "before" if _has(words, BEFORE_WORDS) else "after" if _has(words, AFTER_WORDS) else "present" if _has(words, PRESENT_WORDS) else "unspecified"
    comparison = "more_than" if _has(words, MORE_WORDS) and "than" in words else "less_than" if _has(words, LESS_WORDS) and "than" in words else "none"

    if meeting and modifiers:
        speech_act = "social.pleasantry"
        intent = "social"
    elif greeting:
        speech_act = "social.greeting"
        intent = "social"
    elif gratitude:
        speech_act = "social.gratitude"
        intent = "social"
    elif farewell:
        speech_act = "social.farewell"
        intent = "social"
    elif wellbeing:
        speech_act = "social.wellbeing"
        intent = "social"
    elif identity:
        speech_act = "request.identity"
        intent = "identity"
    elif "?" in parts or (words and words[0] in QUESTION_WORDS):
        speech_act = "request.information"
        intent = (
            "define" if words[:2] == ("what", "is") or words[:2] == ("what", "are")
            else "explain" if words and words[0] in {"why", "how"}
            else "question"
        )
    elif words and words[0] in {"please", "build", "make", "create", "fix", "explain", "show", "tell", "compare"}:
        speech_act = "request.action"
        intent = "instruction"
    else:
        speech_act = "assertion"
        intent = "statement"

    act_id = add(speech_act, "speech_act", speech_act.split(".")[-1])
    if meeting:
        event_id = add("event.meeting", "event", "meeting", next((w for w in words if w in MEETING_WORDS), "meet"))
        builder.edge(act_id, event_id, "has_event")
    else:
        event_id = None

    if modifiers:
        modifier_word, modifier_value = modifiers[0]
        affect_id = add("affect.positive", "affect", "positive", modifier_word)
        if event_id is not None:
            builder.edge(event_id, affect_id, "has_modifier")

    if greeting:
        add("act.greet", "speech_content", "greeting", next((w for w in words if w in GREETING_WORDS), "hello"))
    if gratitude:
        add("act.thank", "speech_content", "gratitude")
    if farewell:
        add("act.farewell", "speech_content", "farewell")
    if wellbeing:
        add("act.ask_wellbeing", "speech_content", "wellbeing_question")

    if negated:
        neg_id = add("operator.negation", "polarity", "negative")
        if event_id is not None:
            builder.edge(neg_id, event_id, "negates")
    else:
        add("operator.affirmation", "polarity", "positive")

    if temporal != "unspecified":
        temporal_id = add(f"temporal.{temporal}", "temporal", temporal)
        if event_id is not None:
            builder.edge(temporal_id, event_id, "temporal_scope")

    if comparison != "none":
        comparison_id = add(f"comparison.{comparison}", "comparison", comparison)
        if event_id is not None:
            builder.edge(comparison_id, event_id, "compares")

    if _has(words, RECIPROCITY_WORDS):
        reciprocal_id = add("operator.reciprocity", "operator", "reciprocal")
        if event_id is not None:
            builder.edge(event_id, reciprocal_id, "reciprocated_by")

    # Participant resolution is semantic: grammatical person and the alias
    # refer to discourse roles, not search keywords.
    has_second_person = _has(words, SECOND_PERSON)
    mentions_assistant = _has(words, ASSISTANT_ALIASES)
    has_first_person = _has(words, FIRST_PERSON)
    if discourse is not None:
        has_second_person = has_second_person or discourse.has_assistant_reference(words)
        mentions_assistant = mentions_assistant or discourse.has_assistant_reference(words)

    user_id = add("entity.user", "participant", "user") if has_first_person or speech_act.startswith("social.") else None
    assistant_id = add("entity.assistant", "participant", "assistant") if has_second_person or mentions_assistant or speech_act.startswith("social.") else None
    if event_id is not None:
        if user_id is not None:
            builder.edge(event_id, user_id, "speaker")
        if assistant_id is not None:
            builder.edge(event_id, assistant_id, "addressee")

    # Store the explicit assistant alias as a reference, not as a topical noun.
    if mentions_assistant:
        add("entity.assistant", "entity", "MCR", "MCR")

    for word in words:
        if word in VERBS:
            verb_id = add(f"event.{VERBS[word]}", "predicate", VERBS[word], word)
            if event_id is None:
                event_id = verb_id
            break

    # Keep named surface entities and unresolved content as topic nodes.
    known = set(POSITIVE_MODIFIERS) | MEETING_WORDS | GREETING_WORDS | GRATITUDE_WORDS | FAREWELL_WORDS | NEGATION_WORDS | BEFORE_WORDS | AFTER_WORDS | PRESENT_WORDS | MORE_WORDS | LESS_WORDS | set(VERBS) | ASSISTANT_ALIASES | SECOND_PERSON | FIRST_PERSON
    for raw in parts:
        lower = raw.lower()
        if raw[:1].isupper() and lower not in known and len(raw) > 1:
            add("entity.named", "entity", raw, raw)
    for word in lexical_tokens:
        if word not in known and not any(node.value.lower() == word for node in builder.nodes):
            add("topic.lexical", "topic", word, word, 0.65)

    if discourse is not None:
        discourse.observe_parse(words, tuple(node.value for node in builder.nodes if node.kind in {"topic", "entity"}))

    return builder.build(original, normalized, words, lexical_tokens, intent)
