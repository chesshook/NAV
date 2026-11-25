"""Compositional parser for conversational acts, topic requests, and simple events.

This parser builds a typed graph. It deliberately preserves grammar words until
speech-act and event patterns have been checked, so words such as "are", "there",
and "you" cannot disappear before the parser decides what they mean.
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
    "nice": "nice", "glad": "glad", "pleased": "pleased", "happy": "happy",
    "great": "great", "good": "good", "wonderful": "wonderful",
    "delighted": "delighted", "pleasure": "pleased",
}
MEETING_WORDS = {"meet", "meets", "meeting", "met"}
GREETING_WORDS = {"hi", "hello", "hey", "howdy", "yo"}
GRATITUDE_WORDS = {"thanks", "thank", "thx", "appreciate", "grateful", "welcome"}
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
    "tell": "tell", "say": "say", "ask": "ask", "check": "check",
}
ASSISTANT_ALIASES = {"mcr", "mcrl", "assistant", "chatgpt"}
SECOND_PERSON = {"you", "u", "ya", "your", "yours"}
FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our"}
RECIPROCITY_WORDS = {"too", "also", "likewise", "either", "as-well"}
PRESENCE_CUES = {"there", "around", "online", "available", "here"}
ANAPHORS = {"it", "this", "that", "they", "them", "those", "these"}

TOPIC_REQUEST_PREFIXES = (
    ("what", "can", "you", "tell", "me", "about"),
    ("what", "do", "you", "know", "about"),
    ("tell", "me", "something", "about"),
    ("tell", "me", "one", "thing", "about"),
    ("give", "me", "facts", "about"),
    ("say", "something", "about"),
    ("tell", "me", "about"),
    ("anything", "about"),
    ("something", "about"),
    ("what", "about"),
)


def _has(words: tuple[str, ...], choices: set[str]) -> bool:
    return any(word in choices for word in words)


def _contains_phrase(words: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    width = len(phrase)
    return any(words[i:i + width] == phrase for i in range(len(words) - width + 1))


def _topic_prefix_length(words: tuple[str, ...]) -> int:
    for prefix in TOPIC_REQUEST_PREFIXES:
        if words[:len(prefix)] == prefix:
            return len(prefix)
    return 0


def _is_presence_check(words: tuple[str, ...], has_question_mark: bool) -> bool:
    has_person = _has(words, SECOND_PERSON) or _has(words, ASSISTANT_ALIASES)
    has_presence = _has(words, PRESENCE_CUES)
    has_question_form = has_question_mark or _has(words, {"are", "is", "can", "still", "you"})
    return has_person and has_presence and has_question_form


def _is_presence_confirmation(words: tuple[str, ...]) -> bool:
    return (
        _contains_phrase(words, ("i", "am", "here"))
        or _contains_phrase(words, ("i'm", "here"))
        or _contains_phrase(words, ("im", "here"))
        or _contains_phrase(words, ("i", "am", "online"))
        or _contains_phrase(words, ("i'm", "online"))
    )


def parse_meaning(text: str, discourse: Any | None = None) -> SemanticGraph:
    original = (text or "").strip()
    parts = TOKEN_RE.findall(original)
    words = tuple(part.lower() for part in parts if part not in {"?", "!", ".", ",", ";", ":"})
    normalized = " ".join(words)
    topic_prefix = _topic_prefix_length(words)
    topic_source = words[topic_prefix:] if topic_prefix else words

    lexical_tokens = tuple(
        word for word in topic_source
        if word not in STOPWORDS and len(word) > 1
    )
    # Resolve simple anaphors against the most recent explicit discourse topic.
    resolved_topic = ""
    if not lexical_tokens and discourse is not None and _has(words, ANAPHORS):
        resolved_topic = discourse.last_topic

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
    has_question_mark = "?" in parts
    presence_check = _is_presence_check(words, has_question_mark)
    presence_confirmation = _is_presence_confirmation(words)
    negated = _has(words, NEGATION_WORDS)
    temporal = "before" if _has(words, BEFORE_WORDS) else "after" if _has(words, AFTER_WORDS) else "present" if _has(words, PRESENT_WORDS) else "unspecified"
    comparison = "more_than" if _has(words, MORE_WORDS) and "than" in words else "less_than" if _has(words, LESS_WORDS) and "than" in words else "none"
    verb_position = next((index for index, word in enumerate(words) if word in VERBS), None)
    main_verb = words[verb_position] if verb_position is not None else None

    if presence_check:
        speech_act = "request.presence_check"
        intent = "presence_check"
    elif topic_prefix:
        speech_act = "request.topic_information"
        intent = "topic_information"
    elif meeting and modifiers and not negated:
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
    elif presence_confirmation:
        speech_act = "response.presence_confirmation"
        intent = "statement"
    elif (has_question_mark or (words and words[0] in QUESTION_WORDS)) and not lexical_tokens and not resolved_topic:
        speech_act = "request.clarification"
        intent = "clarification"
    elif has_question_mark or (words and words[0] in QUESTION_WORDS):
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

    act_id = add(speech_act, "speech_act", speech_act)
    if presence_check:
        event_id = add("event.presence_check", "event", "presence_check")
        builder.edge(act_id, event_id, "has_event")
    elif presence_confirmation:
        event_id = add("event.presence_confirmation", "event", "presence_confirmation")
        builder.edge(act_id, event_id, "has_event")
    elif meeting:
        event_id = add("event.meeting", "event", "meeting", next((w for w in words if w in MEETING_WORDS), "meet"))
        builder.edge(act_id, event_id, "has_event")
    elif greeting or gratitude or farewell or wellbeing:
        social_event = (
            "greeting" if greeting else "gratitude" if gratitude
            else "farewell" if farewell else "wellbeing_question"
        )
        event_id = add(f"event.{social_event}", "event", social_event)
        builder.edge(act_id, event_id, "has_event")
    elif main_verb is not None:
        event_id = add(f"event.{VERBS[main_verb]}", "event", VERBS[main_verb], main_verb)
        builder.edge(act_id, event_id, "has_event")
    else:
        event_id = None

    if modifiers:
        modifier_word, _modifier_value = modifiers[0]
        affect_id = add("affect.positive", "affect", "positive", modifier_word)
        if event_id is not None:
            builder.edge(event_id, affect_id, "has_modifier")

    if greeting:
        add("act.greet", "speech_content", "greeting", next((w for w in words if w in GREETING_WORDS), "hello"))
        if "there" in words:
            add("marker.salutation_extension", "discourse_marker", "there", "there")
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
    elif speech_act.startswith("social.") or speech_act == "response.presence_confirmation":
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

    has_second_person = _has(words, SECOND_PERSON)
    mentions_assistant = _has(words, ASSISTANT_ALIASES)
    has_first_person = _has(words, FIRST_PERSON)
    if discourse is not None:
        has_second_person = has_second_person or discourse.has_assistant_reference(words)
        mentions_assistant = mentions_assistant or discourse.has_assistant_reference(words)

    user_id = (
        add("entity.user", "participant", "user")
        if has_first_person or speech_act.startswith("social.") or speech_act in {"request.presence_check", "response.presence_confirmation", "request.action"}
        else None
    )
    assistant_id = (
        add("entity.assistant", "participant", "assistant")
        if has_second_person or mentions_assistant or speech_act.startswith("social.") or speech_act in {"request.presence_check", "response.presence_confirmation"}
        else None
    )

    if event_id is not None:
        if speech_act.startswith("social."):
            if user_id is not None:
                builder.edge(event_id, user_id, "speaker")
            if assistant_id is not None:
                builder.edge(event_id, assistant_id, "addressee")
        elif speech_act == "request.presence_check":
            if user_id is not None:
                builder.edge(event_id, user_id, "asker")
            if assistant_id is not None:
                builder.edge(event_id, assistant_id, "target")
        elif speech_act == "response.presence_confirmation":
            if assistant_id is not None:
                builder.edge(event_id, assistant_id, "speaker")
            if user_id is not None:
                builder.edge(event_id, user_id, "recipient")
        elif verb_position is not None:
            before_verb = words[:verb_position]
            after_verb = words[verb_position + 1:]
            if user_id is not None and any(word in FIRST_PERSON for word in before_verb):
                builder.edge(event_id, user_id, "agent")
            if assistant_id is not None and any(word in SECOND_PERSON or word in ASSISTANT_ALIASES for word in before_verb):
                builder.edge(event_id, assistant_id, "agent")
            if user_id is not None and any(word in FIRST_PERSON for word in after_verb):
                builder.edge(event_id, user_id, "patient")
            if assistant_id is not None and any(word in SECOND_PERSON or word in ASSISTANT_ALIASES for word in after_verb):
                builder.edge(event_id, assistant_id, "patient")
            if speech_act == "request.action" and user_id is not None and not before_verb:
                builder.edge(event_id, user_id, "agent")

    if mentions_assistant:
        add("entity.assistant", "entity", "MCR", "MCR")

    if main_verb is not None:
        add(f"predicate.{VERBS[main_verb]}", "predicate", VERBS[main_verb], main_verb)

    if resolved_topic:
        add("topic.resolved", "topic", resolved_topic, next((word for word in words if word in ANAPHORS), "it"), 0.85)
    for word in lexical_tokens:
        add("topic.lexical", "topic", word, word, 0.70)

    known = (
        set(POSITIVE_MODIFIERS) | MEETING_WORDS | GREETING_WORDS | GRATITUDE_WORDS
        | FAREWELL_WORDS | NEGATION_WORDS | BEFORE_WORDS | AFTER_WORDS | PRESENT_WORDS
        | MORE_WORDS | LESS_WORDS | set(VERBS) | ASSISTANT_ALIASES | SECOND_PERSON
        | FIRST_PERSON | QUESTION_WORDS | STOPWORDS
    )
    for raw in parts:
        lower = raw.lower()
        if raw[:1].isupper() and lower not in known and len(raw) > 1:
            add("entity.named", "entity", raw, raw)

    if discourse is not None:
        discourse.observe_parse(
            words,
            tuple(node.value for node in builder.nodes if node.kind in {"topic", "entity"}),
        )
    return builder.build(original, normalized, words, lexical_tokens, intent)
