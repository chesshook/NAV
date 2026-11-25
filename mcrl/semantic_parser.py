"""Bounded compositional parser with topic isolation and dialogue-act cues.

The parser keeps grammatical words until construction analysis is complete,
expands common chat contractions, and emits explicit topic targets. It never
turns every content word into a world topic or silently borrows old topics.
"""
from __future__ import annotations

import re
from typing import Any

from .meaning_graph import MeaningGraphBuilder, SemanticGraph
from .pragmatics import analyze_pragmatics

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[?!.,;:]")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i",
    "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "please", "so", "that", "the", "their", "them", "there", "these", "this",
    "to", "us", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",
}
CONTRACTIONS = {
    "u": ("you",),
    "ya": ("you",),
    "thats": ("that", "is"),
    "that's": ("that", "is"),
    "dont": ("do", "not"),
    "don't": ("do", "not"),
    "doesnt": ("does", "not"),
    "doesn't": ("does", "not"),
    "isnt": ("is", "not"),
    "isn't": ("is", "not"),
    "arent": ("are", "not"),
    "aren't": ("are", "not"),
    "cant": ("can", "not"),
    "can't": ("can", "not"),
    "wont": ("will", "not"),
    "won't": ("will", "not"),
    "im": ("i", "am"),
    "i'm": ("i", "am"),
    "youre": ("you", "are"),
    "you're": ("you", "are"),
    "its": ("it", "is"),
    "it's": ("it", "is"),
    "what's": ("what", "is"),
    "how's": ("how", "is"),
}

POSITIVE_MODIFIERS = {
    "nice": "nice", "glad": "glad", "pleased": "pleased", "happy": "happy",
    "great": "great", "good": "good", "wonderful": "wonderful",
    "delighted": "delighted", "awesome": "awesome", "amazing": "amazing",
    "cool": "cool", "perfect": "perfect", "fantastic": "fantastic",
}
NEGATIVE_MODIFIERS = {
    "bad": "bad", "terrible": "terrible", "awful": "awful",
    "sad": "sad", "upset": "upset", "frustrating": "frustrating",
    "annoying": "annoying", "sucks": "sucks", "horrible": "horrible",
}
MEETING_WORDS = {"meet", "meets", "meeting", "met"}
GREETING_WORDS = {"hi", "hello", "hey", "howdy", "yo"}
GRATITUDE_WORDS = {"thanks", "thank", "thx", "appreciate", "grateful", "welcome"}
FAREWELL_WORDS = {"bye", "goodbye", "farewell", "later"}
NEGATION_WORDS = {"not", "never", "no", "cannot", "without"}
BEFORE_WORDS = {"before", "earlier", "previously"}
AFTER_WORDS = {"after", "later", "subsequently"}
PRESENT_WORDS = {"now", "today", "currently"}
MORE_WORDS = {"more", "greater", "higher", "faster", "stronger", "better"}
LESS_WORDS = {"less", "lower", "slower", "weaker", "worse"}
QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which"}
ASSISTANT_ALIASES = {"mcr", "mcrl", "assistant", "chatgpt"}
SECOND_PERSON = {"you", "your", "yours"}
FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our"}
RECIPROCITY_WORDS = {"too", "also", "likewise", "either", "as-well"}
PRESENCE_CUES = {"there", "around", "online", "available", "here"}
ANAPHORS = {"it", "this", "that", "they", "them", "those", "these"}

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
    "tell": "tell", "tells": "tell", "say": "say", "says": "say",
    "ask": "ask", "check": "check", "add": "add", "sum": "sum",
    "calculate": "calculate", "compute": "compute", "show": "show",
}

PROPERTY_WORDS = {
    "color", "colour", "white", "gray", "grey", "appearance", "look",
    "light", "shine", "sunlight", "glow", "bright", "size", "big", "large",
    "wide", "diameter", "small", "distance", "far", "near", "away",
    "weight", "heavy", "population", "capital", "height", "age", "old",
    "meaning", "definition", "cause", "reason", "history", "date", "price",
}
TOPIC_FILLER = {
    "tell", "know", "about", "anything", "something", "facts", "fact",
    "please", "give", "get", "one", "thing", "can", "could", "would",
    "do", "does", "did", "you", "your", "u", "me", "my", "what",
    "which", "who", "why", "how", "is", "are", "was", "were", "the",
    "a", "an", "of", "from", "to", "in", "on", "for", "and", "with",
    "much", "many", "there", "it", "its", "that", "this", "be",
}
TOPIC_PREFIXES = (
    ("what", "do", "you", "know", "about"),
    ("what", "do", "u", "know", "about"),
    ("what", "can", "you", "tell", "me", "about"),
    ("what", "can", "u", "tell", "me", "about"),
    ("can", "you", "tell", "me", "about"),
    ("can", "u", "tell", "me", "about"),
    ("tell", "me", "something", "about"),
    ("tell", "me", "one", "thing", "about"),
    ("give", "me", "facts", "about"),
    ("say", "something", "about"),
    ("tell", "me", "about"),
    ("anything", "about"),
    ("something", "about"),
    ("what", "about"),
)


def _expand(words: tuple[str, ...]) -> tuple[str, ...]:
    expanded: list[str] = []
    for word in words:
        expanded.extend(CONTRACTIONS.get(word, (word,)))
    return tuple(expanded)


def _has(words: tuple[str, ...], choices: set[str]) -> bool:
    return any(word in choices for word in words)


def _contains_phrase(words: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    width = len(phrase)
    return any(words[i:i + width] == phrase for i in range(len(words) - width + 1))


def _prefix_length(words: tuple[str, ...]) -> int:
    for prefix in TOPIC_PREFIXES:
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
        or _contains_phrase(words, ("i", "am", "online"))
        or _contains_phrase(words, ("i", "am", "around"))
    )


def _extract_topic(
    words: tuple[str, ...],
    explicit_about: bool,
    discourse: Any | None,
    has_question_mark: bool,
) -> tuple[str, bool]:
    """Return current-turn topic text and whether it was resolved anaphorically."""
    about_positions = [index for index, word in enumerate(words) if word == "about"]
    if about_positions and explicit_about:
        suffix = words[about_positions[-1] + 1:]
        terms = [word for word in suffix if word not in TOPIC_FILLER and not word.isdigit()]
        return " ".join(terms), False

    # For property questions like "what color is the Moon?" or relation
    # questions like "distance from Moon to Earth", retain the subject nouns,
    # not the question/property vocabulary.
    filtered = [
        word for word in words
        if word not in TOPIC_FILLER
        and word not in QUESTION_WORDS
        and word not in VERBS
        and word not in PROPERTY_WORDS
        and word not in POSITIVE_MODIFIERS
        and word not in NEGATIVE_MODIFIERS
        and word not in GREETING_WORDS
        and word not in GRATITUDE_WORDS
        and word not in FAREWELL_WORDS
        and word not in MEETING_WORDS
        and word not in ASSISTANT_ALIASES
        and word not in SECOND_PERSON
        and word not in FIRST_PERSON
        and word not in NEGATION_WORDS
        and not word.isdigit()
    ]
    if filtered:
        return " ".join(filtered), False

    # A prior topic may only be borrowed when the current utterance contains
    # an actual anaphor in a question-like construction. Never inherit it merely
    # because the last message happened to mention a topic.
    if has_question_mark and discourse is not None and _has(words, ANAPHORS):
        return discourse.last_topic, bool(discourse.last_topic)
    return "", False


def parse_meaning(text: str, discourse: Any | None = None) -> SemanticGraph:
    original = (text or "").strip()
    parts = TOKEN_RE.findall(original)
    raw_words = tuple(part.lower() for part in parts if part not in {"?", "!", ".", ",", ";", ":"})
    words = _expand(raw_words)
    normalized = " ".join(words)
    has_question_mark = "?" in parts
    indirect_question = (
        words[:3] in {("can", "you", "tell"), ("could", "you", "tell"), ("would", "you", "tell")}
        or words[:4] in {("can", "you", "explain", "the"), ("could", "you", "explain", "the")}
        or _contains_phrase(words, ("do", "you", "know"))
    )
    question_form = has_question_mark or bool(words and words[0] in QUESTION_WORDS) or indirect_question

    modifiers = [word for word in words if word in POSITIVE_MODIFIERS]
    negative_modifiers = [word for word in words if word in NEGATIVE_MODIFIERS]
    negated = _has(words, NEGATION_WORDS)
    greeting = _has(words, GREETING_WORDS)
    meeting = _has(words, MEETING_WORDS)
    gratitude = _has(words, GRATITUDE_WORDS) or _contains_phrase(words, ("thank", "you"))
    farewell = _has(words, FAREWELL_WORDS) or _contains_phrase(words, ("see", "you"))
    wellbeing = _contains_phrase(words, ("how", "are", "you")) or _contains_phrase(words, ("how", "is", "it", "going"))
    pragmatic = analyze_pragmatics(words)
    identity = pragmatic.assistant_identity
    user_identity = pragmatic.user_identity
    capability_target = pragmatic.capability_target
    compliment = pragmatic.compliment
    presence_check = _is_presence_check(words, has_question_mark)
    presence_confirmation = _is_presence_confirmation(words)
    prefix_length = _prefix_length(words)
    about_request = "about" in words and (
        question_form or _has(words, {"tell", "say", "give", "know", "anything", "something"})
    )
    if pragmatic.prevents_topic_extraction(
        greeting=greeting, wellbeing=wellbeing, gratitude=gratitude, farewell=farewell
    ) or meeting:
        topic_text, topic_is_resolved = "", False
    else:
        topic_text, topic_is_resolved = _extract_topic(words, about_request, discourse, question_form)
    property_words = [word for word in words if word in PROPERTY_WORDS]
    temporal = "before" if _has(words, BEFORE_WORDS) else "after" if _has(words, AFTER_WORDS) else "present" if _has(words, PRESENT_WORDS) else "unspecified"
    comparison = "more_than" if _has(words, MORE_WORDS) and "than" in words else "less_than" if _has(words, LESS_WORDS) and "than" in words else "none"
    verb_position = next((index for index, word in enumerate(words) if word in VERBS), None)
    main_verb = words[verb_position] if verb_position is not None else None

    avoid_topic = (
        negated and _has(words, {"tell", "say", "mention", "discuss"})
        and "about" in words and bool(topic_text)
    )
    positive_ack = (
        bool(modifiers)
        and (
            _has(words, {"that", "it", "this"})
            or len(words) <= 2
            or _has(words, {"thanks", "thank"})
        )
        and not greeting and not meeting
    )
    negative_ack = (
        bool(negative_modifiers)
        and _has(words, {"that", "it", "this"})
        and not negated
    )

    if presence_check:
        speech_act, intent = "request.presence_check", "presence_check"
    elif avoid_topic:
        speech_act, intent = "request.avoid_topic", "avoid_topic"
    elif user_identity:
        speech_act, intent = "request.user_identity", "identity"
    elif identity:
        speech_act, intent = "request.identity", "identity"
    elif capability_target:
        speech_act, intent = "request.capability", "capability"
    elif compliment:
        speech_act, intent = "social.compliment", "social"
    elif topic_text and about_request:
        speech_act, intent = "request.topic_information", "topic_information"
    elif meeting and modifiers and not negated:
        speech_act, intent = "social.pleasantry", "social"
    elif greeting:
        speech_act, intent = "social.greeting", "social"
    elif gratitude:
        speech_act, intent = "social.gratitude", "social"
    elif farewell:
        speech_act, intent = "social.farewell", "social"
    elif wellbeing:
        speech_act, intent = "social.wellbeing", "social"
    elif identity:
        speech_act, intent = "request.identity", "identity"
    elif presence_confirmation:
        speech_act, intent = "response.presence_confirmation", "statement"
    elif positive_ack:
        speech_act, intent = "social.positive_acknowledgement", "social"
    elif negative_ack:
        speech_act, intent = "social.empathetic_acknowledgement", "social"
    elif question_form and not topic_text:
        speech_act, intent = "request.clarification", "clarification"
    elif question_form:
        speech_act, intent = "request.information", "question"
    elif about_request and topic_text:
        speech_act, intent = "request.topic_information", "topic_information"
    elif len(words) == 1 and topic_text:
        speech_act, intent = "request.topic_information", "topic_information"
    elif words and words[0] in {"please", "build", "make", "create", "fix", "explain", "show", "tell", "compare"}:
        speech_act, intent = "request.action", "instruction"
    else:
        speech_act, intent = "assertion", "statement"

    builder = MeaningGraphBuilder()
    node_ids: dict[str, int] = {}

    def add(concept: str, kind: str, value: str, surface: str = "", confidence: float = 1.0) -> int:
        key = f"{kind}:{value}"
        if key in node_ids:
            return node_ids[key]
        node_id = builder.node(concept, kind, value, confidence, surface)
        node_ids[key] = node_id
        return node_id

    act_id = add(speech_act, "speech_act", speech_act)
    if presence_check:
        event_id = add("event.presence_check", "event", "presence_check")
    elif speech_act == "request.capability":
        event_id = add("event.capability_check", "event", "capability_check", capability_target)
    elif speech_act == "request.user_identity":
        event_id = add("event.user_identity_question", "event", "user_identity_question")
    elif speech_act == "request.identity":
        event_id = add("event.assistant_identity_question", "event", "assistant_identity_question")
    elif speech_act == "social.compliment":
        event_id = add("event.compliment", "event", "compliment")
    elif presence_confirmation:
        event_id = add("event.presence_confirmation", "event", "presence_confirmation")
    elif meeting:
        event_id = add("event.meeting", "event", "meeting", next((word for word in words if word in MEETING_WORDS), "meet"))
    elif greeting or gratitude or farewell or wellbeing:
        event_name = "greeting" if greeting else "gratitude" if gratitude else "farewell" if farewell else "wellbeing_question"
        event_id = add(f"event.{event_name}", "event", event_name)
    elif main_verb is not None:
        event_id = add(f"event.{VERBS[main_verb]}", "event", VERBS[main_verb], main_verb)
    elif positive_ack:
        event_id = add("event.positive_acknowledgement", "event", "positive_acknowledgement")
    elif negative_ack:
        event_id = add("event.empathetic_acknowledgement", "event", "empathetic_acknowledgement")
    else:
        event_id = None
    if event_id is not None:
        builder.edge(act_id, event_id, "has_event")

    # Keep independent moves in a compound turn instead of discarding all but
    # the first label. "Hi! How is it going?" has both acts in one graph.
    if greeting and wellbeing and speech_act == "social.greeting":
        wellbeing_act_id = add("social.wellbeing", "speech_act", "social.wellbeing")
        wellbeing_event_id = add(
            "event.wellbeing_question", "event", "wellbeing_question"
        )
        builder.edge(wellbeing_act_id, wellbeing_event_id, "has_event")

    if capability_target:
        add("capability.target", "capability", capability_target, capability_target)

    if modifiers:
        affect_id = add("affect.positive", "affect", "positive", modifiers[0])
        if event_id is not None:
            builder.edge(event_id, affect_id, "has_modifier")
    if negative_modifiers:
        affect_id = add("affect.negative", "affect", "negative", negative_modifiers[0])
        if event_id is not None:
            builder.edge(event_id, affect_id, "has_modifier")

    if greeting:
        add("act.greet", "speech_content", "greeting", next((word for word in words if word in GREETING_WORDS), "hello"))
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
    user_id = (
        add("entity.user", "participant", "user")
        if has_first_person or speech_act.startswith("social.") or speech_act in {
            "request.presence_check", "response.presence_confirmation", "request.action",
            "request.avoid_topic", "request.identity", "request.user_identity", "request.capability",
        }
        else None
    )
    assistant_id = (
        add("entity.assistant", "participant", "assistant")
        if has_second_person or mentions_assistant or speech_act.startswith("social.") or speech_act in {
            "request.presence_check", "response.presence_confirmation", "request.identity",
            "request.user_identity", "request.capability",
        }
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
        elif speech_act in {"request.identity", "request.user_identity", "request.capability"}:
            if user_id is not None:
                builder.edge(event_id, user_id, "asker")
            if assistant_id is not None:
                builder.edge(event_id, assistant_id, "target")
            if speech_act == "request.user_identity" and user_id is not None:
                builder.edge(event_id, user_id, "identity_subject")
        elif verb_position is not None:
            before_verb, after_verb = words[:verb_position], words[verb_position + 1:]
            if user_id is not None and any(word in FIRST_PERSON for word in before_verb):
                builder.edge(event_id, user_id, "agent")
            if assistant_id is not None and any(word in SECOND_PERSON or word in ASSISTANT_ALIASES for word in before_verb):
                builder.edge(event_id, assistant_id, "agent")
            if user_id is not None and any(word in FIRST_PERSON for word in after_verb):
                builder.edge(event_id, user_id, "patient")
            if assistant_id is not None and any(word in SECOND_PERSON or word in ASSISTANT_ALIASES for word in after_verb):
                builder.edge(event_id, assistant_id, "patient")
            if speech_act in {"request.action", "request.avoid_topic"} and user_id is not None and not before_verb:
                builder.edge(event_id, user_id, "agent")

    if topic_text:
        topic_value = " ".join(topic_text.split())
        concept = "topic.resolved" if topic_is_resolved else "topic.target"
        add(concept, "topic", topic_value, topic_value, 0.85 if topic_is_resolved else 0.75)

    # Content terms are retained for diagnostics and document retrieval, but
    # only topic.target/topic.resolved is allowed to update discourse topic.
    lexical_tokens = tuple(
        word for word in words
        if word not in STOPWORDS and word not in ASSISTANT_ALIASES and len(word) > 1
    )
    return builder.build(original, normalized, words, lexical_tokens, intent)
