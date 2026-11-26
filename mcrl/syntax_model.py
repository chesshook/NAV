"""CPU-trained lexical/syntactic parser for mathematical comprehension.

Averaged perceptrons learn part-of-speech decisions and dependency transitions from
Universal Dependencies annotations. At run time, they produce a dependency tree. No
LLM, embedding service, transformer, or autoregressive decoder is called.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .averaged_perceptron import AveragedPerceptron
from .conllu import ConlluToken

MODEL_VERSION = 1
TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*|[^\w\s]", re.UNICODE)
CONTRACTION_SPLITS = {
    "don't": ("do", "n't"), "dont": ("do", "n't"),
    "doesn't": ("does", "n't"), "doesnt": ("does", "n't"),
    "didn't": ("did", "n't"), "didnt": ("did", "n't"),
    "isn't": ("is", "n't"), "isnt": ("is", "n't"),
    "aren't": ("are", "n't"), "arent": ("are", "n't"),
    "wasn't": ("was", "n't"), "wasnt": ("was", "n't"),
    "weren't": ("were", "n't"), "werent": ("were", "n't"),
    "can't": ("ca", "n't"), "cant": ("ca", "n't"),
    "couldn't": ("could", "n't"), "couldnt": ("could", "n't"),
    "wouldn't": ("would", "n't"), "wouldnt": ("would", "n't"),
    "shouldn't": ("should", "n't"), "shouldnt": ("should", "n't"),
    "won't": ("wo", "n't"), "wont": ("wo", "n't"),
    "shan't": ("sha", "n't"), "shant": ("sha", "n't"),
    "i'm": ("I", "'m"), "im": ("I", "'m"),
    "you're": ("you", "'re"), "youre": ("you", "'re"),
    "we're": ("we", "'re"), "were": ("we", "'re"),
    "they're": ("they", "'re"), "theyre": ("they", "'re"),
    "it's": ("it", "'s"), "its": ("it", "'s"),
    "he's": ("he", "'s"), "she's": ("she", "'s"),
    "that's": ("that", "'s"), "thats": ("that", "'s"),
    "what's": ("what", "'s"), "whats": ("what", "'s"),
    "who's": ("who", "'s"), "whos": ("who", "'s"),
    "there's": ("there", "'s"), "theres": ("there", "'s"),
    "I've": ("I", "'ve"), "i've": ("I", "'ve"),
    "we've": ("we", "'ve"), "they've": ("they", "'ve"),
    "I'll": ("I", "'ll"), "i'll": ("I", "'ll"),
    "you'll": ("you", "'ll"), "we'll": ("we", "'ll"),
}
IRREGULAR_LEMMAS = {
    "am": "be", "is": "be", "are": "be", "was": "be", "were": "be",
    "been": "be", "being": "be", "did": "do", "does": "do", "done": "do",
    "has": "have", "had": "have", "went": "go", "gone": "go",
    "taught": "teach", "thought": "think", "made": "make",
    "saw": "see", "seen": "see", "bought": "buy", "brought": "bring",
    "children": "child", "men": "man", "women": "woman",
}


@dataclass
class ParseToken:
    id: int
    form: str
    lemma: str
    upos: str
    feats: str = "_"
    head: int = 0
    deprel: str = "dep"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParseResult:
    text: str
    tokens: tuple[ParseToken, ...]
    pos_accuracy_claim: str = "not calibrated; measure on held-out UD and interview data"

    def dependencies(self) -> list[dict[str, Any]]:
        return [
            {
                "head": token.head,
                "head_form": "ROOT" if token.head == 0 else self.tokens[token.head - 1].form,
                "dependent": token.id,
                "dependent_form": token.form,
                "relation": token.deprel,
            }
            for token in self.tokens
        ]


def tokenize(text: str) -> list[str]:
    raw = TOKEN_RE.findall(text or "")
    output: list[str] = []
    for item in raw:
        forms = CONTRACTION_SPLITS.get(item.lower())
        if forms:
            output.extend(forms)
        else:
            output.append(item)
    return output


def _shape(word: str) -> str:
    chars = []
    for char in word:
        if char.isupper():
            marker = "X"
        elif char.islower():
            marker = "x"
        elif char.isdigit():
            marker = "d"
        else:
            marker = char
        if not chars or chars[-1] != marker:
            chars.append(marker)
    return "".join(chars)[:12]


def _affixes(word: str) -> list[str]:
    low = word.lower()
    features = []
    for size in (1, 2, 3, 4):
        if len(low) >= size:
            features.append(f"PRE{size}={low[:size]}")
            features.append(f"SUF{size}={low[-size:]}")
    return features


def pos_features(
    forms: Sequence[str],
    index: int,
    previous_tags: Sequence[str],
    dictionary_pos: Sequence[set[str]] | None = None,
) -> list[str]:
    word = forms[index]
    low = word.lower()
    prev = forms[index - 1].lower() if index > 0 else "<BOS>"
    prev2 = forms[index - 2].lower() if index > 1 else "<BOS2>"
    next_word = forms[index + 1].lower() if index + 1 < len(forms) else "<EOS>"
    next2 = forms[index + 2].lower() if index + 2 < len(forms) else "<EOS2>"
    prev_tag = previous_tags[-1] if previous_tags else "<BOS>"
    prev2_tag = previous_tags[-2] if len(previous_tags) > 1 else "<BOS2>"
    features = [
        "BIAS",
        f"W={low}",
        f"SHAPE={_shape(word)}",
        f"PREV={prev}",
        f"NEXT={next_word}",
        f"PREV2={prev2}",
        f"NEXT2={next2}",
        f"W_PREV={low}|{prev}",
        f"W_NEXT={low}|{next_word}",
        f"PREV_NEXT={prev}|{next_word}",
        f"PTAG={prev_tag}",
        f"P2TAG={prev2_tag}",
        f"PTAG_W={prev_tag}|{low}",
        f"PTAG_NEXT={prev_tag}|{next_word}",
        f"IS_CAP={int(word[:1].isupper())}",
        f"IS_ALLCAP={int(word.isupper() and any(ch.isalpha() for ch in word))}",
        f"IS_DIGIT={int(word.isdigit())}",
        f"HAS_HYPHEN={int('-' in word)}",
    ]
    features.extend(_affixes(word))
    if dictionary_pos is not None and index < len(dictionary_pos):
        for candidate in sorted(dictionary_pos[index]):
            features.append(f"DICTPOS={candidate}")
    return features


def parser_features(
    stack: Sequence[int],
    buffer: Sequence[int],
    tokens: Sequence[ParseToken],
    heads: dict[int, tuple[int, str]],
) -> list[str]:
    root = ParseToken(0, "<ROOT>", "<ROOT>", "ROOT")
    def item(position: int) -> ParseToken:
        return root if position == 0 else tokens[position - 1]

    s0 = stack[-1] if stack else 0
    s1 = stack[-2] if len(stack) > 1 else 0
    s2 = stack[-3] if len(stack) > 2 else 0
    b0 = buffer[0] if buffer else 0
    b1 = buffer[1] if len(buffer) > 1 else 0
    b2 = buffer[2] if len(buffer) > 2 else 0
    selected = {
        "S0": item(s0), "S1": item(s1), "S2": item(s2),
        "B0": item(b0), "B1": item(b1), "B2": item(b2),
    }
    features = ["BIAS"]
    for name, token in selected.items():
        features.extend((
            f"{name}W={token.form.lower()}",
            f"{name}L={token.lemma.lower()}",
            f"{name}P={token.upos}",
            f"{name}F={token.feats}",
            f"{name}WP={token.form.lower()}|{token.upos}",
        ))
    for left, right in (
        ("S0", "S1"), ("S0", "B0"), ("S0", "B1"),
        ("S1", "B0"), ("B0", "B1"), ("B0", "B2"),
        ("S1", "S2"),
    ):
        a, b = selected[left], selected[right]
        features.extend((
            f"{left}_{right}_P={a.upos}|{b.upos}",
            f"{left}_{right}_W={a.form.lower()}|{b.form.lower()}",
            f"{left}_{right}_L={a.lemma.lower()}|{b.lemma.lower()}",
        ))
    features.append(f"DIST_S0_B0={min(10, abs(s0-b0))}")
    for position, name in ((s0, "S0"), (s1, "S1")):
        dependents = [
            dep for dep, (head, _rel) in heads.items()
            if head == position
        ]
        features.append(f"{name}_DEPS={min(6, len(dependents))}")
        if dependents:
            features.append(
                f"{name}_LEFT_DEPS={min(4, sum(dep < position for dep in dependents))}"
            )
            features.append(
                f"{name}_RIGHT_DEPS={min(4, sum(dep > position for dep in dependents))}"
            )
    return features


class MathematicalSyntaxModel:
    """Greedy POS tagging plus arc-standard dependency parsing."""

    def __init__(self) -> None:
        self.pos = AveragedPerceptron()
        self.dependencies = AveragedPerceptron()
        self.pos_labels: set[str] = set()
        self.dependency_labels: set[str] = set()
        self.metadata: dict[str, Any] = {}

    @classmethod
    def load(cls, path: str | Path) -> "MathematicalSyntaxModel":
        with open(path, "r", encoding="utf-8") as source:
            payload = json.load(source)
        if payload.get("model_version") != MODEL_VERSION:
            raise ValueError("Unsupported mathematical syntax model version.")
        model = cls()
        model.pos.set_weights(payload["pos_weights"])
        model.dependencies.set_weights(payload["dependency_weights"])
        model.pos_labels = set(payload["pos_labels"])
        model.dependency_labels = set(payload["dependency_labels"])
        model.metadata = dict(payload.get("metadata") or {})
        if not model.pos_labels or not model.dependency_labels:
            raise ValueError("Syntax model is missing POS/dependency labels.")
        return model

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_version": MODEL_VERSION,
            "pos_labels": sorted(self.pos_labels),
            "dependency_labels": sorted(self.dependency_labels),
            "pos_weights": self.pos.averaged_weights(),
            "dependency_weights": self.dependencies.averaged_weights(),
            "metadata": self.metadata,
            "architecture": "sparse averaged perceptron POS tagger + arc-standard transition parser",
            "notices": [
                "No transformer or autoregressive language model is used by this syntax engine.",
                "Scores are not calibrated probabilities.",
            ],
        }
        with open(target, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, separators=(",", ":"))

    def tag(
        self,
        forms: Sequence[str],
        dictionary_pos: Sequence[set[str]] | None = None,
    ) -> list[str]:
        tags: list[str] = []
        for index in range(len(forms)):
            features = pos_features(forms, index, tags, dictionary_pos)
            tags.append(self.pos.predict(features, self.pos_labels))
        return tags

    def parse_forms(
        self,
        forms: Sequence[str],
        dictionary_pos: Sequence[set[str]] | None = None,
        lemmas: Sequence[str] | None = None,
        feats: Sequence[str] | None = None,
    ) -> ParseResult:
        if not forms:
            return ParseResult(text="", tokens=())
        tags = self.tag(forms, dictionary_pos)
        tokens = [
            ParseToken(
                id=i + 1,
                form=form,
                lemma=(
                    lemmas[i] if lemmas and i < len(lemmas) and lemmas[i] not in {"", "_"}
                    else IRREGULAR_LEMMAS.get(form.lower(), form.lower())
                ),
                upos=tags[i],
                feats=feats[i] if feats and i < len(feats) else "_",
            )
            for i, form in enumerate(forms)
        ]
        stack = [0]
        buffer = list(range(1, len(tokens) + 1))
        heads: dict[int, tuple[int, str]] = {}
        max_steps = len(tokens) * 3 + 8
        steps = 0
        while (buffer or len(stack) > 1) and steps < max_steps:
            steps += 1
            features = parser_features(stack, buffer, tokens, heads)
            allowed: list[str] = []
            if buffer:
                allowed.append("SHIFT")
            if len(stack) >= 2:
                if stack[-2] != 0:
                    allowed.extend(f"LEFT|{label}" for label in self.dependency_labels)
                if stack[-1] != 0:
                    allowed.extend(f"RIGHT|{label}" for label in self.dependency_labels)
            if not allowed:
                break
            action = self.dependencies.predict(features, allowed)
            if action == "SHIFT":
                stack.append(buffer.pop(0))
            elif action.startswith("LEFT|") and len(stack) >= 2 and stack[-2] != 0:
                relation = action.split("|", 1)[1]
                dependent = stack.pop(-2)
                heads[dependent] = (stack[-1], relation)
            elif action.startswith("RIGHT|") and len(stack) >= 2:
                relation = action.split("|", 1)[1]
                dependent = stack.pop()
                heads[dependent] = (stack[-1], relation)
            else:
                # Invalid model output is prevented by the allowed-action filter.
                break

        # A valid tree assigns one head to every token. If an untrained model or
        # unexpected sentence shape prevents completion, expose missing arcs rather
        # than fabricating a root or silently dropping words.
        for token in tokens:
            if token.id in heads:
                token.head, token.deprel = heads[token.id]
            else:
                token.head, token.deprel = 0, "unparsed"
        return ParseResult(text=" ".join(forms), tokens=tuple(tokens))

    def parse_text(self, text: str, lexical_network=None, dialogue_context: str = "") -> ParseResult:
        forms = tokenize(text)
        dictionary_pos: list[set[str]] = []
        lemmas: list[str] = []
        if lexical_network is not None:
            for form in forms:
                rows = lexical_network.lookup(form, limit=24)
                pos_values = {
                    _wordnet_pos_to_ud(str(row.get("part_of_speech") or ""))
                    for row in rows
                }
                pos_values.discard("")
                dictionary_pos.append(pos_values)
                matching = [str(row.get("lemma") or "") for row in rows]
                lemmas.append(matching[0] if matching and matching[0] else IRREGULAR_LEMMAS.get(form.lower(), form.lower()))
        else:
            dictionary_pos = [set() for _ in forms]
            lemmas = [IRREGULAR_LEMMAS.get(form.lower(), form.lower()) for form in forms]
        return self.parse_forms(forms, dictionary_pos, lemmas)


def _wordnet_pos_to_ud(value: str) -> str:
    return {
        "n": "NOUN", "v": "VERB", "a": "ADJ", "s": "ADJ", "r": "ADV",
        "p": "PRON",
    }.get(value.lower(), "")


def _gold_children(sentence: Sequence[ConlluToken]) -> dict[int, set[int]]:
    children: dict[int, set[int]] = defaultdict(set)
    for token in sentence:
        children[token.head].add(token.id)
    return children


def _oracle_actions(sentence: Sequence[ConlluToken]):
    """Yield gold arc-standard states/actions; return None for non-projective trees."""
    stack = [0]
    buffer = list(range(1, len(sentence) + 1))
    gold_heads = {token.id: token.head for token in sentence}
    gold_rels = {token.id: token.deprel for token in sentence}
    children = _gold_children(sentence)
    attached: set[int] = set()
    tokens = [
        ParseToken(item.id, item.form, item.lemma, item.upos, item.feats, item.head, item.deprel)
        for item in sentence
    ]
    heads: dict[int, tuple[int, str]] = {}
    steps = 0
    max_steps = len(sentence) * 3 + 8
    while (buffer or len(stack) > 1) and steps < max_steps:
        steps += 1
        action = None
        if len(stack) >= 2:
            s1, s0 = stack[-2], stack[-1]
            if (
                s1 != 0
                and gold_heads[s1] == s0
                and children.get(s1, set()).issubset(attached)
            ):
                action = f"LEFT|{gold_rels[s1]}"
                stack.pop(-2)
                heads[s1] = (s0, gold_rels[s1])
                attached.add(s1)
            elif (
                s0 != 0
                and gold_heads[s0] == s1
                and children.get(s0, set()).issubset(attached)
            ):
                action = f"RIGHT|{gold_rels[s0]}"
                stack.pop()
                heads[s0] = (s1, gold_rels[s0])
                attached.add(s0)
        if action is None:
            if buffer:
                action = "SHIFT"
                stack.append(buffer.pop(0))
            else:
                return None
        features = parser_features(stack, buffer, tokens, heads)
        # Features describe the state before the action. Reconstruct the previous
        # state by reversing the simple stack/buffer transition.
        if action == "SHIFT":
            buffer.insert(0, stack.pop())
        elif action.startswith("LEFT|"):
            dependent = next(
                token_id for token_id, (head, _rel) in heads.items()
                if token_id not in attached - {token_id} and False
            ) if False else None
            # The left arc dependent was the former second-to-top stack item.
            # Rebuild state from gold prefix is less error-prone in the caller.
            return None
        else:
            return None
        yield stack, buffer, heads, action
    if len(attached) != len(sentence):
        return None


def training_examples(sentence: Sequence[ConlluToken]):
    """Materialize gold parser configurations for one projective dependency tree."""
    stack = [0]
    buffer = list(range(1, len(sentence) + 1))
    gold_heads = {token.id: token.head for token in sentence}
    gold_rels = {token.id: token.deprel for token in sentence}
    children = _gold_children(sentence)
    attached: set[int] = set()
    tokens = [
        ParseToken(item.id, item.form, item.lemma, item.upos, item.feats, item.head, item.deprel)
        for item in sentence
    ]
    heads: dict[int, tuple[int, str]] = {}
    steps = 0
    max_steps = len(sentence) * 3 + 8
    while (buffer or len(stack) > 1) and steps < max_steps:
        steps += 1
        features = parser_features(stack, buffer, tokens, heads)
        action = None
        if len(stack) >= 2:
            s1, s0 = stack[-2], stack[-1]
            if (
                s1 != 0
                and gold_heads[s1] == s0
                and children.get(s1, set()).issubset(attached)
            ):
                action = f"LEFT|{gold_rels[s1]}"
                stack.pop(-2)
                heads[s1] = (s0, gold_rels[s1])
                attached.add(s1)
            elif (
                s0 != 0
                and gold_heads[s0] == s1
                and children.get(s0, set()).issubset(attached)
            ):
                action = f"RIGHT|{gold_rels[s0]}"
                stack.pop()
                heads[s0] = (s1, gold_rels[s0])
                attached.add(s0)
        if action is None:
            if not buffer:
                return None
            action = "SHIFT"
            stack.append(buffer.pop(0))
        yield features, action
    if len(attached) != len(sentence):
        return None


def train_models(
    sentences: Sequence[Sequence[ConlluToken]],
    epochs: int = 5,
    progress=None,
) -> tuple[MathematicalSyntaxModel, dict[str, int]]:
    """Train a POS tagger and dependency transition model from gold UD trees."""
    if not sentences:
        raise ValueError("Training corpus is empty.")
    model = MathematicalSyntaxModel()
    model.pos_labels = {token.upos for sent in sentences for token in sent}
    model.dependency_labels = {token.deprel for sent in sentences for token in sent} | {"dep"}
    counts = {"sentences": len(sentences), "tokens": sum(len(x) for x in sentences), "nonprojective_skipped": 0}

    for epoch in range(max(1, epochs)):
        mistakes = 0
        for sent_index, sentence in enumerate(sentences):
            forms = [token.form for token in sentence]
            previous: list[str] = []
            for index, token in enumerate(sentence):
                features = pos_features(forms, index, previous)
                guess = model.pos.predict(features, model.pos_labels)
                model.pos.update(token.upos, guess, features)
                mistakes += int(guess != token.upos)
                previous.append(token.upos)
        if progress:
            progress(f"POS epoch {epoch + 1}/{epochs}: token mistakes={mistakes}")

    parser_mistakes = 0
    valid_sentences = 0
    for epoch in range(max(1, epochs)):
        parser_mistakes = 0
        valid_sentences = 0
        for sentence in sentences:
            examples = list(training_examples(sentence))
            if examples is None or not examples:
                counts["nonprojective_skipped"] += 1
                continue
            valid_sentences += 1
            for features, truth in examples:
                # The action space is the complete label set plus SHIFT. The
                # perceptron learns transition scores; inference masks invalid arcs.
                allowed = {"SHIFT"}
                allowed.update(f"LEFT|{label}" for label in model.dependency_labels)
                allowed.update(f"RIGHT|{label}" for label in model.dependency_labels)
                guess = model.dependencies.predict(features, allowed)
                model.dependencies.update(truth, guess, features)
                parser_mistakes += int(guess != truth)
        if progress:
            progress(
                f"Dependency epoch {epoch + 1}/{epochs}: "
                f"transition mistakes={parser_mistakes}; projective sentences={valid_sentences}"
            )
    counts["projective_sentences_per_epoch"] = valid_sentences
    counts["transition_mistakes_last_epoch"] = parser_mistakes
    model.metadata = {
        "training_sentences": len(sentences),
        "training_tokens": counts["tokens"],
        "epochs": max(1, epochs),
        "nonprojective_skipped_per_epoch": counts["nonprojective_skipped"] // max(1, epochs),
    }
    return model, counts


def evaluate_model(model: MathematicalSyntaxModel, sentences: Sequence[Sequence[ConlluToken]]) -> dict[str, float | int]:
    token_count = 0
    pos_correct = 0
    uas_correct = 0
    las_correct = 0
    complete = 0
    for sentence in sentences:
        forms = [token.form for token in sentence]
        parsed = model.parse_forms(forms)
        if len(parsed.tokens) != len(sentence):
            continue
        complete += int(all(token.deprel != "unparsed" for token in parsed.tokens))
        for gold, predicted in zip(sentence, parsed.tokens):
            token_count += 1
            pos_correct += int(gold.upos == predicted.upos)
            uas_correct += int(gold.head == predicted.head)
            las_correct += int(gold.head == predicted.head and gold.deprel == predicted.deprel)
    denominator = max(1, token_count)
    return {
        "sentences": len(sentences),
        "tokens": token_count,
        "pos_accuracy": pos_correct / denominator,
        "uas": uas_correct / denominator,
        "las": las_correct / denominator,
        "complete_tree_fraction": complete / max(1, len(sentences)),
    }
