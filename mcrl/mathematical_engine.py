"""Independent mathematical understanding engine; no language model dependency."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .compositional_semantics import compose_meaning
from .semantic_network import LexicalSemanticNetwork
from .syntax_model import MathematicalSyntaxModel, ParseResult


@dataclass(frozen=True)
class UnderstandingResult:
    original: str
    parse: ParseResult
    meaning: dict[str, Any]
    lexical_diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "meaning": self.meaning,
            "syntax": {
                "tokens": [token.to_dict() for token in self.parse.tokens],
                "dependencies": self.parse.dependencies(),
            },
            "lexical": self.lexical_diagnostics,
        }


class MathematicalComprehensionEngine:
    """Pass each utterance through the dictionary and learned mathematical grammar.

    This engine never calls Qwen, another LLM, a remote API, or an answer-template
    bank. It returns an inspectable interpretation structure. It deliberately does
    not pretend to answer factual questions before a world-knowledge layer exists.
    """

    def __init__(
        self,
        syntax_model_path: str,
        lexicon: LexicalSemanticNetwork,
        history_limit: int = 12,
    ) -> None:
        self.syntax = MathematicalSyntaxModel.load(syntax_model_path)
        self.lexicon = lexicon
        self.history_limit = max(1, int(history_limit))
        self.history: list[dict[str, str]] = []

    def understand(self, utterance: str) -> UnderstandingResult:
        text = (utterance or "").strip()
        if not text:
            raise ValueError("utterance must not be empty")
        dialogue_context = " ".join(
            f"{turn['user']} {turn['meaning']}"
            for turn in self.history[-self.history_limit:]
        )
        lexical = self.lexicon.analyze(text, dialogue_context=dialogue_context)
        parsed = self.syntax.parse_text(
            text, lexical_network=self.lexicon, dialogue_context=dialogue_context
        )
        meaning = compose_meaning(text, parsed, lexical, self.history)
        self.history.append({
            "user": text,
            "meaning": meaning["meaning_summary"],
        })
        self.history = self.history[-self.history_limit:]
        return UnderstandingResult(
            original=text,
            parse=parsed,
            meaning=meaning,
            lexical_diagnostics=lexical.diagnostics(),
        )
