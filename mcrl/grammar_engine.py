"""Grammar operators that linearize a response meaning into a sentence AST."""
from __future__ import annotations

from dataclasses import dataclass

from .discourse_state import DiscourseState
from .lexical_compiler import LexicalCompiler
from .meaning_graph import SemanticGraph
from .meaning_solver import ResponseMeaning


@dataclass(frozen=True)
class SentenceAST:
    construction: str
    modifier: str = ""
    verb: str = ""
    object_pronoun: str = ""
    reciprocal_marker: str = ""
    fixed_lexemes: tuple[str, ...] = ()

    def linearize(self) -> str:
        if self.construction == "adjective_infinitive_reciprocal":
            words = [self.modifier, "to", self.verb, self.object_pronoun, self.reciprocal_marker]
            return " ".join(word for word in words if word).strip() + "!"
        if self.construction == "single_word_greeting":
            return (self.fixed_lexemes[0] if self.fixed_lexemes else "Hello") + "!"
        if self.construction == "positive_acknowledgement":
            return "Glad it helps! What should we tackle next?"
        if self.construction == "empathetic_acknowledgement":
            return "Ah, that sounds rough. Want to tell me what happened?"
        if self.construction == "topic_avoidance_acknowledgement":
            topic = self.fixed_lexemes[0] if self.fixed_lexemes else "that topic"
            return f"Got it — I won’t bring up {topic} unless you ask."
        if self.construction == "presence_confirmation":
            return "Yes, I am here!"
        if self.construction == "gratitude_acknowledgement":
            return "You're welcome!"
        if self.construction == "farewell":
            return (self.fixed_lexemes[0] if self.fixed_lexemes else "Goodbye") + "!"
        if self.construction == "wellbeing_answer":
            return "I'm running and ready to help. I don't have feelings, but thanks for asking."
        if self.construction == "clarification_prompt":
            return "What would you like me to clarify?"
        if self.construction == "identity_answer":
            return (
                "I'm MCR-L, a CPU-first semantic-composition prototype. "
                "I use an explicit, bounded grammar and structured meaning rules; "
                "I am not a general-purpose foundation model."
            )
        if self.construction == "generic_acknowledgement":
            return "It's good to meet you too!"
        return ""


class GrammarEngine:
    """Builds a syntax tree from semantic roles, then linearizes it."""

    def __init__(self, lexical: LexicalCompiler | None = None) -> None:
        self.lexical = lexical or LexicalCompiler()

    def build_ast(self, response: ResponseMeaning, source: SemanticGraph) -> SentenceAST:
        if response.act == "reciprocate_meeting":
            return SentenceAST(
                construction="adjective_infinitive_reciprocal",
                modifier=self.lexical.choose_meeting_modifier(source),
                verb=response.predicate or "meet",
                object_pronoun=response.recipient or "you",
                reciprocal_marker="too" if response.reciprocal else "",
            )
        if response.act == "generic_acknowledgement":
            return SentenceAST(construction="generic_acknowledgement")
        if response.act == "reciprocate_greeting":
            return SentenceAST(
                construction="single_word_greeting",
                fixed_lexemes=(self.lexical.choose_greeting(source),),
            )
        if response.act == "acknowledge_positive_statement":
            return SentenceAST(construction="positive_acknowledgement")
        if response.act == "acknowledge_negative_statement":
            return SentenceAST(construction="empathetic_acknowledgement")
        if response.act == "acknowledge_topic_avoidance":
            return SentenceAST(
                construction="topic_avoidance_acknowledgement",
                fixed_lexemes=(str(response.recipient).strip().title(),),
            )
        if response.act == "confirm_presence":
            return SentenceAST(construction="presence_confirmation")
        if response.act == "acknowledge_gratitude":
            return SentenceAST(construction="gratitude_acknowledgement")
        if response.act == "reciprocate_farewell":
            return SentenceAST(
                construction="farewell",
                fixed_lexemes=(self.lexical.choose_farewell(source),),
            )
        if response.act == "answer_wellbeing":
            return SentenceAST(construction="wellbeing_answer")
        if response.act == "ask_clarification":
            return SentenceAST(construction="clarification_prompt")
        if response.act == "identify_self":
            return SentenceAST(construction="identity_answer")
        return SentenceAST(construction="unsupported")

    def realize(self, response: ResponseMeaning, source: SemanticGraph) -> tuple[str, SentenceAST]:
        ast = self.build_ast(response, source)
        return ast.linearize(), ast
