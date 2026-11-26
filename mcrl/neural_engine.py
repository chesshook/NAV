"""MCR-L engine variant powered by a local learned language model."""
from __future__ import annotations

from .calculator import try_calculate
from .engine import Answer, MCRLanguageEngine
from .knowledge import KnowledgeStore
from .neural_dialogue import (
    LocalModelError,
    NeuralDialogueCore,
    OpenAICompatibleClient,
)
from .situation_model import TurnRecord
from .semantic_network import LexicalSemanticNetwork, SemanticAnalysis


class NeuralMCRLanguageEngine(MCRLanguageEngine):
    """Use a pretrained local model for language; retain deterministic safe arithmetic.

    The legacy symbolic engine remains available as an explicit fallback mode. When
    neural mode is selected, a model failure is surfaced instead of silently routing
    ordinary language into the bounded keyword parser.
    """

    def __init__(
        self,
        store: KnowledgeStore,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "gguf/Qwen3.5-2B-GGUF:Q4_K_M",
        api_key: str = "none",
        timeout: float = 180.0,
        deep_reasoning: bool = False,
        core: NeuralDialogueCore | None = None,
        semantic_network: LexicalSemanticNetwork | None = None,
    ) -> None:
        super().__init__(store)
        client = OpenAICompatibleClient(
            base_url=base_url, model=model, api_key=api_key, timeout=timeout
        )
        self.neural_core = core or NeuralDialogueCore(client)
        self.deep_reasoning = bool(deep_reasoning)
        self.semantic_network = semantic_network
        self.neural_history: list[dict[str, str]] = []

    def _remember(self, user_text: str, assistant_text: str) -> None:
        self.neural_history.append({
            "user": " ".join(user_text.split())[:2400],
            "assistant": " ".join(assistant_text.split())[:2400],
        })
        self.neural_history = self.neural_history[-12:]

    def _record_neural_turn(self, user_text: str, assistant_text: str, frame: dict) -> None:
        speech_act = str(frame.get("speech_act") or "other")
        intent = str(frame.get("intent") or "continue_conversation")
        topic = frame.get("topic")
        if isinstance(topic, str) and topic.strip():
            self.situation.current_topic = topic.strip()
        elif speech_act in {
            "assistant_identity_question", "user_identity_question",
            "conversation_history_request", "request_teaching",
        }:
            self.situation.current_topic = ""

        self.situation.turn_count += 1
        self.situation.active_goal = intent
        self.situation.last_user_moves = (speech_act,)
        self.situation.last_speaker = "user"
        self.situation.last_addressee = "assistant"
        self.situation.last_assistant_action = "neural_response_realization"
        self.situation.last_outcome = (
            "clarification_requested"
            if frame.get("needs_clarification") else "answered"
        )
        self.situation.pending_question = False
        self.situation.recent_turns.append(TurnRecord(
            index=self.situation.turn_count,
            user_moves=(speech_act,),
            user_goal=intent,
            topic=self.situation.current_topic,
            speaker="user",
            addressee="assistant",
            assistant_action=self.situation.last_assistant_action,
            outcome=self.situation.last_outcome,
            user_text=" ".join(user_text.split()),
            assistant_text=" ".join(assistant_text.split()),
        ))
        self.situation.recent_turns = self.situation.recent_turns[-self.situation.history_limit:]
        self.discourse.turn_count += 1
        self.discourse.last_speech_act = speech_act
        if isinstance(topic, str) and topic.strip():
            self.discourse.remember_topic(topic)

    def answer(self, utterance: str) -> Answer:
        # Encode every user utterance through the lexical-semantic network first.
        # Unknown words remain explicit subword nodes; they are not discarded.
        lexical_analysis: SemanticAnalysis | None = None
        lexical_context: dict | None = None
        if self.semantic_network is not None:
            previous_context = " ".join(
                f"{turn.get('user', '')} {turn.get('assistant', '')}"
                for turn in self.neural_history[-8:]
            )
            lexical_analysis = self.semantic_network.analyze(
                utterance, dialogue_context=previous_context
            )
            lexical_context = lexical_analysis.prompt_view()

        # Exact arithmetic remains a deterministic capability after lexical encoding.
        if try_calculate(utterance) is not None:
            result = super().answer(utterance)
            if lexical_analysis is not None:
                result.diagnostics["semantic_network"] = lexical_analysis.diagnostics()
            self._remember(utterance, result.text)
            return result

        try:
            request = {
                "history": self.neural_history,
                "deep_reasoning": self.deep_reasoning,
            }
            if lexical_context is not None:
                request["lexical_context"] = lexical_context
            turn = self.neural_core.respond(utterance, **request)
        except (LocalModelError, ValueError) as exc:
            return Answer(
                text=(
                    "My local language model did not produce a usable interpretation, "
                    "so I am not going to guess or silently fall back to the older "
                    f"keyword parser. Model error: {exc}"
                ),
                confidence=0.0,
                abstained=True,
                selected_title="Local neural language backend unavailable or invalid",
                source="MCR-L local model adapter",
                diagnostics={
                    "mode": "neural_backend_error",
                    "error": str(exc),
                    "semantic_network": (
                        lexical_analysis.diagnostics() if lexical_analysis is not None else None
                    ),
                    "model": getattr(
                        getattr(self.neural_core, "client", None), "model", "injected-core"
                    ),
                },
            )

        self._record_neural_turn(utterance, turn.text, turn.interpretation)
        self._remember(utterance, turn.text)
        return Answer(
            text=turn.text,
            confidence=0.0,
            abstained=False,
            selected_title="Contextual learned-language interpretation and realization",
            source=(
                "MCR-L lexical-semantic network plus local pretrained model"
                if lexical_analysis is not None
                else "Local pretrained model via OpenAI-compatible inference API"
            ),
            diagnostics={
                "mode": "neural_dialogue",
                "model": turn.model,
                "deliberated": turn.deliberated,
                "interpretation": turn.interpretation,
                "response_plan": turn.plan,
                "semantic_network": (
                    lexical_analysis.diagnostics() if lexical_analysis is not None else None
                ),
                "confidence_semantics": (
                    "No calibrated probability is claimed; inspect interpretation, "
                    "ambiguity, and uncertainties instead."
                ),
                "situation": self.situation.snapshot(),
            },
        )
