from __future__ import annotations

import unittest

from mcrl.engine import MCRLanguageEngine
from mcrl.knowledge import KnowledgeStore
from mcrl.neural_dialogue import LocalModelError, NeuralTurn
from mcrl.neural_engine import NeuralMCRLanguageEngine


class FakeCore:
    def __init__(self, outputs=None, error=None):
        self.outputs = list(outputs or [])
        self.error = error
        self.calls = []

    def respond(self, utterance, history=(), deep_reasoning=False):
        self.calls.append({
            "utterance": utterance,
            "history": list(history),
            "deep_reasoning": deep_reasoning,
        })
        if self.error:
            raise self.error
        return self.outputs.pop(0)


def neural_turn(text, speech_act="other", topic=None):
    frame = {
        "interpretation": "A contextual conversational interpretation.",
        "speech_act": speech_act,
        "intent": "continue_conversation",
        "topic": topic,
        "references": [],
        "clauses": [],
        "requires_knowledge": False,
        "knowledge_target": None,
        "needs_clarification": False,
        "clarification_question": "",
        "response_goal": "Continue naturally.",
        "constraints": [],
        "ambiguity": "low",
    }
    return NeuralTurn(text, frame, None, "fake-model", False)


class NeuralEngineTests(unittest.TestCase):
    def setUp(self):
        self.store = KnowledgeStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_neural_engine_uses_model_and_carries_actual_context_forward(self):
        core = FakeCore([
            neural_turn("Sure—what would you like to learn?", "request_teaching"),
            neural_turn("Yes. I was referring to the session recap.", "confirmation_request"),
        ])
        engine = NeuralMCRLanguageEngine(self.store, core=core)

        first = engine.answer("Can you teach me?")
        second = engine.answer("Really?")

        self.assertEqual(first.diagnostics["mode"], "neural_dialogue")
        self.assertEqual(second.diagnostics["mode"], "neural_dialogue")
        self.assertEqual(core.calls[1]["history"][0]["user"], "Can you teach me?")
        self.assertEqual(
            core.calls[1]["history"][0]["assistant"],
            "Sure—what would you like to learn?",
        )
        self.assertEqual(engine.situation.recent_turns[-1].user_text, "Really?")

    def test_neural_backend_failure_does_not_fall_back_to_symbolic_keyword_parser(self):
        core = FakeCore(error=LocalModelError("server offline"))
        engine = NeuralMCRLanguageEngine(self.store, core=core)
        result = engine.answer("really?")

        self.assertEqual(result.diagnostics["mode"], "neural_backend_error")
        self.assertTrue(result.abstained)
        self.assertIn("not going to guess", result.text.lower())
        self.assertNotIn("Really", result.text)

    def test_exact_arithmetic_remains_on_deterministic_path(self):
        core = FakeCore([])
        engine = NeuralMCRLanguageEngine(self.store, core=core)
        result = engine.answer("What is 12 * 7?")

        self.assertEqual(result.diagnostics["mode"], "calculation")
        self.assertEqual(result.text, "12 * 7 = 84.")
        self.assertEqual(core.calls, [])


if __name__ == "__main__":
    unittest.main()
