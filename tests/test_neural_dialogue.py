from __future__ import annotations

import json
import unittest

from mcrl.neural_dialogue import (
    FRAME_SCHEMA,
    PLAN_SCHEMA,
    NeuralDialogueCore,
    _compact_history,
    _validate_interpretation,
    LocalModelError,
)


def frame(**overrides):
    value = {
        "interpretation": "A context-dependent confirmation request.",
        "speech_act": "confirmation_request",
        "intent": "continue_conversation",
        "topic": None,
        "references": ["previous_assistant_turn"],
        "clauses": [],
        "requires_knowledge": False,
        "knowledge_target": None,
        "needs_clarification": False,
        "clarification_question": "",
        "response_goal": "Confirm the preceding statement.",
        "constraints": ["Use only what the previous answer supports."],
        "ambiguity": "low",
    }
    value.update(overrides)
    return value


class FakeClient:
    model = "fake-local-model"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def complete_json(self, messages, schema_name, schema, max_tokens=700):
        self.calls.append({
            "kind": "json",
            "messages": list(messages),
            "schema_name": schema_name,
            "schema": schema,
        })
        return self.outputs.pop(0)

    def complete_text(self, messages, max_tokens=320, temperature=0.45):
        self.calls.append({
            "kind": "text",
            "messages": list(messages),
        })
        return self.outputs.pop(0)


class NeuralDialogueTests(unittest.TestCase):
    def test_validates_typed_interpretation_before_realization(self):
        parsed = _validate_interpretation(frame())
        self.assertEqual(parsed["speech_act"], "confirmation_request")
        with self.assertRaises(LocalModelError):
            _validate_interpretation(frame(speech_act="random_keyword_guess"))

    def test_context_is_supplied_to_understanding_and_realization(self):
        client = FakeClient([
            frame(),
            "Yes. I meant that the recap contains the messages recorded in this session.",
        ])
        core = NeuralDialogueCore(client)
        history = [{
            "user": "What we talked till now?",
            "assistant": "Here is the recap of this session.",
        }]
        answer = core.respond("really?", history=history)
        self.assertIn("Yes.", answer.text)
        self.assertEqual(answer.interpretation["references"], ["previous_assistant_turn"])
        first_user_payload = client.calls[0]["messages"][1]["content"]
        final_user_payload = client.calls[1]["messages"][1]["content"]
        self.assertIn("What we talked till now?", first_user_payload)
        self.assertIn("Here is the recap", final_user_payload)

    def test_deep_reasoning_inserts_a_separate_plan_pass(self):
        plan = {
            "decision": "reply_from_context",
            "steps": ["Resolve the reaction against the previous assistant reply."],
            "evidence_to_use": ["The latest assistant turn."],
            "uncertainties": [],
            "response_requirements": ["Confirm only what is recorded."],
        }
        client = FakeClient([frame(), plan, "Yes, that is what I meant."])
        answer = NeuralDialogueCore(client).respond("really?", deep_reasoning=True)
        self.assertTrue(answer.deliberated)
        self.assertEqual(answer.plan["decision"], "reply_from_context")
        self.assertEqual([call["kind"] for call in client.calls], ["json", "json", "text"])
        self.assertEqual(client.calls[1]["schema"], PLAN_SCHEMA)
        self.assertEqual(client.calls[0]["schema"], FRAME_SCHEMA)

    def test_standard_mode_does_not_run_deliberation_pass(self):
        client = FakeClient([frame(), "Sure—what would you like to learn?"])
        answer = NeuralDialogueCore(client).respond("Can you teach me?")
        self.assertFalse(answer.deliberated)
        self.assertIsNone(answer.plan)
        self.assertEqual(len(client.calls), 2)

    def test_history_is_bounded_and_preserves_recent_turn_text(self):
        history = [
            {"user": f"u{i}", "assistant": f"a{i}"}
            for i in range(12)
        ]
        compact = _compact_history(history, limit=8)
        self.assertEqual(len(compact), 8)
        self.assertEqual(compact[0]["user"], "u4")
        self.assertEqual(compact[-1]["assistant"], "a11")

    def test_interpreter_never_answers_instead_of_returning_frame(self):
        client = FakeClient([frame(), "A natural response."])
        core = NeuralDialogueCore(client)
        result = core.respond("hello")
        self.assertEqual(result.interpretation["speech_act"], "confirmation_request")
        self.assertEqual(result.text, "A natural response.")
        self.assertEqual(client.calls[0]["schema_name"], "mcrl_semantic_frame")


if __name__ == "__main__":
    unittest.main()
