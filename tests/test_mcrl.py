from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from mcrl.cognition import MCRState
from mcrl.engine import MCRLanguageEngine
from mcrl.knowledge import KnowledgeStore
from mcrl.semantic import FEATURE_NAMES, classify_social, parse_semantics


ROOT = Path(__file__).resolve().parents[1]


class MCRLTests(unittest.TestCase):
    def make_store(self, path=":memory:"):
        store = KnowledgeStore(path)
        store.import_jsonl(str(ROOT / "mcrl" / "data" / "seed.jsonl"))
        return store

    def test_semantic_parser_extracts_definition_intent(self):
        frame = parse_semantics("What is MCR-L?")
        self.assertEqual(frame.intent, "define")
        self.assertIn("mcr-l", frame.content_tokens)
        self.assertEqual(len(frame.features), len(FEATURE_NAMES))

    def test_basic_greeting_is_handled_without_retrieval(self):
        store = self.make_store()
        try:
            engine = MCRLanguageEngine(store)
            result = engine.answer("hi")
            self.assertFalse(result.abstained)
            self.assertEqual(result.diagnostics["intent"], "social.greeting")
            self.assertEqual(result.diagnostics["mode"], "dialogue_composition")
            self.assertEqual(result.text, "Hi!")
            self.assertEqual(engine.last_plan, None)
        finally:
            store.close()

    def test_common_social_turns_are_classified(self):
        expected = {
            "Hello!": "greeting",
            "  good   morning  ": "greeting",
            "how are you?": "wellbeing",
            "thank you": "thanks",
            "goodbye": "farewell",
            "who are you?": "identity",
        }
        for text, intent in expected.items():
            with self.subTest(text=text):
                self.assertEqual(classify_social(text), intent)

    def test_recurrent_state_is_bounded_and_changes(self):
        frame = parse_semantics("Why does a computer slow down?")
        state = MCRState()
        first = state.step(frame.features)
        second = state.step(frame.features)
        self.assertEqual(len(first), len(FEATURE_NAMES))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in second))
        self.assertGreater(state.steps, 1)
        self.assertNotEqual(first, tuple(0.0 for _ in first))

    def test_known_question_returns_sourced_answer(self):
        store = self.make_store()
        try:
            engine = MCRLanguageEngine(store)
            result = engine.answer("What is MCR-L?")
            self.assertFalse(result.abstained)
            self.assertIn("CPU-first", result.text)
            self.assertTrue(result.source)
            self.assertGreaterEqual(result.confidence, engine.minimum_score)
        finally:
            store.close()

    def test_unknown_question_abstains(self):
        store = self.make_store()
        try:
            result = MCRLanguageEngine(store).answer(
                "What is the orbital phase of quasar ZY-999?"
            )
            self.assertTrue(result.abstained)
            self.assertIn("know enough", result.text.lower())
        finally:
            store.close()

    def test_jsonl_import_stream_and_persist(self):
        with tempfile.TemporaryDirectory() as temp:
            db = str(Path(temp) / "knowledge.sqlite")
            source = Path(temp) / "extra.jsonl"
            record = {
                "id": "test-001",
                "title": "Test record",
                "text": "A controlled test record for import.",
                "answer": "The import path works.",
                "source": "unit test",
                "keywords": ["controlled", "test"],
                "intents": ["question"],
                "tags": ["evidence"],
            }
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")
            store = KnowledgeStore(db)
            try:
                self.assertEqual(store.import_jsonl(str(source), batch_size=1), 1)
                self.assertEqual(store.count(), 1)
            finally:
                store.close()
            reopened = KnowledgeStore(db)
            try:
                self.assertEqual(reopened.count(), 1)
                self.assertTrue(reopened.search("controlled test"))
            finally:
                reopened.close()

    def test_feedback_keeps_weights_bounded(self):
        state = MCRState()
        state.step(parse_semantics("Why is this happening?").features)
        state.step(parse_semantics("It happened after the update.").features)
        state.observe_feedback(True)
        for row in state.weights:
            for value in row:
                self.assertGreaterEqual(value, -0.48)
                self.assertLessEqual(value, 1.18)


if __name__ == "__main__":
    unittest.main()
