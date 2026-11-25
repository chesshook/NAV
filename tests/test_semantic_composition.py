from __future__ import annotations

import unittest

from mcrl.discourse_state import DiscourseState
from mcrl.engine import MCRLanguageEngine
from mcrl.knowledge import KnowledgeStore
from mcrl.meaning_solver import solve_response
from mcrl.meaning_validator import validate_response
from mcrl.semantic_parser import parse_meaning
from mcrl.sensitivity import calculate_sensitivity


class SemanticCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = KnowledgeStore(":memory:")
        self.engine = MCRLanguageEngine(self.store)

    def tearDown(self) -> None:
        self.store.close()

    def test_nice_to_meet_u_mcr_is_composed_from_meaning(self) -> None:
        result = self.engine.answer("Nice to meet u MCR")
        self.assertEqual(result.text, "Nice to meet you too!")
        self.assertFalse(result.abstained)
        self.assertEqual(result.diagnostics["mode"], "semantic_composition")
        self.assertEqual(result.diagnostics["verification"]["preservation"], 1.0)

    def test_bare_what_is_a_clarification_request_not_a_fact_lookup(self) -> None:
        graph = parse_meaning("what?")
        self.assertEqual(graph.speech_act, "request.clarification")
        self.assertEqual(graph.lexical_tokens, ())
        self.assertFalse(graph.by_kind("polarity"))
        result = self.engine.answer("what?")
        self.assertEqual(result.text, "What would you like me to clarify?")
        self.assertFalse(result.abstained)
        self.assertEqual(result.diagnostics["mode"], "semantic_composition")
        self.assertTrue(result.diagnostics["verification"]["valid"])

    def test_greeting_graph_connects_speaker_and_addressee_to_event(self) -> None:
        graph = parse_meaning("hi")
        self.assertEqual(graph.speech_act, "social.greeting")
        self.assertEqual(graph.first("event"), "greeting")
        self.assertEqual(graph.edge_targets("speaker")[0].value, "user")
        self.assertEqual(graph.edge_targets("addressee")[0].value, "assistant")

    def test_paraphrases_share_the_same_core_event_and_speech_act(self) -> None:
        examples = (
            "Nice to meet you MCR",
            "Pleased to meet you, MCR",
            "Glad to meet u MCR",
        )
        graphs = [parse_meaning(item) for item in examples]
        self.assertTrue(all(graph.speech_act == "social.pleasantry" for graph in graphs))
        self.assertTrue(all(graph.predicate == "meeting" for graph in graphs))
        self.assertTrue(all(graph.edge_targets("addressee") for graph in graphs))

    def test_sensitivity_is_counterfactual_and_normalized(self) -> None:
        graph = parse_meaning("Nice to meet u MCR")
        result = calculate_sensitivity(graph)
        self.assertGreater(result.baseline_utility, 0.0)
        self.assertAlmostEqual(sum(result.node_weights.values()), 1.0, places=6)
        self.assertGreater(result.kind_weights.get("event", 0.0), 0.0)
        self.assertGreater(result.kind_weights.get("participant", 0.0), 0.0)

    def test_response_planner_uses_input_sensitivity(self) -> None:
        graph = parse_meaning("Nice to meet u MCR")
        sensitivity = calculate_sensitivity(graph)
        response = solve_response(graph, sensitivity)
        self.assertIsNotNone(response)
        assert response is not None
        self.assertEqual(response.act, "reciprocate_meeting")
        self.assertTrue(response.reciprocal)
        text = "Nice to meet you too!"
        self.assertTrue(validate_response(response, graph, text).valid)

    def test_negated_meeting_is_not_misread_as_a_pleasantry(self) -> None:
        graph = parse_meaning("It is not nice to meet you")
        self.assertEqual(graph.polarity, "negative")
        self.assertNotEqual(graph.speech_act, "social.pleasantry")

    def test_basic_clause_assigns_agent_and_patient_roles(self) -> None:
        graph = parse_meaning("I like you")
        self.assertEqual(graph.predicate, "like")
        self.assertEqual(graph.edge_targets("agent")[0].value, "user")
        self.assertEqual(graph.edge_targets("patient")[0].value, "assistant")

    def test_temporal_and_comparative_relations_are_separate_nodes(self) -> None:
        before = parse_meaning("Build this before launch")
        comparison = parse_meaning("This engine is faster than the other engine")
        self.assertEqual(before.temporal_relation, "before")
        self.assertEqual(comparison.comparison, "more_than")

    def test_discourse_state_resolves_roles_and_tracks_turns(self) -> None:
        state = DiscourseState()
        self.engine.discourse = state
        result = self.engine.answer("Nice to meet u MCR")
        self.assertFalse(result.abstained)
        self.assertEqual(state.turn_count, 1)
        self.assertEqual(state.last_speech_act, "social.pleasantry")
        self.assertEqual(state.resolve_pronoun("you"), "assistant")

    def test_single_lexical_hit_cannot_pass_factual_evidence_gate(self) -> None:
        self.store.upsert_records([
            {
                "id": "one-hit",
                "title": "Recurrent situation model",
                "text": "A recurrent model retains a compact state.",
                "answer": "A recurrent model retains a compact state.",
                "source": "Local test reference",
                "keywords": ["recurrent"],
            }
        ])
        result = self.engine.answer("What is recurrent quantum machinery?")
        self.assertTrue(result.abstained)
        self.assertEqual(result.diagnostics["mode"], "evidence_gate")
        self.assertLess(result.diagnostics["evidence"]["term_coverage"], 0.70)

    def test_unknown_factual_question_abstains(self) -> None:
        result = self.engine.answer("What is an unknown semantic mechanism?")
        self.assertTrue(result.abstained)
        self.assertIn("does not contain enough", result.text)


if __name__ == "__main__":
    unittest.main()
