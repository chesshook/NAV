from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from mcrl.calculator import try_calculate
from mcrl.discourse_state import DiscourseState
from mcrl.engine import MCRLanguageEngine
from mcrl.knowledge import KnowledgeStore
from mcrl.meaning_solver import solve_response
from mcrl.meaning_validator import validate_response
from mcrl.pragmatics import analyze_pragmatics
from mcrl.interaction_composer import plan_interaction, summarize_session
from mcrl.semantic_parser import parse_meaning
from mcrl.sensitivity import calculate_sensitivity


ROOT = Path(__file__).resolve().parents[1]


class SemanticCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = KnowledgeStore(":memory:")
        self.engine = MCRLanguageEngine(self.store)

    def tearDown(self) -> None:
        self.store.close()

    def load_optional_moon_sample(self) -> None:
        """Load the demo fact file only in tests that explicitly request it."""
        facts = ROOT / "mcrl" / "data" / "facts.jsonl"
        self.store.import_facts_jsonl(str(facts))

    def test_engine_does_not_preload_a_topic_knowledge_domain(self) -> None:
        self.assertEqual(self.store.fact_count(), 0)
        self.assertEqual(self.store.count(), 0)
        result = self.engine.answer("What do you know about the Moon?")
        self.assertTrue(result.abstained)
        self.assertEqual(result.diagnostics["mode"], "unknown_topic_or_property")
        self.assertIn("don't know enough about Moon", result.text)
        self.assertNotIn("white", result.text.lower())

    def test_nice_to_meet_u_mcr_is_composed_from_meaning(self) -> None:
        result = self.engine.answer("Nice to meet u MCR")
        self.assertEqual(result.text, "Nice to meet you too!")
        self.assertFalse(result.abstained)
        self.assertEqual(result.diagnostics["mode"], "dialogue_composition")
        self.assertEqual(result.diagnostics["verification"]["preservation"], 1.0)

    def test_hello_there_preserves_the_salutation_extension(self) -> None:
        graph = parse_meaning("Hello there")
        self.assertEqual(graph.speech_act, "social.greeting")
        self.assertFalse(graph.by_kind("topic"))
        result = self.engine.answer("Hello there")
        self.assertEqual(result.text, "Hello there!")
        self.assertFalse(result.abstained)
        self.assertTrue(result.diagnostics["verification"]["valid"])

    def test_are_u_there_is_a_presence_check_not_a_clarification(self) -> None:
        graph = parse_meaning("are u there?")
        self.assertEqual(graph.speech_act, "request.presence_check")
        self.assertEqual(graph.predicate, "presence_check")
        self.assertEqual(graph.edge_targets("asker")[0].value, "user")
        self.assertEqual(graph.edge_targets("target")[0].value, "assistant")
        result = self.engine.answer("are u there?")
        self.assertEqual(result.text, "Yes, I am here!")
        self.assertFalse(result.abstained)
        self.assertEqual(
            result.diagnostics["verification"]["reparsed_speech_act"],
            "response.presence_confirmation",
        )

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

    def test_moon_topic_answer_uses_only_explicitly_loaded_facts(self) -> None:
        self.load_optional_moon_sample()
        result = self.engine.answer("Tell me about the Moon")
        self.assertFalse(result.abstained)
        self.assertEqual(result.diagnostics["mode"], "fact_composition")
        self.assertIn("bright, almost white", result.text)
        self.assertIn("dark-gray", result.text)
        self.assertIn("reflected from its surface", result.text)
        self.assertTrue(result.diagnostics["topic_answer"]["sources"])
        self.assertTrue(
            any(
                source["url"].startswith("https://science.nasa.gov/")
                for source in result.diagnostics["topic_answer"]["sources"]
            )
        )

    def test_what_do_you_know_about_moon_maps_to_the_same_topic(self) -> None:
        self.load_optional_moon_sample()
        result = self.engine.answer("What do u know about the moon?")
        self.assertFalse(result.abstained)
        self.assertEqual(result.diagnostics["mode"], "fact_composition")
        self.assertIn("bright, almost white", result.text)

    def test_moon_followup_resolves_it_and_selects_size_fact(self) -> None:
        self.load_optional_moon_sample()
        first = self.engine.answer("Tell me about the Moon")
        second = self.engine.answer("Is it big?")
        self.assertFalse(first.abstained)
        self.assertFalse(second.abstained)
        self.assertIn("one-fourth as wide as Earth", second.text)
        self.assertTrue(
            any(
                item["predicate"] == "diameter_relative_to_earth"
                for item in second.diagnostics["topic_answer"]["selected_facts"]
            )
        )

    def test_indirect_distance_question_is_recognized_without_question_mark(self) -> None:
        graph = parse_meaning("can u tell me how much distance is from moon to earth")
        self.assertEqual(graph.intent, "question")
        self.assertEqual(graph.first("topic"), "moon earth")
        self.assertTrue(self.engine.answer("can u tell me how much distance is from moon to earth").abstained)

    def test_distance_question_does_not_fall_back_to_unrelated_moon_facts(self) -> None:
        self.load_optional_moon_sample()
        result = self.engine.answer("Can u tell me how much distance is from moon to earth?")
        self.assertTrue(result.abstained)
        self.assertEqual(result.diagnostics["mode"], "unknown_topic_or_property")
        self.assertIn("reliable distance fact", result.text)
        self.assertNotIn("white", result.text.lower())
        self.assertNotIn("dark-gray", result.text.lower())
        self.assertEqual(result.diagnostics["topic_answer"]["selected_facts"], [])

    def test_fact_jsonl_import_preserves_provenance(self) -> None:
        record = {
            "id": "venus-sample-fact",
            "subject": "Venus",
            "aliases": ["morning star"],
            "predicate": "example_fact",
            "object": "Venus is a planet.",
            "confidence": 0.98,
            "epistemic_status": "established",
            "priority": 10,
            "source": "Unit test source",
            "source_url": "https://example.org/venus",
            "tags": ["astronomy"],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "facts.jsonl"
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")
            imported = self.store.import_facts_jsonl(str(source), batch_size=1)
        self.assertEqual(imported, 1)
        found = self.store.search_facts(["venus"], subject="Venus")
        self.assertTrue(any(item["fact_id"] == "venus-sample-fact" for item in found))
        venus = next(item for item in found if item["fact_id"] == "venus-sample-fact")
        self.assertEqual(venus["source_url"], "https://example.org/venus")
        self.assertEqual(venus["epistemic_status"], "established")

    def test_unknown_topic_is_admitted_without_fabricating_a_fact(self) -> None:
        result = self.engine.answer("Anything about Neptune?")
        self.assertTrue(result.abstained)
        self.assertEqual(result.diagnostics["mode"], "unknown_topic_or_property")
        self.assertIn("don't know enough about Neptune", result.text)
        self.assertNotIn("Neptune is", result.text)

    def test_unrelated_new_topic_does_not_inherit_previous_topic(self) -> None:
        first = self.engine.answer("Anything about Neptune?")
        second = self.engine.answer("What is the capital of France?")
        self.assertTrue(first.abstained)
        self.assertTrue(second.abstained)
        self.assertIn("France", second.text)
        self.assertNotIn("Neptune", second.text)
        self.assertEqual(second.diagnostics["discourse"]["last_topic"].lower(), "france")

    def test_calculation_is_independent_of_previous_topic(self) -> None:
        self.engine.answer("Anything about Neptune?")
        result = self.engine.answer("Can u tell me the value of sum of 4 and 5?")
        self.assertFalse(result.abstained)
        self.assertEqual(result.diagnostics["mode"], "calculation")
        self.assertEqual(result.diagnostics["result"], 9)
        self.assertEqual(result.text, "4 + 5 = 9.")
        self.assertNotIn("Neptune", result.text)

    def test_sum_extraction_is_safe_and_bounded(self) -> None:
        result = try_calculate("What is the sum of 4 and 5?")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.result, 9)
        self.assertIsNone(try_calculate("sum of 4 and __import__('os').system('whoami')"))

    def test_thats_great_is_acknowledged_without_overwriting_topic(self) -> None:
        self.engine.answer("Anything about Neptune?")
        result = self.engine.answer("thats great")
        self.assertFalse(result.abstained)
        self.assertEqual(result.diagnostics["mode"], "dialogue_composition")
        self.assertEqual(result.diagnostics["response_plan"]["act"], "acknowledge_positive_statement")
        self.assertEqual(result.diagnostics["discourse"]["last_topic"].lower(), "neptune")
        self.assertIn("glad", result.text.lower())

    def test_dont_tell_me_about_topic_is_acknowledged_as_a_boundary(self) -> None:
        result = self.engine.answer("dont tell me anything about moon")
        self.assertFalse(result.abstained)
        self.assertEqual(result.diagnostics["mode"], "dialogue_composition")
        self.assertEqual(result.diagnostics["response_plan"]["act"], "acknowledge_topic_avoidance")
        self.assertIn("won’t bring up moon", result.text.lower())

    def test_color_question_selects_only_color_facts(self) -> None:
        self.load_optional_moon_sample()
        result = self.engine.answer("What color is the Moon?")
        self.assertFalse(result.abstained)
        predicates = {
            item["predicate"]
            for item in result.diagnostics["topic_answer"]["selected_facts"]
        }
        self.assertEqual(
            predicates,
            {"apparent_color_from_earth", "surface_color_up_close"},
        )

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

    def test_bare_what_is_a_clarification_request_not_a_fact_lookup(self) -> None:
        graph = parse_meaning("what?")
        self.assertEqual(graph.speech_act, "request.clarification")
        self.assertEqual(graph.lexical_tokens, ())
        self.assertFalse(graph.by_kind("polarity"))
        result = self.engine.answer("what?")
        self.assertEqual(result.text, "What would you like me to clarify?")
        self.assertFalse(result.abstained)
        self.assertEqual(result.diagnostics["mode"], "dialogue_composition")
        self.assertTrue(result.diagnostics["verification"]["valid"])

    def test_greeting_graph_connects_speaker_and_addressee_to_event(self) -> None:
        graph = parse_meaning("hi")
        self.assertEqual(graph.speech_act, "social.greeting")
        self.assertEqual(graph.first("event"), "greeting")
        self.assertEqual(graph.edge_targets("speaker")[0].value, "user")
        self.assertEqual(graph.edge_targets("addressee")[0].value, "assistant")

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
        self.assertEqual(result.diagnostics["mode"], "unknown_topic_or_property")
        self.assertNotIn("Recurrent situation model", result.text)

    def test_unknown_factual_question_abstains(self) -> None:
        result = self.engine.answer("What is an unknown semantic mechanism?")
        self.assertTrue(result.abstained)
        self.assertIn("don't know enough", result.text.lower())



    def test_compound_greeting_and_wellbeing_preserves_both_moves(self) -> None:
        utterance = "Hi! How is it going?"
        graph = parse_meaning(utterance)
        self.assertEqual(graph.speech_act, "social.greeting")
        self.assertEqual(
            set(graph.speech_acts),
            {"social.greeting", "social.wellbeing"},
        )
        self.assertFalse(graph.by_kind("topic"))
        result = self.engine.answer(utterance)
        self.assertFalse(result.abstained)
        self.assertEqual(
            result.diagnostics["response_plan"]["act"],
            "greet_and_answer_wellbeing",
        )
        self.assertIn("ready to help", result.text.lower())
        self.assertIn("thanks for asking", result.text.lower())
        self.assertEqual(
            result.diagnostics["discourse"]["last_speech_act"],
            "social.greeting+social.wellbeing",
        )
        self.assertEqual(result.diagnostics["discourse"]["last_topic"], "")

    def test_assistant_compliment_is_social_not_a_factual_topic(self) -> None:
        utterance = "Great Mannn u are a super hero"
        graph = parse_meaning(utterance)
        self.assertEqual(graph.speech_act, "social.compliment")
        self.assertFalse(graph.by_kind("topic"))
        self.assertEqual(graph.edge_targets("addressee")[0].value, "assistant")
        result = self.engine.answer(utterance)
        self.assertFalse(result.abstained)
        self.assertEqual(
            result.diagnostics["response_plan"]["act"],
            "acknowledge_compliment",
        )
        self.assertIn("kind of you", result.text.lower())
        self.assertNotIn("what would you like to explore", result.text.lower())
        self.assertEqual(result.diagnostics["discourse"]["last_topic"], "")

    def test_user_identity_question_does_not_extract_am_as_a_topic(self) -> None:
        graph = parse_meaning("do you know who I am?")
        self.assertEqual(graph.speech_act, "request.user_identity")
        self.assertFalse(graph.by_kind("topic"))
        result = self.engine.answer("do you know who I am?")
        self.assertFalse(result.abstained)
        self.assertEqual(
            result.diagnostics["response_plan"]["act"],
            "answer_user_identity",
        )
        self.assertIn("only know what you've shared", result.text.lower())
        self.assertIn("can't identify you independently", result.text.lower())
        self.assertNotIn("about Am", result.text)
        score_components = result.diagnostics["response_plan"]["score_components"]
        self.assertGreater(score_components["semantic_preservation"], 0.0)
        self.assertGreater(score_components["pragmatic_task_fit"], 0.0)

    def test_assistant_identity_variants_share_one_semantic_route(self) -> None:
        examples = (
            "who are u",
            "what's your name",
            "tell me your name",
            "do you know about yourself?",
        )
        for utterance in examples:
            with self.subTest(utterance=utterance):
                graph = parse_meaning(utterance)
                self.assertEqual(graph.speech_act, "request.identity")
                self.assertFalse(graph.by_kind("topic"))
                result = self.engine.answer(utterance)
                self.assertFalse(result.abstained)
                self.assertEqual(
                    result.diagnostics["response_plan"]["act"],
                    "identify_self",
                )
                self.assertIn("mcr-l", result.text.lower())

    def test_capability_question_is_not_misread_as_clarification(self) -> None:
        utterance = "do u know how to sum?"
        graph = parse_meaning(utterance)
        self.assertEqual(graph.speech_act, "request.capability")
        self.assertEqual(graph.first("capability"), "arithmetic")
        self.assertFalse(graph.by_kind("topic"))
        result = self.engine.answer(utterance)
        self.assertFalse(result.abstained)
        self.assertEqual(
            result.diagnostics["response_plan"]["act"],
            "answer_capability",
        )
        self.assertTrue(result.text.lower().startswith("yes"))
        self.assertIn("arithmetic", result.text.lower())

    def test_direct_verb_form_addition_is_evaluated(self) -> None:
        result = self.engine.answer("can u sum 2 and 3?")
        self.assertFalse(result.abstained)
        self.assertEqual(result.diagnostics["mode"], "calculation")
        self.assertEqual(result.diagnostics["result"], 5)
        self.assertEqual(result.text, "2 + 3 = 5.")


    def test_pragmatic_hypotheses_have_weighted_evidence_and_sensitivity(self) -> None:
        signals = analyze_pragmatics(("do", "you", "know", "who", "i", "am"))
        self.assertEqual(signals.identity_target, "user")
        candidate = next(
            item for item in signals.candidates
            if item.label == "identity_question" and item.target == "user"
        )
        self.assertGreaterEqual(candidate.score, 0.95)
        self.assertAlmostEqual(sum(candidate.sensitivity.values()), 1.0, places=6)
        self.assertIn("identity_pattern", candidate.sensitivity)
        graph = parse_meaning("do you know who I am?")
        act_node = graph.by_kind("speech_act")[0]
        self.assertAlmostEqual(act_node.confidence, candidate.score, places=6)
        diagnostics = signals.diagnostics()
        self.assertEqual(diagnostics["method"], "weighted_semantic_constraint_support")
        self.assertIn("not a calibrated probability", diagnostics["score_semantics"])

    def test_negated_positive_attribution_is_not_scored_as_a_compliment(self) -> None:
        signals = analyze_pragmatics(("you", "are", "not", "helpful"))
        self.assertFalse(signals.compliment)
        self.assertFalse(
            any(item.label == "social_compliment" for item in signals.candidates)
        )

    def test_situation_model_tracks_compound_moves_and_participant_roles(self) -> None:
        result = self.engine.answer("Hi! How is it going?")
        situation = result.diagnostics["situation"]
        self.assertEqual(situation["active_goal"], "social_interaction")
        self.assertEqual(
            situation["last_user_moves"],
            ["social.greeting", "social.wellbeing"],
        )
        self.assertEqual(situation["last_speaker"], "user")
        self.assertEqual(situation["last_addressee"], "assistant")
        self.assertFalse(situation["pending_question"])
        self.assertEqual(situation["current_topic"], "")

    def test_situation_model_preserves_referent_across_social_acknowledgement(self) -> None:
        self.engine.answer("Anything about Neptune?")
        result = self.engine.answer("That's great")
        self.assertEqual(
            result.diagnostics["situation"]["current_topic"].lower(),
            "neptune",
        )
        self.assertEqual(result.diagnostics["situation"]["active_goal"], "social_interaction")

    def test_unknown_capability_is_admitted_without_false_claim(self) -> None:
        result = self.engine.answer("Do you know how to swim?")
        self.assertFalse(result.abstained)
        self.assertEqual(result.diagnostics["response_plan"]["act"], "answer_capability")
        self.assertIn("don't have a reliable swim capability", result.text.lower())
        self.assertIn("counterfactual_sensitivity", result.diagnostics["pragmatic_interpretation"]["candidates"][0])



    def test_smile_request_is_an_action_not_a_factual_topic(self) -> None:
        plan = plan_interaction("can u smile for me?")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.operation, "perform_symbolic_action")
        self.assertEqual(plan.action, "smile")
        result = self.engine.answer("can u smile for me?")
        self.assertEqual(result.text, "😊")
        self.assertEqual(result.diagnostics["mode"], "interaction_composition")
        self.assertNotEqual(result.diagnostics["situation"]["current_topic"].lower(), "smile")

    def test_unknown_physical_action_is_not_misrouted_as_a_topic(self) -> None:
        result = self.engine.answer("can you sing for me?")
        self.assertFalse(result.abstained)
        self.assertIn("can't physically sing", result.text.lower())
        self.assertEqual(
            result.diagnostics["interaction_plan"]["operation"],
            "perform_symbolic_action",
        )

    def test_standalone_wowow_is_a_reaction_not_a_subject_lookup(self) -> None:
        plan = plan_interaction("wowow")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.operation, "acknowledge_interjection")
        result = self.engine.answer("wowow")
        self.assertEqual(result.diagnostics["mode"], "interaction_composition")
        self.assertIn("what's going on", result.text.lower())
        self.assertNotIn("Wowow", result.diagnostics.get("topic_answer", {}).get("topic", ""))

    def test_explicit_speech_directive_overrides_embedded_identity_clause(self) -> None:
        utterance = "dont tell me who are you, just say Helo"
        plan = plan_interaction(utterance)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.operation, "emit_user_requested_text")
        self.assertEqual(plan.payload, "Helo")
        self.assertIn("respect_negated_identity_disclosure", plan.constraints)
        result = self.engine.answer(utterance)
        self.assertEqual(result.text, "Helo")
        self.assertEqual(result.diagnostics["mode"], "interaction_composition")
        self.assertNotIn("MCR-L", result.text)

    def test_direct_output_instruction_is_compositional_across_payloads(self) -> None:
        examples = (
            ('Please just say "Good morning".', "Good morning"),
            ("Only say ready", "ready"),
            ("Reply with I understand.", "I understand"),
            ("answer with 42", "42"),
        )
        for utterance, expected in examples:
            with self.subTest(utterance=utterance):
                result = self.engine.answer(utterance)
                self.assertEqual(result.text, expected)
                self.assertEqual(result.diagnostics["mode"], "interaction_composition")

    def test_conversation_summary_uses_actual_prior_session_messages(self) -> None:
        self.engine.answer("Hi! How is it going?")
        self.engine.answer("can u smile for me?")
        self.engine.answer("Do you know what is a computer?")
        result = self.engine.answer("what we talked till now?")
        self.assertEqual(result.diagnostics["mode"], "interaction_composition")
        self.assertIn("Hi! How is it going?", result.text)
        self.assertIn("can u smile for me?", result.text)
        self.assertIn("Do you know what is a computer?", result.text)
        self.assertNotIn("what we talked till now?", result.text)

    def test_history_summary_is_empty_when_session_has_no_previous_turns(self) -> None:
        result = self.engine.answer("what have we talked about so far?")
        self.assertIn("history is empty", result.text.lower())

    def test_history_question_does_not_become_a_topic_named_talked(self) -> None:
        plan = plan_interaction("what we talked till now?")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.operation, "summarize_session_history")

    def test_computer_definition_stays_a_knowledge_question_not_a_topic_guess(self) -> None:
        result = self.engine.answer("do you know what is a computer?")
        self.assertTrue(result.abstained)
        self.assertEqual(result.diagnostics["mode"], "unknown_topic_or_property")
        self.assertIn("Computer", result.text)
        self.assertNotIn("Talked", result.text)
        self.assertNotIn("Smile", result.text)


if __name__ == "__main__":
    unittest.main()
