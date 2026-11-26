from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from mcrl.conllu import ConlluToken, iter_conllu
from mcrl.syntax_model import MathematicalSyntaxModel, evaluate_model, train_models
from mcrl.semantic_network import LexicalSemanticNetwork
from mcrl.compositional_semantics import compose_meaning
from mcrl.syntax_model import ParseResult, ParseToken


def token(i, form, lemma, upos, head, relation, feats="_"):
    return ConlluToken(i, form, lemma, upos, upos, feats, head, relation, "_", "_")


TRAINING_SENTENCES = [
    [
        token(1, "I", "I", "PRON", 2, "nsubj", "Case=Nom|Number=Sing|Person=1|PronType=Prs"),
        token(2, "like", "like", "VERB", 0, "root", "Mood=Ind|Tense=Pres|VerbForm=Fin"),
        token(3, "pizza", "pizza", "NOUN", 2, "obj", "Number=Sing"),
        token(4, ".", ".", "PUNCT", 2, "punct"),
    ],
    [
        token(1, "You", "you", "PRON", 2, "nsubj", "Case=Nom|Number=Sing|Person=2|PronType=Prs"),
        token(2, "teach", "teach", "VERB", 0, "root", "Mood=Ind|Tense=Pres|VerbForm=Fin"),
        token(3, "me", "I", "PRON", 2, "obj", "Case=Acc|Number=Sing|Person=1|PronType=Prs"),
        token(4, ".", ".", "PUNCT", 2, "punct"),
    ],
    [
        token(1, "She", "she", "PRON", 2, "nsubj", "Case=Nom|Gender=Fem|Number=Sing|Person=3|PronType=Prs"),
        token(2, "reads", "read", "VERB", 0, "root", "Mood=Ind|Tense=Pres|VerbForm=Fin"),
        token(3, "books", "book", "NOUN", 2, "obj", "Number=Plur"),
        token(4, ".", ".", "PUNCT", 2, "punct"),
    ],
    [
        token(1, "Do", "do", "AUX", 3, "aux", "Mood=Ind|Tense=Pres|VerbForm=Fin"),
        token(2, "you", "you", "PRON", 3, "nsubj", "Case=Nom|Number=Sing|Person=2|PronType=Prs"),
        token(3, "like", "like", "VERB", 0, "root", "VerbForm=Inf"),
        token(4, "music", "music", "NOUN", 3, "obj", "Number=Sing"),
        token(5, "?", "?", "PUNCT", 3, "punct"),
    ],
    [
        token(1, "I", "I", "PRON", 4, "nsubj", "Case=Nom|Number=Sing|Person=1|PronType=Prs"),
        token(2, "do", "do", "AUX", 4, "aux", "Mood=Ind|Tense=Pres|VerbForm=Fin"),
        token(3, "not", "not", "PART", 4, "advmod", "_"),
        token(4, "like", "like", "VERB", 0, "root", "Mood=Ind|Tense=Pres|VerbForm=Fin"),
        token(5, "pizza", "pizza", "NOUN", 4, "obj", "Number=Sing"),
        token(6, ".", ".", "PUNCT", 4, "punct"),
    ],
]


class MathematicalSyntaxTests(unittest.TestCase):
    def test_conllu_reader_skips_multiword_rows_and_preserves_dependency_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.conllu"
            path.write_text(
                "# sent_id = 1\n"
                "1-2\\tdon't\\t_\\t_\\t_\\t_\\t_\\t_\\t_\\t_\\n"
                "1\\tdo\\tdo\\tAUX\\tVBP\\tVerbForm=Fin\\t3\\taux\\t_\\t_\\n"
                "2\\tn't\\tnot\\tPART\\tRB\\tPolarity=Neg\\t3\\tadvmod\\t_\\t_\\n"
                "3\\tgo\\tgo\\tVERB\\tVB\\tVerbForm=Inf\\t0\\troot\\t_\\t_\\n\\n",
                encoding="utf-8",
            )
            sentence = next(iter_conllu(path))
        self.assertEqual([item.form for item in sentence], ["do", "n't", "go"])
        self.assertEqual(sentence[1].head, 3)
        self.assertEqual(sentence[1].deprel, "advmod")

    def test_training_builds_a_pure_cpu_pos_and_dependency_model(self):
        corpus = TRAINING_SENTENCES * 8
        model, counts = train_models(corpus, epochs=5)
        self.assertGreater(counts["tokens"], 0)
        self.assertTrue(model.pos_labels)
        self.assertTrue(model.dependency_labels)
        metrics = evaluate_model(model, TRAINING_SENTENCES)
        self.assertGreater(metrics["pos_accuracy"], 0.65)
        self.assertGreater(metrics["complete_tree_fraction"], 0.8)

    def test_model_round_trip_and_unknown_words_still_receive_syntactic_nodes(self):
        model, _ = train_models(TRAINING_SENTENCES * 8, epochs=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "syntax.json"
            model.save(path)
            loaded = MathematicalSyntaxModel.load(path)
        parsed = loaded.parse_text("I like quuxblorf.")
        self.assertEqual([item.form for item in parsed.tokens], ["I", "like", "quuxblorf", "."])
        self.assertTrue(all(item.deprel != "unparsed" for item in parsed.tokens))
        self.assertTrue(any(item.form == "quuxblorf" for item in parsed.tokens))

    def test_dependency_composition_exposes_subject_object_and_negation(self):
        model, _ = train_models(TRAINING_SENTENCES * 10, epochs=6)
        parsed = model.parse_forms(
            ["I", "do", "not", "like", "pizza", "."],
            lemmas=["I", "do", "not", "like", "pizza", "."],
        )
        lexicon = LexicalSemanticNetwork(":memory:")
        try:
            meaning = compose_meaning("I do not like pizza.", parsed, None)
        finally:
            lexicon.close()
        self.assertTrue(meaning["parse_complete"])
        self.assertTrue(meaning["clauses"])
        clause = meaning["clauses"][0]
        self.assertEqual(clause["predicate"], "like")
        self.assertEqual(clause["polarity"], "negative")
        self.assertTrue(any(role["role"] == "grammatical_subject" for role in clause["roles"]))
        self.assertTrue(any(role["role"] == "direct_object_or_patient_candidate" for role in clause["roles"]))
        self.assertFalse(meaning["language_model_used_for_understanding"])

    def test_context_dependent_short_reaction_points_to_previous_turn(self):
        parsed = ParseResult(
            text="really ?",
            tokens=(
                ParseToken(1, "really", "really", "ADV", head=0, deprel="root"),
                ParseToken(2, "?", "?", "PUNCT", head=1, deprel="punct"),
            ),
        )
        meaning = compose_meaning(
            "really?",
            parsed,
            history=[{"user": "Previous utterance", "meaning": "prior proposition"}],
        )
        self.assertEqual(
            meaning["communicative_function"],
            "fragment_or_elliptical_follow_up",
        )
        self.assertTrue(meaning["discourse_reference_candidates"])
        self.assertIn("possible reference to the previous turn", meaning["meaning_summary"])

    def test_question_and_declarative_sentences_have_different_structural_functions(self):
        model, _ = train_models(TRAINING_SENTENCES * 10, epochs=6)
        question = model.parse_forms(
            ["Do", "you", "like", "music", "?"],
            lemmas=["do", "you", "like", "music", "?"],
        )
        statement = model.parse_forms(
            ["You", "like", "music", "."],
            lemmas=["you", "like", "music", "."],
        )
        question_meaning = compose_meaning("Do you like music?", question)
        statement_meaning = compose_meaning("You like music.", statement)
        self.assertEqual(question_meaning["communicative_function"], "interrogative")
        self.assertEqual(question_meaning["question_type"], "polar_or_yes_no_question")
        self.assertEqual(statement_meaning["communicative_function"], "declarative_or_assertive_candidate")


if __name__ == "__main__":
    unittest.main()
