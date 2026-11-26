from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from mcrl.semantic_network import LexicalSemanticNetwork


LMF_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<LexicalResource>
  <Lexicon id="en" label="test">
    <LexicalEntry id="bank-fin" partOfSpeech="n">
      <Lemma writtenForm="bank"/>
      <Form writtenForm="banks"/>
      <Sense id="bank-fin-sense" synset="bank-financial">
        <SenseRelation relType="antonym" target="bank-riv-sense"/>
      </Sense>
    </LexicalEntry>
    <LexicalEntry id="bank-riv" partOfSpeech="n">
      <Lemma writtenForm="bank"/>
      <Sense id="bank-riv-sense" synset="bank-river"/>
    </LexicalEntry>
    <LexicalEntry id="money" partOfSpeech="n">
      <Lemma writtenForm="money"/>
      <Sense id="money-sense" synset="money-concept"/>
    </LexicalEntry>
    <LexicalEntry id="river" partOfSpeech="n">
      <Lemma writtenForm="river"/>
      <Sense id="river-sense" synset="river-concept"/>
    </LexicalEntry>
    <LexicalEntry id="ice-cream" partOfSpeech="n">
      <Lemma writtenForm="ice cream"/>
      <Sense id="ice-cream-sense" synset="ice-cream-concept"/>
    </LexicalEntry>
    <Synset id="bank-financial" partOfSpeech="n">
      <Definition>A financial institution that accepts deposits and lends money.</Definition>
      <Example>I deposited money at the bank.</Example>
      <SynsetRelation relType="hypernym" target="financial-institution"/>
    </Synset>
    <Synset id="bank-river" partOfSpeech="n">
      <Definition>Sloping land beside a river or other body of water.</Definition>
      <Example>He sat on the river bank.</Example>
    </Synset>
    <Synset id="money-concept" partOfSpeech="n">
      <Definition>A medium of exchange used to buy goods and services.</Definition>
    </Synset>
    <Synset id="river-concept" partOfSpeech="n">
      <Definition>A large natural stream of flowing water.</Definition>
    </Synset>
    <Synset id="ice-cream-concept" partOfSpeech="n">
      <Definition>A sweet frozen food made from milk or cream.</Definition>
    </Synset>
  </Lexicon>
</LexicalResource>
"""


class SemanticNetworkTests(unittest.TestCase):
    def setUp(self):
        self.network = LexicalSemanticNetwork(":memory:")
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.network.close()
        self.temp.cleanup()

    def import_sample(self):
        path = Path(self.temp.name) / "sample.xml"
        path.write_text(LMF_SAMPLE, encoding="utf-8")
        counts = self.network.import_lmf(
            path,
            dataset="unit-test-wordnet",
            license_name="test-only",
            source_url="https://example.org/test-wordnet",
        )
        return counts

    def test_lmf_import_builds_lexemes_senses_and_typed_relations(self):
        counts = self.import_sample()
        self.assertEqual(counts["concepts"], 5)
        self.assertGreaterEqual(counts["lemmas"], 5)
        self.assertEqual(
            self.network.lookup("bank")[0]["lemma"],
            "bank",
        )
        relation = self.network.conn.execute(
            "SELECT relation_type FROM relations "
            "WHERE source_synset='bank-financial' AND target_synset='financial-institution'"
        ).fetchone()
        self.assertIsNotNone(relation)
        self.assertEqual(relation[0], "hypernym")
        self.assertEqual(counts["sense_relations"], 1)
        self.assertEqual(
            self.network._relation_support(
                "bank-financial", "bank-river",
                "bank-fin-sense", "bank-riv-sense",
            ),
            0.88,
        )

    def test_word_sense_is_selected_from_sentence_context_not_last_word(self):
        self.import_sample()
        analysis = self.network.analyze("I deposited money at the bank.")
        bank = next(item for item in analysis.tokens if item.surface == "bank")
        self.assertEqual(bank.status, "resolved")
        self.assertEqual(bank.selected_sense, "bank-fin-sense")
        self.assertGreater(bank.candidates[0].total_support, bank.candidates[-1].total_support)

    def test_current_utterance_overrides_an_old_topic_in_sense_scoring(self):
        self.import_sample()
        analysis = self.network.analyze(
            "money bank",
            dialogue_context="river water shore land river bank",
        )
        bank = next(item for item in analysis.tokens if item.surface == "bank")
        self.assertEqual(bank.selected_sense, "bank-fin-sense")

    def test_dictionary_inflections_map_to_the_lemma_sense(self):
        self.import_sample()
        banks = self.network.analyze("banks")
        self.assertEqual(banks.tokens[0].status, "resolved")
        self.assertEqual(banks.tokens[0].selected_sense, "bank-fin-sense")

    def test_ambiguous_word_is_not_forced_when_context_does_not_disambiguate(self):
        self.import_sample()
        analysis = self.network.analyze("bank")
        bank = analysis.tokens[0]
        self.assertEqual(bank.status, "ambiguous")
        self.assertIsNone(bank.selected_sense)
        self.assertEqual(len(bank.candidates), 2)

    def test_multiword_dictionary_concept_is_preserved_as_phrase_node(self):
        self.import_sample()
        analysis = self.network.analyze("I like ice cream.")
        self.assertTrue(any(item["phrase"] == "ice cream" for item in analysis.phrase_activations))
        view = analysis.prompt_view()
        self.assertTrue(any(item["phrase"] == "ice cream" for item in view["phrase_activations"]))

    def test_unknown_words_are_retained_as_subword_nodes(self):
        self.import_sample()
        analysis = self.network.analyze("quuxblorf")
        self.assertEqual(analysis.unresolved_words, ("quuxblorf",))
        unknown = analysis.tokens[0]
        self.assertEqual(unknown.status, "unknown")
        self.assertTrue(unknown.subword_units)

    def test_chat_spelling_maps_to_dictionary_lemma_without_discarding_surface(self):
        self.import_sample()
        # Add a 'you' entry to the small test dictionary.
        self.network.import_jsonl(
            self._write_jsonl([{
                "id": "you-pronoun", "pos": "n",
                "definition": "The person being addressed.",
                "lemmas": ["you"], "examples": [], "relations": [],
            }]),
            dataset="test-chat-normalization",
            license_name="test-only",
        )
        analysis = self.network.analyze("u")
        self.assertEqual(analysis.tokens[0].surface, "u")
        self.assertIn("you", analysis.tokens[0].normalized_forms)
        self.assertEqual(analysis.tokens[0].status, "resolved")

    def _write_jsonl(self, records):
        path = Path(self.temp.name) / "lexicon.jsonl"
        path.write_text(
            "\n".join(json.dumps(item) for item in records) + "\n",
            encoding="utf-8",
        )
        return path

    def test_custom_jsonl_import_is_streamed_into_same_graph(self):
        path = self._write_jsonl([{
            "id": "concept-a",
            "pos": "n",
            "definition": "A simple test concept.",
            "examples": ["This concept is used in a sentence."],
            "lemmas": ["concept alpha"],
            "relations": [{"type": "hypernym", "target": "concept-b"}],
        }, {
            "id": "concept-b",
            "pos": "n",
            "definition": "A wider concept.",
            "examples": [],
            "lemmas": ["concept beta"],
            "relations": [],
        }])
        counts = self.network.import_jsonl(
            path,
            dataset="custom-test",
            license_name="CC0",
        )
        self.assertEqual(counts["concepts"], 2)
        self.assertEqual(counts["relations"], 1)
        self.assertTrue(self.network.lookup("concept alpha"))

    def test_empty_network_keeps_all_words_visible(self):
        analysis = self.network.analyze("can you explain quantum biology?")
        self.assertFalse(analysis.dictionary_ready)
        self.assertEqual(
            [item.surface for item in analysis.tokens],
            ["can", "you", "explain", "quantum", "biology"],
        )
        self.assertTrue(all(item.status == "unknown" for item in analysis.tokens))


if __name__ == "__main__":
    unittest.main()
