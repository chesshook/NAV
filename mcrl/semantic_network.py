"""Mathematical lexical-semantic network for MCR-L.

Every lexical item is retained, mapped to candidate senses when possible, scored
against utterance context, and connected through typed semantic relations. Unknown
words remain explicit nodes; they are never silently dropped or guessed.

This is a lexical-semantic layer, not a complete grammar parser or a claim of
human-level understanding. It provides inspectable evidence to the compositional
language layer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Iterator
import xml.etree.ElementTree as ET
import zipfile


TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*|\d+(?:[.,]\d+)*|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "hers", "him", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "me", "my", "of", "on", "or", "our", "ours", "she", "so",
    "that", "the", "their", "them", "these", "they", "this", "those", "to",
    "us", "was", "we", "were", "what", "when", "where", "which", "who", "whom",
    "why", "will", "with", "would", "you", "your", "yours",
}
CHAT_NORMALIZATION = {
    "u": ("you",),
    "ur": ("your",),
    "ya": ("you",),
    "thx": ("thanks",),
    "pls": ("please",),
    "plz": ("please",),
}
RELATION_WEIGHTS = {
    "hypernym": 0.82,
    "hyponym": 0.82,
    "instance_hypernym": 0.76,
    "instance_hyponym": 0.76,
    "part_meronym": 0.72,
    "member_meronym": 0.68,
    "substance_meronym": 0.62,
    "part_holonym": 0.72,
    "member_holonym": 0.68,
    "substance_holonym": 0.62,
    "antonym": 0.88,
    "similar": 0.58,
    "attribute": 0.55,
    "entails": 0.62,
    "causes": 0.62,
    "also": 0.35,
    "derivation": 0.32,
    "domain_topic": 0.25,
    "domain_region": 0.22,
    "other": 0.18,
}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _terms(value: str) -> set[str]:
    return {
        item.lower().replace("_", " ")
        for item in WORD_RE.findall(value or "")
        if len(item) > 1 and item.lower() not in STOPWORDS
    }


def _ngrams(value: str, minimum: int = 3, maximum: int = 5) -> list[str]:
    normalized = f"^{value.lower()}$"
    found: list[str] = []
    for width in range(minimum, maximum + 1):
        found.extend(normalized[i:i + width] for i in range(max(0, len(normalized) - width + 1)))
    return found[:48]


def _lemma_variants(word: str) -> tuple[str, ...]:
    """Conservative morphology candidates; variants are checked against the lexicon."""
    base = word.lower().replace("’", "'")
    candidates = [base]
    if len(base) >= 5:
        if base.endswith("ies"):
            candidates.append(base[:-3] + "y")
        if base.endswith("ves"):
            candidates.extend((base[:-3] + "f", base[:-3] + "fe"))
        if base.endswith("ing"):
            stem = base[:-3]
            candidates.extend((stem, stem + "e"))
            if len(stem) > 2 and stem[-1] == stem[-2]:
                candidates.append(stem[:-1])
        if base.endswith("ied"):
            candidates.append(base[:-3] + "y")
        if base.endswith("ed"):
            stem = base[:-2]
            candidates.extend((stem, stem + "e"))
            if len(stem) > 2 and stem[-1] == stem[-2]:
                candidates.append(stem[:-1])
        if base.endswith("es"):
            candidates.append(base[:-2])
        if base.endswith("s") and not base.endswith("ss"):
            candidates.append(base[:-1])
        if base.endswith("ly"):
            candidates.append(base[:-2])
        if base.endswith("est"):
            candidates.append(base[:-3])
        if base.endswith("er"):
            candidates.append(base[:-2])
    return tuple(dict.fromkeys(item for item in candidates if item))


@dataclass(frozen=True)
class SenseCandidate:
    synset_id: str
    sense_id: str
    lemma: str
    part_of_speech: str
    definition: str
    examples: tuple[str, ...]
    lexical_support: float
    relation_support: float
    total_support: float

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["examples"] = list(self.examples)
        return value


@dataclass(frozen=True)
class WordActivation:
    position: int
    surface: str
    normalized_forms: tuple[str, ...]
    status: str
    selected_sense: str | None
    ambiguity: str
    support_margin: float
    candidates: tuple[SenseCandidate, ...]
    subword_units: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "surface": self.surface,
            "normalized_forms": list(self.normalized_forms),
            "status": self.status,
            "selected_sense": self.selected_sense,
            "ambiguity": self.ambiguity,
            "support_margin": round(self.support_margin, 5),
            "candidates": [item.to_dict() for item in self.candidates],
            "subword_units": list(self.subword_units),
        }


@dataclass(frozen=True)
class SemanticAnalysis:
    original: str
    tokens: tuple[WordActivation, ...]
    activated_relations: tuple[dict[str, Any], ...]
    dictionary_ready: bool
    dictionary_counts: dict[str, int]
    phrase_activations: tuple[dict[str, Any], ...] = ()
    method: str = "contextual lexical activation plus typed relation consistency"

    @property
    def unresolved_words(self) -> tuple[str, ...]:
        return tuple(
            item.surface for item in self.tokens
            if item.status == "unknown" and any(ch.isalpha() for ch in item.surface)
        )

    def prompt_view(self, max_words: int = 48, max_senses: int = 3) -> dict[str, Any]:
        """Bound dictionary evidence before supplying it to another language stage."""
        words = []
        for item in self.tokens[:max_words]:
            words.append({
                "surface": item.surface,
                "normalized_forms": list(item.normalized_forms),
                "status": item.status,
                "ambiguity": item.ambiguity,
                "selected_sense": item.selected_sense,
                "candidate_senses": [
                    {
                        "synset_id": sense.synset_id,
                        "lemma": sense.lemma,
                        "pos": sense.part_of_speech,
                        "definition": sense.definition[:420],
                        "support": round(sense.total_support, 4),
                    }
                    for sense in item.candidates[:max_senses]
                ],
            })
        return {
            "method": self.method,
            "dictionary_ready": self.dictionary_ready,
            "words": words,
            "activated_relations": list(self.activated_relations[:32]),
            "phrase_activations": list(self.phrase_activations[:24]),
            "unresolved_words": list(self.unresolved_words[:24]),
            "notice": (
                "Lexical evidence proposes word senses; it does not by itself determine "
                "the sentence's full grammar, intent, truth, or speaker commitment."
            ),
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "dictionary_ready": self.dictionary_ready,
            "dictionary_counts": dict(self.dictionary_counts),
            "word_count": len(self.tokens),
            "resolved_count": sum(item.status == "resolved" for item in self.tokens),
            "ambiguous_count": sum(item.status == "ambiguous" for item in self.tokens),
            "unknown_count": len(self.unresolved_words),
            "tokens": [item.to_dict() for item in self.tokens],
            "activated_relations": list(self.activated_relations),
            "phrase_activations": list(self.phrase_activations),
        }


class LexicalSemanticNetwork:
    """SQLite-backed lexical concept graph with context-sensitive word-sense scoring."""

    SCHEMA = """
    PRAGMA foreign_keys = ON;
    CREATE TABLE IF NOT EXISTS concepts (
        synset_id TEXT PRIMARY KEY,
        part_of_speech TEXT NOT NULL DEFAULT '',
        definition TEXT NOT NULL DEFAULT '',
        examples_json TEXT NOT NULL DEFAULT '[]',
        ili TEXT NOT NULL DEFAULT '',
        dataset TEXT NOT NULL DEFAULT '',
        license TEXT NOT NULL DEFAULT '',
        source_url TEXT NOT NULL DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS lemmas (
        lemma_norm TEXT NOT NULL,
        lemma TEXT NOT NULL,
        part_of_speech TEXT NOT NULL DEFAULT '',
        synset_id TEXT NOT NULL,
        sense_id TEXT NOT NULL DEFAULT '',
        dataset TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (lemma_norm, part_of_speech, synset_id)
    );
    CREATE INDEX IF NOT EXISTS idx_lemmas_norm ON lemmas(lemma_norm);
    CREATE INDEX IF NOT EXISTS idx_lemmas_synset ON lemmas(synset_id);
    CREATE TABLE IF NOT EXISTS relations (
        source_synset TEXT NOT NULL,
        target_synset TEXT NOT NULL,
        relation_type TEXT NOT NULL,
        dataset TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (source_synset, target_synset, relation_type)
    );
    CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_synset);
    CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_synset);
    CREATE TABLE IF NOT EXISTS network_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS learned_associations (
        source_key TEXT NOT NULL,
        target_key TEXT NOT NULL,
        relation_type TEXT NOT NULL,
        weight REAL NOT NULL DEFAULT 0.5,
        evidence_count INTEGER NOT NULL DEFAULT 1,
        provenance TEXT NOT NULL DEFAULT 'user_taught',
        PRIMARY KEY (source_key, target_key, relation_type)
    );
    """

    def __init__(self, database: str | Path = ":memory:") -> None:
        self.database = str(database)
        if self.database != ":memory:":
            Path(self.database).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.database)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def counts(self) -> dict[str, int]:
        return {
            "lemmas": int(self.conn.execute("SELECT COUNT(*) FROM lemmas").fetchone()[0]),
            "concepts": int(self.conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]),
            "relations": int(self.conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]),
            "learned_associations": int(self.conn.execute("SELECT COUNT(*) FROM learned_associations").fetchone()[0]),
        }

    def _meta(self, values: dict[str, str]) -> None:
        self.conn.executemany(
            "INSERT INTO network_meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            list(values.items()),
        )

    @staticmethod
    def _open_xml(path: str | Path):
        name = str(path).lower()
        if name.endswith(".gz"):
            return gzip.open(path, "rb")
        if name.endswith(".zip"):
            archive = zipfile.ZipFile(path)
            matches = [item for item in archive.namelist() if item.lower().endswith((".xml", ".lmf"))]
            if not matches:
                archive.close()
                raise ValueError("The ZIP archive contains no XML/LMF dictionary file.")
            return _ZipStream(archive, archive.open(matches[0]))
        return open(path, "rb")

    def import_lmf(
        self,
        path: str | Path,
        dataset: str = "Open English WordNet 2025+",
        license_name: str = "CC BY 4.0",
        source_url: str = "https://github.com/globalwordnet/english-wordnet",
        batch_size: int = 2000,
    ) -> dict[str, int]:
        """Stream-import a Global WordNet LMF XML file (plain, gzip, or ZIP)."""
        source = str(path)
        imported_lemmas = 0
        imported_concepts = 0
        imported_relations = 0

        # Pass 1: lexical forms and their sense-to-synset mappings.
        stream = self._open_xml(path)
        try:
            stack: list[ET.Element] = []
            pending: list[tuple[str, str, str, str, str, str]] = []
            for event, elem in ET.iterparse(stream, events=("start", "end")):
                if event == "start":
                    stack.append(elem)
                    continue
                if _local(elem.tag) == "LexicalEntry":
                    lemma_elem = next(
                        (child for child in elem.iter() if _local(child.tag) == "Lemma"),
                        None,
                    )
                    if lemma_elem is not None:
                        lemma = lemma_elem.attrib.get("writtenForm", "").strip()
                        pos = elem.attrib.get("partOfSpeech", lemma_elem.attrib.get("partOfSpeech", ""))
                        entry_id = elem.attrib.get("id", "")
                        if lemma:
                            for sense in elem.iter():
                                if _local(sense.tag) != "Sense":
                                    continue
                                synset = sense.attrib.get("synset", "").strip()
                                sense_id = sense.attrib.get("id", "")
                                if synset:
                                    normalized = lemma.lower().replace("_", " ").strip()
                                    pending.append((normalized, lemma, pos, synset, sense_id, dataset))
                    parent = stack[-2] if len(stack) > 1 else None
                    if parent is not None:
                        parent.remove(elem)
                    elem.clear()
                    if len(pending) >= batch_size:
                        self.conn.executemany(
                            "INSERT OR IGNORE INTO lemmas "
                            "(lemma_norm,lemma,part_of_speech,synset_id,sense_id,dataset) "
                            "VALUES(?,?,?,?,?,?)",
                            pending,
                        )
                        imported_lemmas += len(pending)
                        pending.clear()
                stack.pop()
            if pending:
                self.conn.executemany(
                    "INSERT OR IGNORE INTO lemmas "
                    "(lemma_norm,lemma,part_of_speech,synset_id,sense_id,dataset) "
                    "VALUES(?,?,?,?,?,?)",
                    pending,
                )
                imported_lemmas += len(pending)
        finally:
            stream.close()

        # Pass 2: concept definitions, examples, and typed semantic edges.
        stream = self._open_xml(path)
        try:
            stack = []
            concepts_batch: list[tuple[str, str, str, str, str, str, str, str]] = []
            relations_batch: list[tuple[str, str, str, str]] = []
            for event, elem in ET.iterparse(stream, events=("start", "end")):
                if event == "start":
                    stack.append(elem)
                    continue
                if _local(elem.tag) == "Synset":
                    synset_id = elem.attrib.get("id", "").strip()
                    pos = elem.attrib.get("partOfSpeech", "")
                    ili = elem.attrib.get("ili", "")
                    definition = ""
                    examples: list[str] = []
                    for child in elem.iter():
                        tag = _local(child.tag)
                        if tag == "Definition" and not definition:
                            definition = "".join(child.itertext()).strip()
                        elif tag == "Example":
                            example = "".join(child.itertext()).strip()
                            if example:
                                examples.append(example)
                        elif tag == "SynsetRelation":
                            target = child.attrib.get("target", "").strip()
                            rel = child.attrib.get("relType", "other").strip()
                            if synset_id and target:
                                relations_batch.append((synset_id, target, rel, dataset))
                    if synset_id:
                        concepts_batch.append((
                            synset_id, pos, definition, json.dumps(examples, ensure_ascii=False),
                            ili, dataset, license_name, source_url,
                        ))
                        imported_concepts += 1
                    parent = stack[-2] if len(stack) > 1 else None
                    if parent is not None:
                        parent.remove(elem)
                    elem.clear()
                    if len(concepts_batch) >= batch_size:
                        self.conn.executemany(
                            "INSERT OR REPLACE INTO concepts "
                            "(synset_id,part_of_speech,definition,examples_json,ili,dataset,license,source_url) "
                            "VALUES(?,?,?,?,?,?,?,?)",
                            concepts_batch,
                        )
                        self.conn.executemany(
                            "INSERT OR IGNORE INTO relations "
                            "(source_synset,target_synset,relation_type,dataset) VALUES(?,?,?,?)",
                            relations_batch,
                        )
                        imported_relations += len(relations_batch)
                        concepts_batch.clear()
                        relations_batch.clear()
                stack.pop()
            if concepts_batch:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO concepts "
                    "(synset_id,part_of_speech,definition,examples_json,ili,dataset,license,source_url) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    concepts_batch,
                )
            if relations_batch:
                self.conn.executemany(
                    "INSERT OR IGNORE INTO relations "
                    "(source_synset,target_synset,relation_type,dataset) VALUES(?,?,?,?)",
                    relations_batch,
                )
                imported_relations += len(relations_batch)
        finally:
            stream.close()

        self._meta({
            "last_import_source": source,
            "last_import_dataset": dataset,
            "last_import_license": license_name,
            "last_import_source_url": source_url,
        })
        self.conn.commit()
        return {
            "lemma_links_imported": imported_lemmas,
            "concepts_imported": imported_concepts,
            "relations_imported": imported_relations,
            **self.counts(),
        }

    def import_jsonl(
        self,
        path: str | Path,
        dataset: str,
        license_name: str,
        source_url: str = "",
        batch_size: int = 1000,
    ) -> dict[str, int]:
        """Stream-import generic concept JSONL for larger custom dictionaries.

        One line per concept:
          {"id":"concept-id","pos":"n","definition":"...","examples":[],
           "lemmas":["word"],"relations":[{"type":"hypernym","target":"..."}]}
        """
        concepts: list[tuple[str, str, str, str, str, str, str, str]] = []
        lemmas: list[tuple[str, str, str, str, str, str]] = []
        relations: list[tuple[str, str, str, str]] = []
        counts = {"concepts_imported": 0, "lemma_links_imported": 0, "relations_imported": 0}
        with open(path, "r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"JSONL line {line_number} must contain an object.")
                synset_id = str(record.get("id") or "").strip()
                if not synset_id:
                    raise ValueError(f"JSONL line {line_number} is missing 'id'.")
                pos = str(record.get("pos") or "")
                definition = str(record.get("definition") or "")
                examples = record.get("examples") or []
                word_forms = record.get("lemmas") or []
                rels = record.get("relations") or []
                if not isinstance(examples, list) or not isinstance(word_forms, list) or not isinstance(rels, list):
                    raise ValueError(f"JSONL line {line_number} has invalid examples, lemmas, or relations.")
                concepts.append((
                    synset_id, pos, definition, json.dumps(examples, ensure_ascii=False),
                    str(record.get("ili") or ""), dataset, license_name, source_url,
                ))
                counts["concepts_imported"] += 1
                for lemma in word_forms:
                    if not isinstance(lemma, str) or not lemma.strip():
                        continue
                    normalized = lemma.lower().replace("_", " ").strip()
                    lemmas.append((normalized, lemma, pos, synset_id, "", dataset))
                    counts["lemma_links_imported"] += 1
                for relation in rels:
                    if not isinstance(relation, dict):
                        continue
                    target = str(relation.get("target") or "").strip()
                    rel_type = str(relation.get("type") or "other").strip()
                    if target:
                        relations.append((synset_id, target, rel_type, dataset))
                        counts["relations_imported"] += 1
                if len(concepts) >= batch_size:
                    self._flush_jsonl(concepts, lemmas, relations)
                    concepts.clear()
                    lemmas.clear()
                    relations.clear()
        if concepts or lemmas or relations:
            self._flush_jsonl(concepts, lemmas, relations)
        self._meta({
            "last_import_source": str(path),
            "last_import_dataset": dataset,
            "last_import_license": license_name,
            "last_import_source_url": source_url,
        })
        self.conn.commit()
        return {**counts, **self.counts()}

    def _flush_jsonl(self, concepts, lemmas, relations) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO concepts "
            "(synset_id,part_of_speech,definition,examples_json,ili,dataset,license,source_url) "
            "VALUES(?,?,?,?,?,?,?,?)",
            concepts,
        )
        if lemmas:
            self.conn.executemany(
                "INSERT OR IGNORE INTO lemmas "
                "(lemma_norm,lemma,part_of_speech,synset_id,sense_id,dataset) "
                "VALUES(?,?,?,?,?,?)",
                lemmas,
            )
        if relations:
            self.conn.executemany(
                "INSERT OR IGNORE INTO relations "
                "(source_synset,target_synset,relation_type,dataset) VALUES(?,?,?,?)",
                relations,
            )

    def lookup(self, lemma: str, limit: int = 16) -> list[dict[str, Any]]:
        normalized = lemma.lower().replace("_", " ").strip()
        rows = self.conn.execute(
            "SELECT l.lemma,l.part_of_speech,l.synset_id,l.sense_id,"
            "c.definition,c.examples_json "
            "FROM lemmas l LEFT JOIN concepts c ON c.synset_id=l.synset_id "
            "WHERE l.lemma_norm=? ORDER BY l.part_of_speech,l.synset_id LIMIT ?",
            (normalized, max(1, min(100, limit))),
        ).fetchall()
        return [dict(row) for row in rows]

    def _relation_support(self, left: str, right: str) -> float:
        if left == right:
            return 0.0
        rows = self.conn.execute(
            "SELECT relation_type FROM relations "
            "WHERE (source_synset=? AND target_synset=?) "
            "OR (source_synset=? AND target_synset=?) LIMIT 8",
            (left, right, right, left),
        ).fetchall()
        return max((RELATION_WEIGHTS.get(row[0], RELATION_WEIGHTS["other"]) for row in rows), default=0.0)

    @staticmethod
    def _score_lexical_context(definition: str, examples: Iterable[str], local_context: set[str], global_context: set[str], lemma_terms: set[str]) -> float:
        if not local_context and not global_context:
            return 0.20 if definition else 0.05
        evidence = _terms(definition)
        for example in examples:
            evidence.update(_terms(example))
        evidence.update(lemma_terms)
        local = len(evidence & local_context) / max(1, len(local_context))
        global_score = len(evidence & global_context) / max(1, len(global_context))
        density = len(evidence & (local_context | global_context)) / max(1, len(evidence))
        return min(1.0, 0.55 * local + 0.25 * global_score + 0.20 * density)

    def analyze(
        self,
        text: str,
        dialogue_context: str = "",
        top_k: int = 4,
    ) -> SemanticAnalysis:
        """Encode all words, resolve candidate senses, then activate relation support."""
        raw_tokens = TOKEN_RE.findall(text or "")
        words = [(i, token) for i, token in enumerate(raw_tokens) if any(ch.isalnum() for ch in token)]
        surfaces = [token.lower() for _, token in words]
        expanded: list[tuple[int, str]] = []
        token_forms: dict[int, tuple[str, ...]] = {}
        for index, surface in words:
            forms = CHAT_NORMALIZATION.get(surface, (surface,))
            normalized: list[str] = []
            for form in forms:
                variants = _lemma_variants(form)
                normalized.extend(variants)
                expanded.append((index, form))
            token_forms[index] = tuple(dict.fromkeys(normalized))

        current_content = set(_terms(" ".join(surfaces)))
        historical_content = set(_terms(dialogue_context))
        # Current-turn evidence dominates; dialogue context is a weaker signal.
        global_context = current_content | historical_content
        counts = self.counts()
        if counts["concepts"] == 0 or counts["lemmas"] == 0:
            activations = tuple(
                WordActivation(
                    position=index,
                    surface=surface,
                    normalized_forms=token_forms.get(index, (surface,)),
                    status="unknown",
                    selected_sense=None,
                    ambiguity="unresolved",
                    support_margin=0.0,
                    candidates=(),
                    subword_units=tuple(_ngrams(surface)),
                )
                for index, surface in words
            )
            return SemanticAnalysis(
                original=text,
                tokens=activations,
                activated_relations=(),
                dictionary_ready=False,
                dictionary_counts=counts,
            )

        provisional: dict[int, list[dict[str, Any]]] = {}
        for word_index, surface in words:
            forms = token_forms.get(word_index, (surface,))
            rows_by_synset: dict[str, dict[str, Any]] = {}
            for form in forms:
                for row in self.lookup(form, limit=64):
                    synset_id = row["synset_id"]
                    existing = rows_by_synset.get(synset_id)
                    if existing is None:
                        rows_by_synset[synset_id] = row
                    elif form == surface:
                        rows_by_synset[synset_id] = row
            if not rows_by_synset:
                provisional[word_index] = []
                continue

            nearby_words = [
                value for j, value in words
                if j != word_index and abs(j - word_index) <= 5
                and value.lower() not in STOPWORDS and any(c.isalpha() for c in value)
            ]
            local_context = set(_terms(" ".join(nearby_words)))
            candidates: list[dict[str, Any]] = []
            for row in rows_by_synset.values():
                try:
                    examples = json.loads(row.get("examples_json") or "[]")
                except json.JSONDecodeError:
                    examples = []
                lemma_terms = _terms(row.get("lemma", ""))
                lexical = self._score_lexical_context(
                    row.get("definition") or "", examples, local_context,
                    global_context, lemma_terms,
                )
                candidates.append({
                    **row,
                    "examples": tuple(examples) if isinstance(examples, list) else (),
                    "lexical_support": lexical,
                    "relation_support": 0.0,
                })
            provisional[word_index] = candidates

        # One relational-consistency pass. It checks whether a candidate sense
        # has a typed relation to plausible senses of other words in this utterance.
        for word_index, candidates in provisional.items():
            for candidate in candidates:
                relation_score = 0.0
                for other_index, other_candidates in provisional.items():
                    if other_index == word_index:
                        continue
                    best_link = 0.0
                    for other in other_candidates[:16]:
                        best_link = max(
                            best_link,
                            self._relation_support(candidate["synset_id"], other["synset_id"]),
                        )
                    relation_score = max(relation_score, best_link)
                candidate["relation_support"] = relation_score

        results: list[WordActivation] = []
        relation_edges: list[dict[str, Any]] = []
        by_position: dict[int, WordActivation] = {}
        for word_index, surface in words:
            candidates = provisional.get(word_index, [])
            forms = token_forms.get(word_index, (surface,))
            if not candidates:
                activation = WordActivation(
                    position=word_index,
                    surface=surface,
                    normalized_forms=forms,
                    status="unknown",
                    selected_sense=None,
                    ambiguity="unresolved",
                    support_margin=0.0,
                    candidates=(),
                    subword_units=tuple(_ngrams(surface)),
                )
            else:
                for item in candidates:
                    item["total_support"] = (
                        0.72 * item["lexical_support"] +
                        0.28 * item["relation_support"]
                    )
                candidates.sort(
                    key=lambda item: (item["total_support"], item["lexical_support"]),
                    reverse=True,
                )
                top = candidates[:max(1, min(8, top_k))]
                best_score = float(top[0]["total_support"])
                second_score = float(top[1]["total_support"]) if len(top) > 1 else 0.0
                margin = max(0.0, best_score - second_score)
                is_resolved = (
                    len(candidates) == 1 or
                    (best_score >= 0.10 and margin >= 0.12)
                )
                selected = top[0]["sense_id"] or top[0]["synset_id"] if is_resolved else None
                sense_objects = tuple(
                    SenseCandidate(
                        synset_id=item["synset_id"],
                        sense_id=item.get("sense_id") or "",
                        lemma=item.get("lemma") or surface,
                        part_of_speech=item.get("part_of_speech") or "",
                        definition=item.get("definition") or "",
                        examples=tuple(item.get("examples") or ()),
                        lexical_support=float(item.get("lexical_support", 0.0)),
                        relation_support=float(item.get("relation_support", 0.0)),
                        total_support=float(item.get("total_support", 0.0)),
                    )
                    for item in top
                )
                activation = WordActivation(
                    position=word_index,
                    surface=surface,
                    normalized_forms=forms,
                    status="resolved" if is_resolved else "ambiguous",
                    selected_sense=selected,
                    ambiguity="none" if is_resolved else "multiple compatible senses",
                    support_margin=margin,
                    candidates=sense_objects,
                    subword_units=(),
                )
            results.append(activation)
            by_position[word_index] = activation

        # Multiword lexical entries are first-class patterns. They are represented
        # separately from component words so a phrase sense is never confused with
        # the independent senses of each token.
        phrase_activations: list[dict[str, Any]] = []
        for start in range(len(words)):
            for width in range(2, 6):
                chunk = words[start:start + width]
                if len(chunk) != width:
                    continue
                positions = [item[0] for item in chunk]
                if any(positions[i] + 1 != positions[i + 1] for i in range(len(positions) - 1)):
                    continue
                phrase = " ".join(item[1].lower() for item in chunk)
                phrase_rows = self.lookup(phrase, limit=24)
                if not phrase_rows:
                    continue
                covered = set(positions)
                outside = [
                    value for pos, value in words
                    if pos not in covered and value.lower() not in STOPWORDS
                ]
                local_context = set(_terms(" ".join(outside)))
                phrase_candidates = []
                for row in phrase_rows:
                    try:
                        examples = json.loads(row.get("examples_json") or "[]")
                    except json.JSONDecodeError:
                        examples = []
                    support = self._score_lexical_context(
                        row.get("definition") or "", examples, local_context,
                        global_context, _terms(row.get("lemma") or phrase),
                    )
                    phrase_candidates.append({
                        "synset_id": row["synset_id"],
                        "sense_id": row.get("sense_id") or "",
                        "lemma": row.get("lemma") or phrase,
                        "pos": row.get("part_of_speech") or "",
                        "definition": row.get("definition") or "",
                        "support": round(support, 5),
                    })
                phrase_candidates.sort(key=lambda item: item["support"], reverse=True)
                phrase_activations.append({
                    "phrase": phrase,
                    "start_position": positions[0],
                    "end_position": positions[-1],
                    "candidates": phrase_candidates[:4],
                    "note": "multiword lexical concept; component words remain separately encoded",
                })

        # Expose only relations between the strongest candidates. This is evidence,
        # not an assertion that the words form a grammatical relation in the sentence.
        for left_index, left in provisional.items():
            for right_index, right in provisional.items():
                if right_index <= left_index:
                    continue
                if not left or not right:
                    continue
                a, b = left[0], right[0]
                score = self._relation_support(a["synset_id"], b["synset_id"])
                if score > 0:
                    relation_edges.append({
                        "left_position": left_index,
                        "left_word": dict(words)[left_index],
                        "right_position": right_index,
                        "right_word": dict(words)[right_index],
                        "relation_support": round(score, 5),
                        "note": "lexical graph relation; not a sentence-role assertion",
                    })
        relation_edges.sort(key=lambda item: item["relation_support"], reverse=True)

        return SemanticAnalysis(
            original=text,
            tokens=tuple(results),
            activated_relations=tuple(relation_edges[:64]),
            dictionary_ready=True,
            dictionary_counts=counts,
            phrase_activations=tuple(phrase_activations[:64]),
        )

    def close(self) -> None:
        self.conn.close()


class _ZipStream:
    """Keep a ZipFile alive for the lifetime of its member stream."""

    def __init__(self, archive: zipfile.ZipFile, stream) -> None:
        self.archive = archive
        self.stream = stream

    def read(self, *args):
        return self.stream.read(*args)

    def close(self) -> None:
        try:
            self.stream.close()
        finally:
            self.archive.close()
