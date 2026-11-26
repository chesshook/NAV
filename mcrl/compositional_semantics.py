"""Compose dependency structures into an inspectable utterance meaning field."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .semantic_network import SemanticAnalysis
from .syntax_model import ParseResult, ParseToken


WH_WORDS = {"who", "whom", "whose", "what", "which", "when", "where", "why", "how"}
FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
SECOND_PERSON = {"you", "your", "yours"}
NEGATIVE_WORDS = {"not", "n't", "never", "no", "neither", "nor", "without"}
TEMPORAL_CASES = {"before", "after", "during", "until", "since", "while"}
LOCATION_CASES = {"in", "at", "on", "under", "above", "near", "inside", "outside", "into", "onto", "through"}
SOURCE_CASES = {"from", "out", "off"}
GOAL_CASES = {"to", "toward", "towards", "into", "onto"}
INSTRUMENT_CASES = {"with", "using", "via", "by"}
CAUSE_CASES = {"because", "due", "since", "because_of"}
CONDITION_MARKERS = {"if", "unless", "provided", "assuming"}
COORDINATORS = {"and", "or", "but", "nor", "yet", "so"}


def _base_relation(value: str) -> str:
    return value.split(":", 1)[0]


def _feature_map(value: str) -> dict[str, str]:
    result = {}
    for part in (value or "").split("|"):
        if "=" in part:
            key, val = part.split("=", 1)
            result[key] = val
    return result


def _lexical_by_form(analysis: SemanticAnalysis | None) -> dict[str, list[dict[str, Any]]]:
    mapped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if analysis is None:
        return mapped
    for activation in analysis.tokens:
        data = activation.to_dict()
        mapped[activation.surface.lower()].append(data)
        for form in activation.normalized_forms:
            mapped[form.lower()].append(data)
    return mapped


def _token_lexical_data(token: ParseToken, mapped: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    candidates = mapped.get(token.form.lower()) or mapped.get(token.lemma.lower()) or []
    if not candidates:
        return {
            "lexical_status": "unresolved",
            "selected_sense": None,
            "candidate_senses": [],
            "subword_units": [],
        }
    best = candidates[0]
    return {
        "lexical_status": best.get("status", "unknown"),
        "selected_sense": best.get("selected_sense"),
        "candidate_senses": [
            {
                "synset_id": item.get("synset_id"),
                "sense_id": item.get("sense_id"),
                "lemma": item.get("lemma"),
                "pos": item.get("part_of_speech"),
                "definition": item.get("definition"),
                "support": item.get("total_support"),
            }
            for item in best.get("candidates", [])
        ],
        "subword_units": best.get("subword_units", []),
    }


def _case_child(token: ParseToken, children: Mapping[int, list[ParseToken]]) -> str:
    cases = [
        child.lemma.lower()
        for child in children.get(token.id, [])
        if _base_relation(child.deprel) == "case"
    ]
    return cases[0] if cases else ""


def _case_semantics(case: str) -> str:
    if case in TEMPORAL_CASES:
        return "temporal_relation_candidate"
    if case in LOCATION_CASES:
        return "location_or_path_candidate"
    if case in SOURCE_CASES:
        return "source_relation_candidate"
    if case in GOAL_CASES:
        return "goal_or_recipient_candidate"
    if case in INSTRUMENT_CASES:
        return "instrument_or_accompaniment_candidate"
    if case in CAUSE_CASES:
        return "cause_or_reason_candidate"
    if case in CONDITION_MARKERS:
        return "condition_candidate"
    return "oblique_relation_unspecified"


def _participant(token: ParseToken) -> dict[str, Any] | None:
    value = token.lemma.lower()
    if value in FIRST_PERSON:
        return {
            "participant": "discourse_speaker",
            "surface": token.form,
            "mapping": "first-person grammatical reference",
        }
    if value in SECOND_PERSON:
        return {
            "participant": "discourse_addressee",
            "surface": token.form,
            "mapping": "second-person grammatical reference",
        }
    return None


def compose_meaning(
    text: str,
    parsed: ParseResult,
    lexical: SemanticAnalysis | None = None,
    history: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    """Create an open-ended meaning graph from syntax and lexical-sense evidence."""
    tokens = list(parsed.tokens)
    children: dict[int, list[ParseToken]] = defaultdict(list)
    by_id = {token.id: token for token in tokens}
    for token in tokens:
        if token.head in by_id:
            children[token.head].append(token)

    lexical_map = _lexical_by_form(lexical)
    nodes: list[dict[str, Any]] = []
    unresolved: list[str] = []
    ambiguous: list[str] = []
    for token in tokens:
        lex = _token_lexical_data(token, lexical_map)
        nodes.append({
            "id": token.id,
            "surface": token.form,
            "lemma": token.lemma,
            "upos": token.upos,
            "morphology": _feature_map(token.feats),
            "lexical": lex,
        })
        if lex["lexical_status"] in {"unknown", "unresolved"} and token.upos != "PUNCT":
            unresolved.append(token.form)
        if lex["lexical_status"] == "ambiguous":
            ambiguous.append(token.form)

    edges: list[dict[str, Any]] = []
    for token in tokens:
        head = by_id.get(token.head)
        edges.append({
            "source": "ROOT" if head is None else head.id,
            "target": token.id,
            "relation": token.deprel,
            "source_word": "ROOT" if head is None else head.form,
            "target_word": token.form,
        })

    roots = [token for token in tokens if token.head == 0]
    primary_root = roots[0] if roots else None
    clause_roots = [
        token for token in tokens
        if token.upos in {"VERB", "AUX"} and (
            token.head == 0 or _base_relation(token.deprel) in {"conj", "ccomp", "xcomp", "advcl", "acl"}
        )
    ]
    if primary_root and primary_root not in clause_roots and primary_root.upos in {"ADJ", "NOUN", "PROPN"}:
        clause_roots.insert(0, primary_root)

    clauses: list[dict[str, Any]] = []
    for predicate in clause_roots or ([primary_root] if primary_root else []):
        assert predicate is not None
        dependents = children.get(predicate.id, [])
        roles = []
        negated = False
        auxiliaries = []
        for dependent in dependents:
            relation = dependent.deprel
            base = _base_relation(relation)
            role = None
            if base in {"nsubj", "csubj"}:
                role = (
                    "patient_candidate"
                    if "pass" in relation
                    else "grammatical_subject"
                )
                participant = _participant(dependent)
                roles.append({
                    "role": role,
                    "token_id": dependent.id,
                    "value": dependent.lemma,
                    "participant": participant,
                    "confidence_note": (
                        "syntactic subject; thematic role requires verb-frame grounding"
                        if role == "grammatical_subject"
                        else "passive subject is a patient candidate"
                    ),
                })
            elif base in {"obj", "iobj"}:
                roles.append({
                    "role": "direct_object_or_patient_candidate" if base == "obj" else "indirect_object_or_recipient_candidate",
                    "token_id": dependent.id,
                    "value": dependent.lemma,
                    "participant": _participant(dependent),
                })
            elif base in {"obl", "nmod"}:
                case = _case_child(dependent, children)
                roles.append({
                    "role": _case_semantics(case),
                    "token_id": dependent.id,
                    "value": dependent.lemma,
                    "case_marker": case or None,
                })
            elif base == "neg" or dependent.lemma.lower() in NEGATIVE_WORDS:
                negated = True
                roles.append({"role": "negation", "token_id": dependent.id, "value": dependent.lemma})
            elif base in {"aux", "cop"}:
                auxiliaries.append(dependent.lemma.lower())
                roles.append({
                    "role": "auxiliary_or_copula",
                    "token_id": dependent.id,
                    "value": dependent.lemma,
                })
            elif base in {"advmod", "amod", "nmod", "compound", "det", "nummod"}:
                roles.append({
                    "role": "modifier_or_determiner",
                    "token_id": dependent.id,
                    "value": dependent.lemma,
                    "dependency": relation,
                })
            elif base in {"ccomp", "xcomp", "advcl", "acl", "conj"}:
                roles.append({
                    "role": "embedded_or_coordinated_clause",
                    "token_id": dependent.id,
                    "value": dependent.lemma,
                    "dependency": relation,
                })
            elif base == "mark":
                roles.append({
                    "role": "clause_marker",
                    "token_id": dependent.id,
                    "value": dependent.lemma,
                })
        clauses.append({
            "predicate": predicate.lemma,
            "predicate_token_id": predicate.id,
            "predicate_upos": predicate.upos,
            "syntactic_relation": predicate.deprel,
            "roles": roles,
            "polarity": "negative" if negated else "not_explicitly_negated",
            "auxiliaries": auxiliaries,
            "morphology": _feature_map(predicate.feats),
        })

    punctuation = {token.form for token in tokens if token.upos == "PUNCT"}
    wh_tokens = [
        token for token in tokens
        if token.lemma.lower() in WH_WORDS and token.upos in {"PRON", "DET", "ADV"}
    ]
    mood = _feature_map(primary_root.feats).get("Mood", "") if primary_root else ""
    if "?" in punctuation or wh_tokens:
        communicative_function = "interrogative"
        if wh_tokens:
            question_type = "wh_question"
            interrogative_focus = [token.lemma.lower() for token in wh_tokens]
        else:
            question_type = "polar_or_yes_no_question"
            interrogative_focus = []
    elif mood == "Imp" or (
        primary_root is not None
        and primary_root.upos == "VERB"
        and not any(_base_relation(dep.deprel) in {"nsubj", "csubj"} for dep in children.get(primary_root.id, []))
    ):
        communicative_function = "directive_or_imperative_candidate"
        question_type = None
        interrogative_focus = []
    elif "!" in punctuation:
        communicative_function = "expressive_or_emphatic"
        question_type = None
        interrogative_focus = []
    elif primary_root is not None and len(tokens) <= 2 and primary_root.upos in {"ADV", "ADJ", "INTJ"}:
        communicative_function = "fragment_or_elliptical_follow_up"
        question_type = None
        interrogative_focus = []
    elif primary_root is not None:
        communicative_function = "declarative_or_assertive_candidate"
        question_type = None
        interrogative_focus = []
    else:
        communicative_function = "unresolved_fragment"
        question_type = None
        interrogative_focus = []

    modal_auxiliaries = {
        token.lemma.lower()
        for token in tokens
        if token.upos == "AUX" and token.lemma.lower() in {
            "can", "could", "may", "might", "must", "shall", "should", "will", "would"
        }
    }
    modality = sorted(modal_auxiliaries)
    if wh_tokens:
        question_focus = " ".join(token.lemma.lower() for token in wh_tokens)
    elif communicative_function == "interrogative":
        question_focus = "truth_or_polarity_of_proposition"
    else:
        question_focus = None

    discourse_reference_candidates = []
    if history and primary_root and len(tokens) <= 3:
        discourse_reference_candidates = [
            {
                "turn_index": len(history),
                "reference": "previous_turn_candidate",
                "reason": "short or elliptical utterance depends on prior discourse",
            }
        ]

    temporal_features = []
    for token in tokens:
        if token.lemma.lower() in {"past", "present", "future", "yesterday", "today", "tomorrow", "now"}:
            temporal_features.append({"token_id": token.id, "value": token.lemma.lower()})
    participant_mentions = [
        item for item in (_participant(token) for token in tokens) if item is not None
    ]

    parse_complete = all(token.deprel != "unparsed" for token in tokens)
    ambiguity_notes = []
    if not parse_complete:
        ambiguity_notes.append("dependency parser did not attach every token")
    if ambiguous:
        ambiguity_notes.append("one or more words have multiple compatible dictionary senses")
    if unresolved:
        ambiguity_notes.append("one or more words are absent from the imported lexical dictionary")
    if discourse_reference_candidates:
        ambiguity_notes.append("short utterance may require a prior-turn antecedent")
    if any("grammatical subject" in role.get("confidence_note", "") for clause in clauses for role in clause["roles"]):
        ambiguity_notes.append("syntactic subject does not by itself determine semantic agent or experiencer")

    meaning_summary = _render_structural_summary(
        communicative_function, question_type, clauses, modality, unresolved, ambiguous
    )
    return {
        "text": text,
        "architecture": "mathematical lexical network + learned sparse syntactic parser + compositional semantic graph",
        "language_model_used_for_understanding": False,
        "communicative_function": communicative_function,
        "question_type": question_type,
        "interrogative_focus": interrogative_focus,
        "question_focus": question_focus,
        "modality": modality,
        "participant_mentions": participant_mentions,
        "clauses": clauses,
        "temporal_features": temporal_features,
        "discourse_reference_candidates": discourse_reference_candidates,
        "word_nodes": nodes,
        "dependency_edges": edges,
        "activated_lexical_relations": list(lexical.activated_relations) if lexical else [],
        "phrase_concepts": list(lexical.phrase_activations) if lexical else [],
        "unknown_lexical_items": list(dict.fromkeys(unresolved)),
        "ambiguous_lexical_items": list(dict.fromkeys(ambiguous)),
        "parse_complete": parse_complete,
        "ambiguity_notes": list(dict.fromkeys(ambiguity_notes)),
        "meaning_summary": meaning_summary,
        "score_semantics": "structural evidence and parser decisions; not calibrated probabilities",
    }


def _render_structural_summary(
    communicative_function: str,
    question_type: str | None,
    clauses: Sequence[dict[str, Any]],
    modality: Sequence[str],
    unknown: Sequence[str],
    ambiguous: Sequence[str],
) -> str:
    """A general graph readout—not a canned answer to the utterance."""
    pieces = [f"communicative structure: {communicative_function.replace('_', ' ')}"]
    if question_type:
        pieces.append(f"question structure: {question_type.replace('_', ' ')}")
    if clauses:
        main = clauses[0]
        pieces.append(f"main predicate: {main['predicate']} ({main['predicate_upos']})")
        for role in main["roles"]:
            if role["role"] in {
                "grammatical_subject", "patient_candidate",
                "direct_object_or_patient_candidate", "indirect_object_or_recipient_candidate",
                "location_or_path_candidate", "temporal_relation_candidate",
                "instrument_or_accompaniment_candidate", "cause_or_reason_candidate",
                "condition_candidate", "source_relation_candidate", "goal_or_recipient_candidate",
            }:
                pieces.append(f"{role['role']}: {role['value']}")
        if main["polarity"] == "negative":
            pieces.append("proposition is explicitly negated")
        if main["auxiliaries"]:
            pieces.append("auxiliaries: " + ", ".join(main["auxiliaries"]))
    if modality:
        pieces.append("modality: " + ", ".join(modality))
    if unknown:
        pieces.append("dictionary gaps: " + ", ".join(dict.fromkeys(unknown)))
    if ambiguous:
        pieces.append("sense ambiguity: " + ", ".join(dict.fromkeys(ambiguous)))
    return "; ".join(pieces) + "."
