"""Compose casual, uncertainty-aware answers from sourced atomic facts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TopicAnswer:
    text: str
    topic: str
    known: bool
    selected_facts: tuple[dict, ...]
    sources: tuple[dict, ...]
    support_score: float

    def diagnostics(self) -> dict:
        return {
            "topic": self.topic,
            "known": self.known,
            "selected_facts": [
                {
                    "id": fact["fact_id"],
                    "predicate": fact["predicate"],
                    "object": fact["object"],
                    "confidence": fact["confidence"],
                    "epistemic_status": fact["epistemic_status"],
                    "source": fact["source"],
                    "source_url": fact["source_url"],
                }
                for fact in self.selected_facts
            ],
            "sources": list(self.sources),
            "support_score": round(self.support_score, 4),
        }


def _display_topic(topic: str) -> str:
    lowered = topic.lower().strip()
    if lowered in {"moon", "lunar"}:
        return "the Moon"
    if lowered in {"earth"}:
        return "Earth"
    if lowered in {"sun"}:
        return "the Sun"
    return topic.strip().title() if topic.strip() else "that topic"


def _select_facts(facts: list[dict], query_terms: set[str]) -> list[dict]:
    by_predicate = {fact["predicate"]: fact for fact in facts}

    if query_terms & {"color", "colour", "white", "gray", "grey", "appearance", "look"}:
        wanted = ("apparent_color_from_earth", "surface_color_up_close")
    elif query_terms & {"light", "shine", "sunlight", "glow", "bright"}:
        wanted = ("light_source", "apparent_color_from_earth")
    elif query_terms & {"size", "big", "large", "wide", "diameter", "small"}:
        wanted = ("diameter_relative_to_earth",)
    else:
        wanted = (
            "apparent_color_from_earth",
            "surface_color_up_close",
            "light_source",
            "natural_satellite_of",
        )

    selected = [by_predicate[key] for key in wanted if key in by_predicate]
    if not selected:
        selected = facts[:2]
    return selected[:3]


def _render_fact(fact: dict, topic: str) -> str:
    predicate = fact["predicate"]
    if predicate == "apparent_color_from_earth":
        return "it often looks bright, almost white, from Earth"
    if predicate == "surface_color_up_close":
        return "up close, much of its surface is made of dark-gray shades"
    if predicate == "light_source":
        return "it shines by reflecting sunlight instead of making its own light"
    if predicate == "natural_satellite_of":
        return "it is Earth's natural satellite and orbits our planet"
    if predicate == "diameter_relative_to_earth":
        return "it is about one-quarter as wide as Earth"
    return fact["object"].rstrip(". ")


def compose_topic_answer(
    topic: str,
    facts: Iterable[dict],
    query_terms: Iterable[str] = (),
) -> TopicAnswer:
    """Build wording from fact predicates and epistemic metadata.

    The lead-in truthfully describes the bounded local fact store; every content
    clause is selected from a source-attributed fact. Unknown topics do not get
    a fabricated filler fact.
    """
    fact_list = list(facts)
    terms = {str(item).lower() for item in query_terms}
    display = _display_topic(topic)

    if not fact_list:
        text = (
            f"Hmm, I don't have reliable facts about {display} in my local knowledge yet, "
            "so I don't want to make something up. LOL."
        )
        return TopicAnswer(
            text=text,
            topic=display,
            known=False,
            selected_facts=(),
            sources=(),
            support_score=0.0,
        )

    selected = _select_facts(fact_list, terms)
    clauses = [_render_fact(fact, display) for fact in selected]
    if len(clauses) == 1:
        body = clauses[0].rstrip(".")
    elif len(clauses) == 2:
        body = clauses[0].rstrip(".") + ", and " + clauses[1]
    else:
        body = clauses[0].rstrip(".") + "; " + clauses[1].rstrip(".") + "; and " + clauses[2].rstrip(".")
    if body:
        body = body[0].upper() + body[1:]

    # This is a transparent statement about the current curated index, not a
    # claim that the system has exhausted the subject or reached full expertise.
    text = (
        f"Hmm, I only have a few sourced basics about {display} so far, "
        f"but here's what I do know: {body}. LOL."
    )
    sources_by_url: dict[str, dict] = {}
    for fact in selected:
        sources_by_url[fact["source_url"]] = {
            "name": fact["source"],
            "url": fact["source_url"],
        }
    support = min(float(fact["confidence"]) for fact in selected)
    return TopicAnswer(
        text=text,
        topic=display,
        known=True,
        selected_facts=tuple(selected),
        sources=tuple(sources_by_url.values()),
        support_score=support,
    )
