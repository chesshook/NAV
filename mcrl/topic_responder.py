"""Topic-neutral answer framing from optional, source-attributed fact records."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


PROPERTY_PREDICATES = {
    "distance": {"distance", "mean_distance", "distance_to", "distance_to_earth"},
    "color": {"color", "colour", "apparent_color", "apparent_color_from_earth", "surface_color", "surface_color_up_close"},
    "size": {"size", "diameter", "diameter_relative_to_earth", "radius", "mass"},
    "population": {"population"},
    "capital": {"capital"},
    "definition": {"definition", "meaning"},
}


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
                    "subject": fact["subject"],
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
    cleaned = " ".join(str(topic or "").split()).strip()
    if not cleaned:
        return "that topic"
    return cleaned[0].upper() + cleaned[1:]


def _select_facts(
    facts: list[dict],
    requested_property: str | None,
    limit: int = 3,
) -> list[dict]:
    if requested_property:
        accepted = PROPERTY_PREDICATES.get(requested_property)
        if accepted is None:
            # The property is understood linguistically, but this prototype has
            # no safe predicate mapping for it. Never substitute unrelated facts.
            return []
        matching = [fact for fact in facts if fact["predicate"].lower() in accepted]
        return matching[:limit]
    return sorted(
        facts,
        key=lambda fact: (
            int(fact.get("priority", 50)),
            -float(fact.get("confidence", 0.0)),
            fact.get("fact_id", ""),
        ),
    )[:limit]


def _fact_clause(fact: dict) -> str:
    statement = str(fact.get("object", "")).strip()
    if not statement:
        return ""
    return statement if statement.endswith((".", "!", "?")) else statement + "."


def compose_topic_answer(
    topic: str,
    facts: Iterable[dict],
    query_terms: Iterable[str] = (),
    requested_property: str | None = None,
) -> TopicAnswer:
    """Compose a natural, topic-specific answer without fabricating missing claims.

    A fact is used only when it is explicitly supplied, attributed, and (when the
    question names a property) matches that property. No default domain facts are
    introduced by this responder.
    """
    fact_list = list(facts)
    display = _display_topic(topic)
    selected = _select_facts(fact_list, requested_property)

    if not selected:
        if requested_property:
            text = (
                f"Hmm, I don't have a reliable {requested_property} fact for {display} yet. "
                "I don't want to bluff or swap in an unrelated fact. If you have a source, "
                "we can add it; otherwise I should leave that detail uncertain."
            )
        else:
            text = (
                f"Hmm, I don't know enough about {display} to give you a reliable answer yet. "
                "I don't want to bluff or make something up. Ask me about a specific angle "
                "and I'll tell you what I can support."
            )
        return TopicAnswer(
            text=text,
            topic=display,
            known=False,
            selected_facts=(),
            sources=(),
            support_score=0.0,
        )

    clauses = [_fact_clause(fact) for fact in selected]
    clauses = [clause for clause in clauses if clause]
    if not clauses:
        return compose_topic_answer(topic, (), query_terms, requested_property)

    if len(clauses) == 1:
        body = clauses[0]
        lead = f"Hmm, I have one sourced note about {display}: "
    elif len(clauses) == 2:
        body = " ".join(clauses)
        lead = f"Hmm, I have a couple of sourced notes about {display}: "
    else:
        body = " ".join(clauses)
        lead = f"Hmm, I have a few sourced notes about {display}: "

    statuses = {fact.get("epistemic_status", "established") for fact in selected}
    if "tentative" in statuses:
        lead = f"Hmm, I have a tentative note about {display}: "

    text = lead + body + " That's what I can support from my current notes."
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
