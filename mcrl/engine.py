"""MCR-L orchestration: parse -> activate -> retrieve -> plan -> verify -> realize."""
from __future__ import annotations

from dataclasses import dataclass, field
from .cognition import MCRState
from .knowledge import KnowledgeStore
from .planner import CandidatePlan, build_plans, score_plan
from .realizer import realize
from .semantic import parse_semantics, with_evidence


@dataclass
class Answer:
    text: str
    confidence: float
    abstained: bool
    selected_title: str = ""
    source: str = ""
    diagnostics: dict = field(default_factory=dict)


class MCRLanguageEngine:
    def __init__(self, store: KnowledgeStore, minimum_score: float = 0.34) -> None:
        self.store = store
        self.minimum_score = minimum_score
        self.state = MCRState()
        self.last_plan: CandidatePlan | None = None

    def answer(self, question: str) -> Answer:
        frame = parse_semantics(question)
        records = self.store.search(" ".join(frame.content_tokens), limit=40)

        # Estimate evidence from lexical retrieval, then update the recurrent
        # state. Evidence is not fabricated from the query itself.
        query_terms = set(frame.content_tokens)
        best_coverage = 0.0
        for record in records:
            record_terms = set(
                (record.get("title", "") + " " + record.get("text", "") + " " +
                 record.get("answer", "") + " " + " ".join(record.get("keywords", []))).lower().split()
            )
            best_coverage = max(best_coverage, len(query_terms & record_terms) / max(1, len(query_terms)))
        frame = with_evidence(frame, best_coverage)
        activation = self.state.step(frame.features)

        if not records or not query_terms:
            self.last_plan = None
            return Answer(
                text="I don't have enough indexed evidence to answer that reliably. Add a vetted knowledge record or ask a more specific question.",
                confidence=0.0,
                abstained=True,
                diagnostics={"intent": frame.intent, "activation": self.state.snapshot()},
            )

        plans = build_plans(frame, records, activation)
        best = plans[0] if plans else None
        if best is None or best.score < self.minimum_score or not best.matched_terms or not best.record.get("source"):
            self.last_plan = None
            return Answer(
                text="I found no sufficiently supported evidence in the local knowledge index. I will not guess; add a relevant sourced record or provide more context.",
                confidence=best.score if best else 0.0,
                abstained=True,
                diagnostics={
                    "intent": frame.intent,
                    "top_candidates": [p.diagnostics() for p in plans[:5]],
                    "activation": self.state.snapshot(),
                },
            )

        self.last_plan = best
        def objective(candidate_activation: tuple[float, ...]) -> float:
            return score_plan(frame, best.record, candidate_activation).score

        sensitivities = self.state.sensitivity(objective)
        text = realize(best)
        return Answer(
            text=text,
            confidence=best.score,
            abstained=False,
            selected_title=best.record["title"],
            source=best.record["source"],
            diagnostics={
                "intent": frame.intent,
                "plan": best.diagnostics(),
                "sensitivity": {k: round(v, 4) for k, v in sensitivities.items() if v > 0},
                "activation": self.state.snapshot(),
                "retrieved_candidates": len(records),
            },
        )

    def observe_feedback(self, positive: bool) -> None:
        """Update recurrent links after explicit human feedback."""
        self.state.observe_feedback(positive)
