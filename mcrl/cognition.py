"""Recurrent situation state inspired by SCENT-X MCR-2."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from .semantic import FEATURE_NAMES

N = len(FEATURE_NAMES)

# Signed links encode explicit hypotheses about signal interaction.
# These are hand-authored starter weights, not learned facts.
_LINKS = (
    ("define", "entity", 0.42),
    ("explain", "causality", 0.35),
    ("causality", "explain", 0.28),
    ("compare", "constraint", 0.22),
    ("how_to", "action", 0.40),
    ("risk", "uncertainty", 0.45),
    ("uncertainty", "evidence", 0.48),
    ("evidence", "uncertainty", -0.12),
    ("negation", "constraint", 0.25),
    ("temporal", "context", 0.24),
    ("follow_up", "context", 0.52),
    ("question", "entity", 0.18),
    ("question", "evidence", 0.16),
    ("action", "how_to", 0.20),
    ("risk", "evidence", 0.34),
)


def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class MCRState:
    """A small recurrent state with bounded, inspectable connection weights."""

    retention: float = 0.56
    input_gain: float = 3.45
    recurrence_gain: float = 1.15
    bias: float = -1.45
    activation: list[float] = field(default_factory=lambda: [0.0] * N)
    weights: list[list[float]] = field(default_factory=lambda: [[0.0] * N for _ in range(N)])
    previous_activation: list[float] = field(default_factory=lambda: [0.0] * N)
    steps: int = 0

    def __post_init__(self) -> None:
        if not any(any(row) for row in self.weights):
            self._install_links()

    def _install_links(self) -> None:
        indices = {name: i for i, name in enumerate(FEATURE_NAMES)}
        for source, target, weight in _LINKS:
            self.weights[indices[source]][indices[target]] = weight

    def step(self, features: tuple[float, ...] | list[float]) -> tuple[float, ...]:
        if len(features) != N:
            raise ValueError(f"Expected {N} semantic features, got {len(features)}")
        old = self.activation[:]
        next_state: list[float] = [0.0] * N
        for target in range(N):
            recurrence = sum(old[source] * self.weights[source][target] for source in range(N))
            driven = _sigmoid(
                self.bias
                + self.input_gain * max(0.0, min(1.0, float(features[target])))
                + self.recurrence_gain * recurrence
            )
            next_state[target] = self.retention * old[target] + (1.0 - self.retention) * driven
        self.previous_activation = old
        self.activation = [max(0.0, min(1.0, v)) for v in next_state]
        self.steps += 1
        return tuple(self.activation)

    def observe_feedback(self, positive: bool, learning_rate: float = 0.002) -> None:
        """Apply tiny bounded Hebbian-style updates only after explicit feedback.

        This is a toy online adaptation rule, not a validated learning algorithm.
        """
        reward = 1.0 if positive else -1.0
        for i in range(N):
            for j in range(N):
                old_weight = self.weights[i][j]
                if old_weight == 0.0:
                    continue
                correlation = (self.previous_activation[i] - 0.38) * (self.activation[j] - 0.38)
                delta = learning_rate * reward * correlation - abs(old_weight) * 0.0002
                self.weights[i][j] = max(-0.48, min(1.18, old_weight + delta))

    def sensitivity(self, score_function) -> dict[str, float]:
        """Finite-difference influence of each activation on a supplied score."""
        baseline = float(score_function(tuple(self.activation)))
        epsilon = 0.02
        result: dict[str, float] = {}
        for i, name in enumerate(FEATURE_NAMES):
            perturbed = self.activation[:]
            perturbed[i] = max(0.0, perturbed[i] - epsilon)
            changed = float(score_function(tuple(perturbed)))
            result[name] = abs(baseline - changed) / epsilon
        total = sum(result.values())
        if total:
            result = {key: value / total for key, value in result.items()}
        return result

    def snapshot(self) -> dict:
        return {"activation": self.activation[:], "steps": self.steps}
