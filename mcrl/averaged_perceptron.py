"""Sparse averaged perceptron with deterministic label selection."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence


class AveragedPerceptron:
    """Feature-label linear model; no token generation or neural hidden states."""

    def __init__(self) -> None:
        self.weights: dict[str, dict[str, float]] = {}
        self._totals: dict[tuple[str, str], float] = {}
        self._timestamps: dict[tuple[str, str], int] = {}
        self._step = 0

    def predict(self, features: Sequence[str], allowed: Iterable[str] | None = None) -> str:
        allowed_set = set(allowed) if allowed is not None else None
        scores: dict[str, float] = defaultdict(float)
        for feature in features:
            for label, weight in self.weights.get(feature, {}).items():
                if allowed_set is None or label in allowed_set:
                    scores[label] += weight
        if allowed_set is None:
            candidates = set(scores)
        else:
            candidates = allowed_set
        if not candidates:
            raise ValueError("AveragedPerceptron cannot predict without candidate labels.")
        return min(candidates, key=lambda label: (-scores.get(label, 0.0), label))

    def _update_weight(self, feature: str, label: str, delta: float) -> None:
        key = (feature, label)
        previous = self.weights.setdefault(feature, {}).get(label, 0.0)
        last = self._timestamps.get(key, 0)
        self._totals[key] = self._totals.get(key, 0.0) + (self._step - last) * previous
        self._timestamps[key] = self._step
        new = previous + delta
        if abs(new) < 1e-12:
            self.weights[feature].pop(label, None)
            if not self.weights[feature]:
                self.weights.pop(feature, None)
        else:
            self.weights[feature][label] = new

    def update(self, truth: str, guess: str, features: Sequence[str]) -> None:
        self._step += 1
        if truth == guess:
            return
        for feature in features:
            self._update_weight(feature, truth, 1.0)
            self._update_weight(feature, guess, -1.0)

    def averaged_weights(self) -> dict[str, dict[str, float]]:
        """Return a detached averaged model, including parameters now at zero."""
        result: dict[str, dict[str, float]] = {}
        keys = set(self._totals) | {
            (feature, label)
            for feature, labels in self.weights.items()
            for label in labels
        }
        for feature, label in keys:
            current = self.weights.get(feature, {}).get(label, 0.0)
            key = (feature, label)
            total = self._totals.get(key, 0.0)
            total += (self._step - self._timestamps.get(key, 0)) * current
            averaged = total / max(1, self._step)
            if abs(averaged) >= 1e-9:
                result.setdefault(feature, {})[label] = averaged
        return result

    def set_weights(self, weights: dict[str, dict[str, float]]) -> None:
        self.weights = {
            str(feature): {
                str(label): float(value)
                for label, value in labels.items()
                if float(value) != 0.0
            }
            for feature, labels in weights.items()
            if labels
        }
        self._totals.clear()
        self._timestamps.clear()
        self._step = 0
