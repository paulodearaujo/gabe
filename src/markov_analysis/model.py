"""Markov chain attribution model."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from attribution.data import TouchSequence

START = "__start__"
CONVERSION = "__conversion__"
DROP = "__drop__"


@dataclass(slots=True)
class TransitionMatrix:
    """Container for transition counts and probabilities."""

    counts: dict[str, Counter[str]]
    probabilities: dict[str, dict[str, float]]

    def states(self) -> set[str]:
        return set(self.counts).union({s for row in self.counts.values() for s in row})


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


class MarkovChainAttribution:
    """Construct a first-order Markov chain and compute attribution metrics."""

    def __init__(self, sequences: Sequence[TouchSequence]) -> None:
        if not sequences:
            msg = "At least one sequence is required to build the Markov model."
            raise ValueError(msg)
        self._sequences = list(sequences)
        self.transition_matrix = self._build_transition_matrix(sequences)
        self._channel_states = self._infer_channel_states()

    @staticmethod
    def _build_transition_matrix(
        sequences: Iterable[TouchSequence],
    ) -> TransitionMatrix:
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for sequence in sequences:
            states = [START, *sequence.steps]
            terminal = CONVERSION if sequence.converted else DROP
            states.append(terminal)
            for source, target in pairwise(states):
                counts[source][target] += 1

        probabilities: dict[str, dict[str, float]] = {}
        for source, destinations in counts.items():
            total = sum(destinations.values())
            if total == 0:
                continue
            probabilities[source] = {
                destination: value / total for destination, value in destinations.items()
            }

        # Ensure absorbing states exist with deterministic self transitions.
        for absorbing in (CONVERSION, DROP):
            counts.setdefault(absorbing, Counter({absorbing: 1}))
            probabilities.setdefault(absorbing, {absorbing: 1.0})

        return TransitionMatrix(counts=dict(counts), probabilities=probabilities)

    def _infer_channel_states(self) -> tuple[str, ...]:
        states = sorted(
            {
                state
                for state in self.transition_matrix.states()
                if state not in {START, CONVERSION, DROP}
            }
        )
        return tuple(states)

    @property
    def channel_states(self) -> tuple[str, ...]:
        return self._channel_states

    @property
    def base_conversion_probability(self) -> float:
        return self._absorption_probability(transition_override=None)

    @property
    def sequences(self) -> tuple[TouchSequence, ...]:
        return tuple(self._sequences)

    @property
    def converted_count(self) -> int:
        return sum(1 for sequence in self._sequences if sequence.converted)

    def removal_effects(self) -> dict[str, float]:
        base_probability = self.base_conversion_probability
        effects: dict[str, float] = {}
        for channel in self.channel_states:
            probability_without = self._absorption_probability(transition_override=channel)
            effects[channel] = max(base_probability - probability_without, 0.0)
        return effects

    def _build_transition_matrix_with_override(
        self,
        override_channel: str | None,
    ) -> TransitionMatrix:
        if override_channel is None:
            return self.transition_matrix

        filtered_sequences: list[TouchSequence] = []
        for sequence in self._sequences:
            filtered_steps = tuple(step for step in sequence.steps if step != override_channel)
            converted = sequence.converted
            if not filtered_steps and not converted:
                continue
            # If conversion occurs but all steps removed, the client converts directly from start.
            filtered_sequences.append(
                TouchSequence(
                    client_id=sequence.client_id,
                    steps=filtered_steps,
                    converted=sequence.converted,
                )
            )

        if not filtered_sequences:
            # No remaining paths; return a degenerate matrix with only absorbing states.
            counts = {
                START: Counter({DROP: 1}),
                CONVERSION: Counter({CONVERSION: 1}),
                DROP: Counter({DROP: 1}),
            }
            probabilities = {
                START: {DROP: 1.0},
                CONVERSION: {CONVERSION: 1.0},
                DROP: {DROP: 1.0},
            }
            return TransitionMatrix(counts=counts, probabilities=probabilities)

        return self._build_transition_matrix(filtered_sequences)

    def _absorption_probability(self, transition_override: str | None) -> float:
        matrix = self._build_transition_matrix_with_override(transition_override)
        states = sorted(matrix.states())

        absorbing_states = [CONVERSION, DROP]
        transient_states = [state for state in states if state not in absorbing_states]

        if not transient_states:
            return 0.0

        index_map = {state: idx for idx, state in enumerate(transient_states + absorbing_states)}
        size = len(index_map)
        transition = np.zeros((size, size), dtype=float)

        for source, destinations in matrix.probabilities.items():
            for destination, probability in destinations.items():
                if source not in index_map or destination not in index_map:
                    continue
                row = index_map[source]
                col = index_map[destination]
                transition[row, col] = probability

        transient_count = len(transient_states)
        if transient_count == 0:
            return 0.0

        q = transition[:transient_count, :transient_count]
        r = transition[:transient_count, transient_count:]

        identity = np.eye(transient_count)
        try:
            fundamental = np.linalg.inv(identity - q)
        except np.linalg.LinAlgError:
            return 0.0

        absorption = fundamental @ r
        start_index = transient_states.index(START) if START in transient_states else None
        if start_index is None:
            return 0.0

        conversion_index = absorbing_states.index(CONVERSION)
        return float(absorption[start_index, conversion_index])

    def transition_dataframe(self) -> pd.DataFrame:
        states = sorted(self.transition_matrix.states())
        state_index = pd.Index(states, dtype="object")
        frame = pd.DataFrame(index=state_index, columns=state_index, data=0.0, dtype=float)
        for source, destinations in self.transition_matrix.probabilities.items():
            for destination, probability in destinations.items():
                frame.loc[source, destination] = probability
        return frame

    def channel_attribution(self, total_conversions: int) -> dict[str, float]:
        effects = self.removal_effects()
        total_effect = sum(effects.values())
        if total_effect <= 0.0:
            return dict.fromkeys(effects, 0.0)
        return {
            channel: total_conversions * (effect / total_effect)
            for channel, effect in effects.items()
        }

    def conversion_stats(self) -> Mapping[str, float]:
        total_sequences = len(self._sequences)
        converted = sum(1 for sequence in self._sequences if sequence.converted)
        average_touches = float(
            sum(sequence.touch_count for sequence in self._sequences) / total_sequences
        )
        average_touches_converted = (
            float(sum(sequence.touch_count for sequence in self._sequences if sequence.converted))
            / converted
            if converted
            else 0.0
        )
        return {
            "total_sequences": float(total_sequences),
            "converted_sequences": float(converted),
            "conversion_rate": float(converted / total_sequences) if total_sequences else 0.0,
            "average_touches": average_touches,
            "average_touches_converted": average_touches_converted,
            "base_conversion_probability": self.base_conversion_probability,
        }
