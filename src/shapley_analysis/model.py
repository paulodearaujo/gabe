"""Shapley-value attribution model."""

from __future__ import annotations

import math
from collections import Counter, abc, defaultdict
from dataclasses import dataclass
from itertools import combinations

from attribution import TouchSequence

MIN_EXACT_PLAYERS = 2


@dataclass(slots=True)
class ChannelSetStats:
    total: int = 0
    converted: int = 0


class ShapleyAttribution:
    """Compute Shapley-value contributions for marketing channels.

    The implementation follows the cooperative-game formulation described by
    Shao & Li (2009), where players are the channels observed in a user journey.
    Conversion probability for any subset of channels is estimated empirically
    from the dataset by observing all journeys that include that subset.

    Parameters
    ----------
    sequences:
        Ordered user journeys.
    max_exact_players:
        Maximum number of distinct channels within a journey for which the exact
        Shapley value is computed via combinatorial enumeration. Journeys with
        a larger number of distinct channels fall back to an equal credit
        approximation to avoid combinatorial explosion.
    """

    def __init__(
        self,
        sequences: abc.Iterable[TouchSequence],
        *,
        max_exact_players: int = 10,
    ) -> None:
        journeys = list(sequences)
        if not journeys:
            msg = "At least one sequence is required to build the Shapley model."
            raise ValueError(msg)
        if max_exact_players < MIN_EXACT_PLAYERS:
            msg = (
                "max_exact_players must be greater or equal to "
                f"{MIN_EXACT_PLAYERS}."
            )
            raise ValueError(msg)
        for sequence in journeys:
            if not isinstance(sequence, TouchSequence):
                msg = "All sequences must be TouchSequence instances."
                raise TypeError(msg)

        self._sequences = tuple(journeys)
        self._max_exact_players = max_exact_players
        self._subset_cache: dict[frozenset[str], float] = {}

        self._channel_states = self._discover_channels(journeys)
        self._stats_by_set = self._aggregate_channel_sets(journeys)
        self._total_sequences = sum(stats.total for stats in self._stats_by_set.values())
        self._converted_sequences = sum(stats.converted for stats in self._stats_by_set.values())

    @staticmethod
    def _discover_channels(sequences: abc.Iterable[TouchSequence]) -> tuple[str, ...]:
        channels = {step for sequence in sequences for step in sequence.steps}
        return tuple(sorted(channels))

    @staticmethod
    def _aggregate_channel_sets(
        sequences: abc.Iterable[TouchSequence],
    ) -> dict[frozenset[str], ChannelSetStats]:
        stats: dict[frozenset[str], ChannelSetStats] = defaultdict(ChannelSetStats)
        for sequence in sequences:
            channel_set = frozenset(sequence.steps)
            stats[channel_set].total += 1
            if sequence.converted:
                stats[channel_set].converted += 1
        return stats

    @property
    def channel_states(self) -> tuple[str, ...]:
        return self._channel_states

    @property
    def sequences(self) -> tuple[TouchSequence, ...]:
        return self._sequences

    @property
    def total_sequences(self) -> int:
        return self._total_sequences

    @property
    def converted_count(self) -> int:
        return self._converted_sequences

    def conversion_stats(self) -> abc.Mapping[str, float]:
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
        }

    def shapley_values(self) -> dict[str, float]:
        """Raw Shapley contributions (conversion lift) per channel."""
        contributions: dict[str, float] = {channel: 0.0 for channel in self.channel_states}

        for channel_set, stats in self._stats_by_set.items():
            if not channel_set:
                continue
            per_set = self._shapley_for_set(channel_set)
            for channel, value in per_set.items():
                contributions[channel] = contributions.get(channel, 0.0) + value * stats.total

        return dict(contributions)

    def channel_attribution(self, total_conversions: float | None = None) -> dict[str, float]:
        """Rescale Shapley values so they distribute observed conversions."""
        raw = self.shapley_values()
        positive_total = sum(value for value in raw.values() if value > 0)
        if positive_total <= 0:
            return dict.fromkeys(self.channel_states, 0.0)

        target = float(total_conversions) if total_conversions is not None else float(
            self.converted_count
        )
        scale = target / positive_total
        return {channel: max(value, 0.0) * scale for channel, value in raw.items()}

    # ------------------------------------------------------------------
    # Internal helpers

    def _shapley_for_set(self, channel_set: frozenset[str]) -> dict[str, float]:
        channels = tuple(sorted(channel_set))
        count = len(channels)
        if count == 0:
            return {}
        if count > self._max_exact_players:
            value = self._subset_value(channel_set)
            if count == 0:
                return {}
            equal_share = value / count
            return dict.fromkeys(channels, equal_share)

        contributions = dict.fromkeys(channels, 0.0)
        denominator = math.factorial(count)
        for channel in channels:
            others = tuple(other for other in channels if other != channel)
            for coalition_size in range(len(others) + 1):
                factorial_a = math.factorial(coalition_size)
                factorial_b = math.factorial(count - coalition_size - 1)
                weight = factorial_a * factorial_b / denominator
                for subset in combinations(others, coalition_size):
                    subset_set = frozenset(subset)
                    delta = self._subset_value(subset_set | {channel}) - self._subset_value(
                        subset_set
                    )
                    contributions[channel] += weight * delta
        return contributions

    def _subset_value(self, subset: frozenset[str]) -> float:
        try:
            return self._subset_cache[subset]
        except KeyError:
            pass

        total = 0
        converted = 0
        for channel_set, stats in self._stats_by_set.items():
            if subset <= channel_set:
                total += stats.total
                converted += stats.converted
        value = 0.0 if total == 0 else converted / total
        self._subset_cache[subset] = value
        return value
