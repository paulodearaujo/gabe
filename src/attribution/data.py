"""Shared data structures and loaders for attribution models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = [
    "EXCLUDED_SUBSTRINGS",
    "TouchSequence",
    "load_sequences",
]

EXCLUDED_SUBSTRINGS = (
    "/cadastro",
    "/checkout",
    "/erro",
    "/plans",
    "/product",
    "/plano-antecipacao",
    "/sac",
    "/receivables",
    "/settings",
    "/select-account",
)


@dataclass(frozen=True, slots=True)
class TouchSequence:
    """Represents the ordered touches for an individual client."""

    client_id: str
    steps: tuple[str, ...]
    converted: bool

    @property
    def touch_count(self) -> int:
        """Number of touchpoints excluding conversion/drop markers."""
        return len(self.steps)


def _normalise_step(raw_step: str) -> str | None:
    """Sanitise a raw touch entry and return a canonical path."""

    cleaned = raw_step.strip().lower()
    if not cleaned:
        return None

    if " @ " in cleaned:
        path_part, *_timestamp = cleaned.split(" @ ", maxsplit=1)
    else:
        path_part = cleaned

    path = path_part.strip()
    if not path:
        return None

    path = path.split("?", maxsplit=1)[0].strip()
    if not path:
        return None
    if not path.startswith("/"):
        return None

    normalised = "/" + path.lstrip("/")
    if normalised.endswith("/") and normalised != "/":
        normalised = normalised.rstrip("/")

    return normalised if normalised != "/" else None


def _extract_steps(raw_sequence: str) -> list[str]:
    """Split a path sequence string into a list of paths."""
    fragments = raw_sequence.split(" > ")
    steps: list[str] = []
    for fragment in fragments:
        path = _normalise_step(fragment)
        if path is None:
            continue
        steps.append(path)
    return steps


def _truncate_at_conversion(steps: list[str]) -> tuple[list[str], bool]:
    cleaned: list[str] = []
    converted = False
    for step in steps:
        if "/sucesso" in step:
            converted = True
            break
        cleaned.append(step)
    return cleaned, converted


def _should_exclude(path: str) -> bool:
    return any(excluded in path for excluded in EXCLUDED_SUBSTRINGS)


def load_sequences(csv_path: Path | str) -> list[TouchSequence]:
    """Load the raw CSV and convert rows into touch sequences.

    Parameters
    ----------
    csv_path:
        Path to the CSV file with `client_id` and `path_sequence` columns.
    """
    csv_path = Path(csv_path)
    dataframe = pd.read_csv(csv_path, dtype={"client_id": str})

    required_columns = {"client_id", "path_sequence"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        msg = f"CSV missing required columns: {missing}"
        raise ValueError(msg)

    sequences: list[TouchSequence] = []
    for client_id_raw, path_sequence_raw in dataframe.itertuples(index=False, name=None):
        client_id = str(client_id_raw)
        steps = _extract_steps(str(path_sequence_raw))
        cleaned_steps, converted = _truncate_at_conversion(steps)
        filtered_steps: list[str] = [
            step for step in cleaned_steps if not _should_exclude(step)
        ]
        if not filtered_steps and not converted:
            # Paths without identifiable touches are ignored.
            continue
        sequences.append(
            TouchSequence(
                client_id=client_id,
                steps=tuple(filtered_steps),
                converted=converted,
            )
        )

    return sequences
