"""Command line interface for Shapley attribution analysis."""

from __future__ import annotations

import argparse
from collections import abc
from dataclasses import dataclass
from pathlib import Path

from attribution import load_sequences

from .model import ShapleyAttribution
from .report import ShapleyReport


@dataclass(slots=True, frozen=True)
class CLIArgs:
    input_csv: Path
    output: Path
    max_exact_players: int
    top_channels: int


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Shapley attribution report.",
    )
    _ = parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the CSV file containing path sequences.",
    )
    _ = parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/shapley_report.md"),
        help="Output path for the generated Markdown report.",
    )
    _ = parser.add_argument(
        "--max-exact-players",
        type=int,
        default=10,
        help="Maximum number of distinct channels in a journey for exact Shapley computation.",
    )
    _ = parser.add_argument(
        "--top-channels",
        type=int,
        default=25,
        help="Number of channels to display in the summary table.",
    )
    return parser


def parse_arguments(argv: abc.Sequence[str] | None = None) -> CLIArgs:
    parser = build_argument_parser()
    if argv is not None and not isinstance(argv, abc.Sequence):
        msg = "argv deve ser uma sequência de strings."
        raise TypeError(msg)
    namespace = parser.parse_args(argv)
    return CLIArgs(
        input_csv=Path(namespace.input_csv),
        output=Path(namespace.output),
        max_exact_players=int(namespace.max_exact_players),
        top_channels=int(namespace.top_channels),
    )


def main(argv: abc.Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)

    sequences = load_sequences(args.input_csv)
    model = ShapleyAttribution(sequences, max_exact_players=args.max_exact_players)
    report = ShapleyReport(model=model, sequences_count=len(sequences))
    report.write_markdown(args.output, top_n=args.top_channels)


if __name__ == "__main__":
    main()
