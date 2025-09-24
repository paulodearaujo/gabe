"""Command line interface for Markov attribution analysis."""

from __future__ import annotations

import argparse
from collections import abc
from dataclasses import dataclass
from pathlib import Path

from attribution.data import load_sequences

from .model import MarkovChainAttribution
from .report import MarkovReport


@dataclass(slots=True, frozen=True)
class CLIArgs:
    """Typed container for parsed CLI arguments."""

    input_csv: Path
    output: Path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Markov attribution report.")
    _ = parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the CSV file containing path sequences.",
    )
    _ = parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/markov_report.md"),
        help="Output path for the generated Markdown report.",
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
    )


def main(argv: abc.Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)

    sequences = load_sequences(args.input_csv)
    model = MarkovChainAttribution(sequences)
    report = MarkovReport(model=model, sequences_count=len(sequences))
    report.write_markdown(args.output)


if __name__ == "__main__":
    main()
