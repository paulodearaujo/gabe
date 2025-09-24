"""Report generation helpers."""

from __future__ import annotations

from collections import abc
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pandas import DataFrame
else:
    DataFrame = Any

from .model import CONVERSION, DROP, START, MarkovChainAttribution


@dataclass(slots=True)
class KPIBundle:
    total_sequences: int
    converted_sequences: int
    conversion_rate: float
    average_touches: float
    average_touches_converted: float
    base_conversion_probability: float


@dataclass(slots=True)
class ChannelAttributionRow:
    channel: str
    removal_effect: float
    attributed_conversions: float


class MarkovReport:
    """Construct and persist a Markdown report for the Markov analysis."""

    def __init__(self, model: MarkovChainAttribution, sequences_count: int) -> None:
        self._model = model
        self._total_sequences = sequences_count

    def build_kpis(self) -> KPIBundle:
        stats = self._model.conversion_stats()
        return KPIBundle(
            total_sequences=int(stats["total_sequences"]),
            converted_sequences=int(stats["converted_sequences"]),
            conversion_rate=stats["conversion_rate"],
            average_touches=stats["average_touches"],
            average_touches_converted=stats["average_touches_converted"],
            base_conversion_probability=stats["base_conversion_probability"],
        )

    def build_channel_attribution(self) -> list[ChannelAttributionRow]:
        effects = self._model.removal_effects()
        attributed = self._model.channel_attribution(total_conversions=self._model.converted_count)
        return [
            ChannelAttributionRow(
                channel=channel,
                removal_effect=effects[channel],
                attributed_conversions=attributed.get(channel, 0.0),
            )
            for channel in sorted(effects)
        ]

    def build_transition_table(self) -> DataFrame:
        frame = self._model.transition_dataframe()
        return frame.loc[:, sorted(frame.columns)]

    def to_markdown(self) -> str:
        kpis = self.build_kpis()
        attribution_rows = self.build_channel_attribution()
        transition_table = self.build_transition_table()

        base_prob = kpis.base_conversion_probability
        kpi_section = "\n".join(
            [
                "## Visão Geral",
                f"- Total de jornadas analisadas: **{kpis.total_sequences:,}**",
                f"- Jornadas convertidas: **{kpis.converted_sequences:,}**",
                f"- Taxa de conversão: **{kpis.conversion_rate:.2%}**",
                f"- Toques médios por jornada: **{kpis.average_touches:.2f}**",
                f"- Toques médios (conversões): **{kpis.average_touches_converted:.2f}**",
                (
                    "- Probabilidade base de conversão (Markov): "
                    f"**{base_prob:.2%}**"
                ),
            ]
        )

        attribution_header = (
            "| Canal | Efeito de Remoção | Conversões Atribuídas |\n| --- | ---: | ---: |"
        )
        attribution_lines = [
            f"| `{row.channel}` | {row.removal_effect:.4f} | {row.attributed_conversions:.2f} |"
            for row in attribution_rows
        ]
        attribution_section = "\n".join(
            ["## Atribuição por Canal", attribution_header, *attribution_lines]
        )

        formatted_states = format_states(tuple(map(str, tuple(transition_table.index))))
        state_section = (
            "## Estados Modelados\n"
            f"{formatted_states}\n"
            "> `__start__` marca o início, `__conversion__` representa a conversão e "
            "`__drop__` indica abandono."
        )

        transition_markdown = transition_table.to_markdown(floatfmt=".3f") or ""
        transition_section = f"## Matriz de Transição\n{transition_markdown}"

        methodology_section = (
            "## Metodologia\n"
            "1. Cortamos jornadas vazias e truncamos no primeiro `/sucesso`.\n"
            "2. Adicionamos estados absorventes de conversão e abandono à cadeia de Markov.\n"
            "3. Calculamos a probabilidade de conversão com a matriz fundamental.\n"
            "4. Reavaliamos conversões removendo cada canal e recomputando a probabilidade.\n"
            "5. Distribuímos conversões conforme o impacto de remoção dos canais."
        )

        sections = (
            "# Relatório de Atribuição via Cadeia de Markov",
            kpi_section,
            attribution_section,
            state_section,
            transition_section,
            methodology_section,
        )
        return "\n\n".join(sections)

    def write_markdown(self, output_path: Path | str) -> None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_markdown(), encoding="utf-8")


def format_states(states: abc.Iterable[str]) -> str:
    if not isinstance(states, abc.Iterable):
        msg = "states deve ser uma coleção iterável de strings."
        raise TypeError(msg)
    states = tuple(states)
    ordering = (START, CONVERSION, DROP)
    specials = {state for state in states if state in ordering}
    channels = sorted(set(states) - specials)
    ordered = [state for state in ordering if state in specials]
    ordered.extend(channels)
    return ", ".join(f"`{state}`" for state in ordered)
