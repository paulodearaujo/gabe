"""Shapley attribution report generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from pandas import DataFrame
else:
    DataFrame = Any

from .model import ShapleyAttribution


@dataclass(slots=True)
class KPIBundle:
    total_sequences: int
    converted_sequences: int
    conversion_rate: float
    average_touches: float
    average_touches_converted: float


@dataclass(slots=True)
class ChannelAttributionRow:
    channel: str
    shapley_value: float
    attributed_conversions: float


class ShapleyReport:
    """Build and render a Markdown report for Shapley attribution."""

    def __init__(self, model: ShapleyAttribution, sequences_count: int) -> None:
        if not isinstance(model, ShapleyAttribution):
            msg = "model must be an instance of ShapleyAttribution."
            raise TypeError(msg)
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
        )

    def build_channel_rows(self) -> list[ChannelAttributionRow]:
        raw = self._model.shapley_values()
        attributed = self._model.channel_attribution(total_conversions=self._model.converted_count)
        ordered_channels = sorted(
            raw,
            key=lambda channel: attributed.get(channel, 0.0),
            reverse=True,
        )
        return [
            ChannelAttributionRow(
                channel=channel,
                shapley_value=raw.get(channel, 0.0),
                attributed_conversions=attributed.get(channel, 0.0),
            )
            for channel in ordered_channels
        ]

    def build_channel_dataframe(self) -> DataFrame:
        rows = self.build_channel_rows()
        data = {
            "Canal": [row.channel for row in rows],
            "Valor Shapley": [row.shapley_value for row in rows],
            "Conversões Atribuídas": [row.attributed_conversions for row in rows],
        }
        frame = pd.DataFrame(data)
        pretty = frame.copy()
        pretty["Valor Shapley"] = pretty["Valor Shapley"].map(lambda value: f"{value:.6f}")
        pretty["Conversões Atribuídas"] = pretty["Conversões Atribuídas"].map(
            lambda value: f"{value:.2f}"
        )
        return pretty

    def to_markdown(self, top_n: int = 25) -> str:
        kpis = self.build_kpis()
        channel_rows = self.build_channel_rows()[:top_n]
        dataframe = self.build_channel_dataframe().head(top_n)

        kpi_section = "\n".join(
            [
                "## Visão Geral",
                f"- Total de jornadas analisadas: **{kpis.total_sequences:,}**",
                f"- Jornadas convertidas: **{kpis.converted_sequences:,}**",
                f"- Taxa de conversão: **{kpis.conversion_rate:.2%}**",
                f"- Toques médios por jornada: **{kpis.average_touches:.2f}**",
                f"- Toques médios (conversões): **{kpis.average_touches_converted:.2f}**",
            ]
        )

        attribution_header = (
            "| Canal | Valor Shapley | Conversões Atribuídas |\n| --- | ---: | ---: |"
        )
        attribution_lines = [
            f"| `{row.channel}` | {row.shapley_value:.6f} | {row.attributed_conversions:.2f} |"
            for row in channel_rows
        ]
        attribution_section = "\n".join(
            [
                "## Atribuição por Canal (Top N)",
                attribution_header,
                *attribution_lines,
            ]
        )

        dataframe_section = (
            "## Tabela Completa\n"
            "Tabela detalhada com os canais ordenados por conversões atribuídas.\n"
            f"{dataframe.to_markdown(index=False)}"
        )

        methodology_section = "\n".join(
            [
                "## Metodologia",
                "1. Agrupamos jornadas por conjunto único de canais (ordem ignorada).",
                "2. Para cada subconjunto de canais, estimamos a taxa de conversão observada.",
                "3. Calculamos o valor de Shapley exato para conjuntos até o limite configurado.",
                "4. Para conjuntos maiores, aplicamos uma aproximação de crédito igualitário.",
                "5. Escalamos as contribuições positivas para distribuir todas as conversões observadas.",
            ]
        )

        sections = (
            "# Relatório de Atribuição pelo Valor de Shapley",
            kpi_section,
            attribution_section,
            dataframe_section,
            methodology_section,
        )
        return "\n\n".join(sections)

    def write_markdown(self, output_path: Path | str, *, top_n: int = 25) -> None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_markdown(top_n=top_n), encoding="utf-8")
