
import argparse
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path(__file__).with_name("Planilha.csv")
NOISE_PATH_KEYWORDS = (
    "/checkout",
    "/cadastro",
    "/erro",
    "/action",
    "/actions",
    "/settings",
    "/plano-antecipacao",
    "/pay/bank-slip",
    "/products",
    "/invoices/create",
    "/plans",
)


def parse_steps(seq_str: object) -> list[str]:
    steps = [s.strip() for s in str(seq_str).split(">")]
    out: list[str] = []
    for step in steps:
        candidate = step.split("@")[0].strip() if "@" in step else step.strip()
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if candidate:
            out.append(candidate)
    return out


def remove_noise_paths(steps: Sequence[str]) -> list[str]:
    def is_noise(path: str) -> bool:
        lower = path.lower()
        return any(keyword in lower for keyword in NOISE_PATH_KEYWORDS)

    return [step for step in steps if not is_noise(step)]


def dedup(seq: Iterable[str]) -> list[str]:
    out: list[str] = []
    for item in seq:
        if not out or out[-1] != item:
            out.append(item)
    return out


def to_path_sequence(steps: Sequence[str]) -> list[str]:
    filtered = remove_noise_paths(steps)
    return dedup(filtered)


def is_checkout_success(steps: Sequence[str]) -> bool:
    joined = " ".join(steps).lower()
    return "/checkout/" in joined and "sucesso" in joined


def build_transition_counts(
    paths_list: Sequence[Sequence[str]],
    success_flags: Sequence[bool],
    remove_path: str | None = None,
) -> defaultdict[str, Counter[str]]:
    trans: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for paths, is_success in zip(paths_list, success_flags):
        filtered = list(paths) if remove_path is None else [p for p in paths if p != remove_path]
        if not filtered:
            trans["Start"]["Null"] += 1
            continue
        trans["Start"][filtered[0]] += 1
        for a, b in zip(filtered[:-1], filtered[1:]):
            trans[a][b] += 1
        trans[filtered[-1]]["Conversion" if is_success else "Null"] += 1
    return trans


def transition_matrix(trans_counts: defaultdict[str, Counter[str]]) -> tuple[list[str], np.ndarray]:
    state_set = {"Start", "Conversion", "Null"}
    for src, outs in trans_counts.items():
        state_set.add(src)
        state_set.update(outs.keys())
    states = sorted(state_set)
    if "Start" in states:
        states.insert(0, states.pop(states.index("Start")))
    idx = {state: i for i, state in enumerate(states)}
    matrix = np.zeros((len(states), len(states)), dtype=float)
    for src, outs in trans_counts.items():
        i = idx[src]
        total = sum(outs.values())
        if total == 0:
            continue
        for dst, count in outs.items():
            matrix[i, idx[dst]] = count / total
    for absorbing_state in ("Conversion", "Null"):
        i = idx[absorbing_state]
        if np.isclose(matrix[i].sum(), 0.0):
            matrix[i, i] = 1.0
    return states, matrix


def absorption_probability_from_start(
    states: Sequence[str],
    matrix: np.ndarray,
    target_abs_state: str = "Conversion",
) -> float:
    state_to_idx = {state: i for i, state in enumerate(states)}
    absorbing: list[str] = []
    for state in states:
        i = state_to_idx[state]
        if np.isclose(matrix[i].sum(), 1.0) and np.isclose(matrix[i, i], 1.0):
            absorbing.append(state)
    transient = [state for state in states if state not in absorbing]
    if not transient or "Start" not in transient or target_abs_state not in absorbing:
        return 0.0
    idx_transient = [state_to_idx[state] for state in transient]
    idx_absorbing = [state_to_idx[state] for state in absorbing]
    sub_q = matrix[np.ix_(idx_transient, idx_transient)]
    sub_r = matrix[np.ix_(idx_transient, idx_absorbing)]
    identity = np.eye(len(transient))
    try:
        fundamental = np.linalg.inv(identity - sub_q)
    except np.linalg.LinAlgError:
        fundamental = np.linalg.pinv(identity - sub_q)
    absorption = fundamental.dot(sub_r)
    start_idx = transient.index("Start")
    target_idx = absorbing.index(target_abs_state)
    return float(absorption[start_idx, target_idx])


def compute_path_influence(
    paths_list: Sequence[Sequence[str]],
    success_flags: Sequence[bool],
    base_prob: float,
    total_checkouts: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_counts: Counter[str] = Counter()
    journey_counts: Counter[str] = Counter()
    touch_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    for paths in paths_list:
        for pos, path in enumerate(paths, start=1):
            path_counts[path] += 1
            touch_counts[path][pos] += 1
        for path in set(paths):
            journey_counts[path] += 1
    total_journeys = len(paths_list)
    total_touches = sum(path_counts.values()) or 1
    influence_records = []
    touch_records = []
    for path in sorted(path_counts):
        trans_removed = build_transition_counts(paths_list, success_flags, remove_path=path)
        states_removed, matrix_removed = transition_matrix(trans_removed)
        p_conv_removed = absorption_probability_from_start(states_removed, matrix_removed)
        removal_effect = (base_prob - p_conv_removed) / base_prob if base_prob > 0 else 0.0
        attributed = total_checkouts * removal_effect
        influence_records.append(
            {
                "path": path,
                "jornadas_com_path": journey_counts[path],
                "perc_jornadas_com_path": journey_counts[path] / total_journeys if total_journeys else 0.0,
                "toques_totais": path_counts[path],
                "perc_toques_totais": path_counts[path] / total_touches,
                "probabilidade_sem_path": p_conv_removed,
                "efeito_remocao_relativo": removal_effect,
                "checkouts_atribuidos": attributed,
            }
        )
        for pos, count in sorted(touch_counts[path].items()):
            touch_records.append(
                {
                    "path": path,
                    "toque": pos,
                    "ocorrencias": count,
                    "participacao_no_path": count / path_counts[path] if path_counts[path] else 0.0,
                    "participacao_global_de_toques": count / total_touches,
                }
            )
    influence_df = pd.DataFrame(influence_records)
    touch_df = pd.DataFrame(touch_records)
    return influence_df, touch_df


def run(input_csv: str, outdir: str) -> None:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_csv).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8")
    if "path_sequence" not in df.columns:
        raise KeyError("A planilha precisa ter a coluna 'path_sequence'.")
    df["steps"] = df["path_sequence"].apply(parse_steps)
    df["is_checkout_success"] = df["steps"].apply(is_checkout_success)
    df["paths"] = df["steps"].apply(to_path_sequence)
    df["num_toques"] = df["paths"].map(len)
    max_len = int(df["num_toques"].max() or 0)
    for idx in range(max_len):
        df[f"path_toque_{idx + 1}"] = df["paths"].map(lambda lst, idx=idx: lst[idx] if len(lst) > idx else None)
    journey_cols: list[str] = []
    if "client_id" in df.columns:
        journey_cols.append("client_id")
    journey_cols += ["is_checkout_success", "num_toques"]
    journey_cols += [f"path_toque_{idx + 1}" for idx in range(max_len)]
    journeys_path = outdir_path / "jornadas_paths.csv"
    df[journey_cols].to_csv(journeys_path, index=False)
    paths_list = df["paths"].tolist()
    success_flags = df["is_checkout_success"].astype(bool).tolist()
    trans_all = build_transition_counts(paths_list, success_flags)
    states_all, matrix_all = transition_matrix(trans_all)
    p_conv_all = absorption_probability_from_start(states_all, matrix_all)
    total_checkouts = int(df["is_checkout_success"].sum())
    influence_df, touch_df = compute_path_influence(paths_list, success_flags, p_conv_all, total_checkouts)
    influence_df = influence_df.sort_values("checkouts_atribuidos", ascending=False)
    touch_df = touch_df.sort_values(["path", "toque"])
    influence_path = outdir_path / "path_influencia_markov.csv"
    influence_df.to_csv(influence_path, index=False)
    touch_path = outdir_path / "path_toques.csv"
    touch_df.to_csv(touch_path, index=False)
    metrics_df = pd.DataFrame(
        {
            "metrica": [
                "probabilidade_conversao_base",
                "total_jornadas",
                "checkouts_totais",
            ],
            "valor": [p_conv_all, len(df), total_checkouts],
        }
    )
    metrics_path = outdir_path / "metricas_gerais.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print("Arquivos gerados:")
    print(" -", journeys_path)
    print(" -", influence_path)
    print(" -", touch_path)
    print(" -", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Caminho do CSV com as colunas client_id e path_sequence (default: Planilha.csv ao lado do script)",
    )
    parser.add_argument("--outdir", default="./saidas", help="Diretório de saída")
    args = parser.parse_args()
    run(args.input, args.outdir)
