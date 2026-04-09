from pathlib import Path
from typing import Dict, List

from heuristic_solver import load_artifacts

import ast
import pandas as pd
from data_type import WeekTuple, DayBits, Stop, SchedTuple, TuplePools
from helper_funcs import a, get_dist, WEEKS, DAYS_5


# ------------------------------------------------------------------
# CSV parsing
# ------------------------------------------------------------------

def parse_week_value(x) -> WeekTuple:
    if x is None:
        return tuple()
    s = str(x).strip()
    if not s:
        return tuple()
    if s[0] in "([{" and s[-1] in ")]}":
        t = ast.literal_eval(s)
        return tuple(int(v) for v in t)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def parse_daybits_tuple_str(x) -> DayBits:
    t = ast.literal_eval(str(x))
    if len(t) != 7:
        raise ValueError(f"daybits must be length 7, got {t}")
    return tuple(int(v) for v in t)


def build_p_from_result_csv(csv_path: Path) -> Dict[int, SchedTuple]:
    df = pd.read_csv(csv_path)

    if "STOP ID" not in df.columns:
        df = pd.read_csv(csv_path, skiprows=3)

    if "STOP ID" in df.columns and "AFT_MON" in df.columns:
        ALL_DAYS_7 = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        pools = TuplePools()
        p: Dict[int, SchedTuple] = {}
        for stop_id, grp in df.groupby("STOP ID"):
            weeks = sorted(grp["WEEK#"].tolist())
            week_t = pools.week(weeks)
            first = grp.iloc[0]
            bits = tuple(int(first[f"AFT_{d}"]) for d in ALL_DAYS_7)
            day_b = pools.day(bits)
            p[int(stop_id)] = pools.schedule(week_t, day_b)
        return p

    elif {"custno", "week_AFT", "day_AFT"}.issubset(df.columns):
        p = {}
        for _, r in df.iterrows():
            p[int(r["custno"])] = (parse_week_value(r["week_AFT"]),
                                   parse_daybits_tuple_str(r["day_AFT"]))
        return p

    elif {"stop_id", "chosen_week", "chosen_daybits"}.issubset(df.columns):
        p = {}
        for _, r in df.iterrows():
            p[int(r["stop_id"])] = (parse_week_value(r["chosen_week"]),
                                    parse_daybits_tuple_str(r["chosen_daybits"]))
        return p

    else:
        raise KeyError(f"Unknown result CSV format. columns={list(df.columns)}")


# ------------------------------------------------------------------
# Objective functions
# ------------------------------------------------------------------

def _get_visited(artifacts: dict, p: Dict[int, SchedTuple]) -> dict:
    """(l,d)별 배정된 stop 목록 반환 (공통 전처리)"""
    ids = list(artifacts["stops"].keys())
    visited = {(l, d): [] for l in WEEKS for d in DAYS_5}
    for i in ids:
        for l in WEEKS:
            for d in DAYS_5:
                if a(p[i], l, d) == 1:
                    visited[(l, d)].append(i)
    return visited


def compute_nn_objective(artifacts: dict, p: Dict[int, SchedTuple]) -> float:
    """
    Nearest-neighbor objective: Σ_{l,d} Σ_i z_{ild}
    각 stop i의 z = 같은 (l,d) 내 가장 가까운 stop까지의 거리
    (stop이 혼자면 z = 0)
    """
    stops, Ddist = artifacts["stops"], artifacts["Ddist"]
    visited = _get_visited(artifacts, p)

    total = 0.0
    for ids_cell in visited.values():
        if len(ids_cell) <= 1:
            continue
        for i in ids_cell:
            total += min(get_dist(i, j, Ddist, stops) for j in ids_cell if j != i)
    return total


def compute_mssc_objective(artifacts: dict, p: Dict[int, SchedTuple]) -> float:
    """
    MSSC objective: Σ_{l,d} (최적 medoid까지의 거리 합)
    각 (l,d)에서 모든 stop까지의 거리 합이 최소인 medoid를 선택
    """
    stops, Ddist = artifacts["stops"], artifacts["Ddist"]
    visited = _get_visited(artifacts, p)

    total = 0.0
    for ids_cell in visited.values():
        if len(ids_cell) <= 1:
            continue
        total += min(
            sum(get_dist(i, j, Ddist, stops) for i in ids_cell)
            for j in ids_cell
        )
    return total


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ARTIFACTS_PATH = Path("baseline_data_store/1027633_artifacts.pkl")
    RESULT_CSV     = Path("./results_optimal/1027633_20260408_144517_resultdetail.csv")

    artifacts = load_artifacts(ARTIFACTS_PATH)
    baseline_sched: Dict[int, SchedTuple] = artifacts["baseline_sched"]
    p_optimal = build_p_from_result_csv(RESULT_CSV)

    for label, p in [("Baseline", baseline_sched), ("Optimal ", p_optimal)]:
        nn   = compute_nn_objective(artifacts, p)
        print(f"[{label}]  NN = {nn:>15,.2f}")

        mssc = compute_mssc_objective(artifacts, p)
        print(f"[{label}]  MSSC = {mssc:>15,.2f}")


    nn_b = compute_nn_objective(artifacts, baseline_sched)
    nn_o = compute_nn_objective(artifacts, p_optimal)
    print(f"\n[NN   improvement] {(nn_b - nn_o) / nn_b * 100:.2f}%")

    mssc_b = compute_mssc_objective(artifacts, baseline_sched)
    mssc_o = compute_mssc_objective(artifacts, p_optimal)
    print(f"[MSSC improvement] {(mssc_b - mssc_o) / mssc_b * 100:.2f}%")

if __name__ == "__main__":
    main()