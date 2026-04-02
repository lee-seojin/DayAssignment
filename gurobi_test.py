from pathlib import Path
from typing import Dict, Tuple

from heuristic_solver import load_artifacts

import ast
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from data_type import WEEKS, DAYS_5, WeekTuple, DayBits, SchedTuple

def parse_week_value(x) -> WeekTuple:

    if x is None:
        return tuple()
    s = str(x).strip()
    if not s:
        return tuple()

    # (1, 3) 같은 파이썬 리터럴
    if (s[0] in "([{" and s[-1] in ")]}"):
        t = ast.literal_eval(s)
        return tuple(int(v) for v in t)

    # "1,3" 또는 "4"
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) for p in parts)

def parse_daybits_tuple_str(x) -> DayBits:
    """
    "(0, 1, 0, 0, 0, 0, 0)" 형태를 DayBits로.
    """
    t = ast.literal_eval(str(x))
    if len(t) != 7:
        raise ValueError(f"daybits must be length 7, got {t}")
    return tuple(int(v) for v in t)  # type: ignore

def build_p_from_result_csv(csv_path: Path) -> Dict[int, SchedTuple]:
    df = pd.read_csv(csv_path, skiprows=3)

    if {"custno", "week_AFT", "day_AFT"}.issubset(df.columns):
        id_col, week_col, day_col = "custno", "week_AFT", "day_AFT"
        week_parser = parse_week_value  # "(1, 3)"도 커버 가능하게
    elif {"stop_id", "chosen_week", "chosen_daybits"}.issubset(df.columns):
        id_col, week_col, day_col = "stop_id", "chosen_week", "chosen_daybits"
        week_parser = parse_week_value
    else:
        raise KeyError(f"Unknown result CSV format. columns={list(df.columns)}")

    p: Dict[int, SchedTuple] = {}
    for _, r in df.iterrows():
        sid = int(r[id_col])
        week_t = week_parser(r[week_col])
        day_b = parse_daybits_tuple_str(r[day_col])
        p[sid] = (week_t, day_b)

    return p

from typing import Dict, Tuple, List
from data_type import Stop, SchedTuple
from helper_funcs import a, get_dist, WEEKS, DAYS_5

Cell = Tuple[int, str]


def build_knn_graph(ids, stops, Ddist, K=20):
    neigh = {}
    for i in ids:
        dists = sorted(
            [(get_dist(i, j, Ddist, stops), j) for j in ids if j != i],
            key=lambda x: x[0]
        )[:K]
        neigh[i] = [j for _, j in dists]
    return neigh


def compute_function(
    artifacts: dict,
    p: Dict[int, SchedTuple],
    w_mst: float = 1.0,
    w_mssc: float = 1.0,
    w_vbal: float = 0.0,
):
    stops: Dict[int, Stop] = artifacts["stops"]
    Ddist = artifacts["Ddist"]
    ids = list(stops.keys())

    # ---------------------------
    # 1. cell grouping
    # ---------------------------
    visited = {(l, d): [] for l in WEEKS for d in DAYS_5}

    for i in ids:
        sched = p[i]
        for l in WEEKS:
            for d in DAYS_5:
                if a(sched, l, d) == 1:
                    visited[(l, d)].append(i)

    # ---------------------------
    # 2. KNN graph (K=20 동일하게)
    # ---------------------------
    K = 20
    neigh = build_knn_graph(ids, stops, Ddist, K=K)

    # ---------------------------
    # 3. MST on KNN graph (Prim restricted)
    # ---------------------------
    def compute_mst_knn(ids_cell: List[int]) -> float:
        if len(ids_cell) <= 1:
            return 0.0

        ids_set = set(ids_cell)
        used = set([ids_cell[0]])
        total = 0.0

        while len(used) < len(ids_cell):
            best = float("inf")
            best_j = None

            for i in used:
                for j in neigh[i]:
                    if j not in ids_set or j in used:
                        continue
                    d = get_dist(i, j, Ddist, stops)
                    if d < best:
                        best = d
                        best_j = j

            # fallback (연결 안될 때)
            if best_j is None:
                for i in used:
                    for j in ids_cell:
                        if j in used:
                            continue
                        d = get_dist(i, j, Ddist, stops)
                        if d < best:
                            best = d
                            best_j = j

            total += best
            used.add(best_j)

        return total

    mst_total = 0.0
    for ids_cell in visited.values():
        mst_total += compute_mst_knn(ids_cell)

    # ---------------------------
    # 4. MSSC (medoid 기반)
    # ---------------------------
    def compute_mssc_medoid(ids_cell: List[int]) -> float:
        if len(ids_cell) <= 1:
            return 0.0

        best_total = float("inf")

        for j in ids_cell:  # j = medoid 후보
            total = 0.0
            for i in ids_cell:
                d = get_dist(i, j, Ddist, stops)
                total += d
            if total < best_total:
                best_total = total

        return best_total

    mssc_total = 0.0
    for ids_cell in visited.values():
        mssc_total += compute_mssc_medoid(ids_cell)

    # ---------------------------
    # 5. Volume balance
    # ---------------------------
    V = {(l, d): 0.0 for l in WEEKS for d in DAYS_5}

    for i, s in stops.items():
        sched = p[i]
        for l in WEEKS:
            for d in DAYS_5:
                if a(sched, l, d) == 1:
                    V[(l, d)] += float(s.volume)

    vbal = 0.0
    for l in WEEKS:
        vals = [V[(l, d)] for d in DAYS_5]
        vbal += (max(vals) - min(vals))

    # ---------------------------
    # 6. Final objective
    # ---------------------------
    obj = (
        w_mst * mst_total +
        w_mssc * mssc_total +
        w_vbal * vbal
    )

    return {
        "obj": obj,
        "mst_knn": mst_total,
        "mssc_medoid": mssc_total,
        "vol_balance": vbal,
    }

def main():
    # 너 프로젝트에서 artifacts.pkl 저장된 위치로만 맞춰주면 됨
    ARTIFACTS_PATH = Path("baseline_data_store/1027633_artifacts.pkl")
    #ARTIFACTS_PATH = Path("baseline_data_store/1042199_artifacts.pkl")

    GUROBI_CSV = Path("./results_optimal/1027633_20260129_233914_optimal_resultdetail.csv")
    #GUROBI_CSV = Path("./results_optimal/1042199_20260129_204546_optimal_resultdetail.csv")

    artifacts = load_artifacts(ARTIFACTS_PATH)
    p = build_p_from_result_csv(GUROBI_CSV)

    obj = compute_objective(artifacts, p, w1=1.0, w2=1.0)


if __name__ == "__main__":
    main()
