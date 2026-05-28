from pathlib import Path
from typing import Dict, List, Tuple
import ast
import itertools

import pandas as pd

from optimal_utils import load_artifacts, get_dist, DAYS_5, _get_visited_by_cell
from data_type import WeekTuple, DayBits, SchedTuple, TuplePools
from shapely.geometry import MultiPoint, Point

# CSV parsing
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

def _dowcd_to_weektuple(x, pools: TuplePools, timecycle: int) -> WeekTuple:
    s = str(x).strip().upper()

    # 1-week compressed model
    if timecycle == 1:
        return pools.week([1])

    if s == "A":
        return pools.week([1, 2, 3, 4])
    if s == "O":
        return pools.week([1, 3])
    if s == "E":
        return pools.week([2, 4])
    if s in {"1", "2", "3", "4"}:
        return pools.week([int(s)])
    if "," in s:
        return pools.week([int(v.strip()) for v in s.split(",") if v.strip()])
    return tuple()

def build_p_from_result_csv(csv_path: Path, artifacts: dict) -> Dict[int, SchedTuple]:
    df = pd.read_csv(csv_path)

    if "STOP ID" not in df.columns:
        df = pd.read_csv(csv_path, skiprows=3)

    timecycle = int(artifacts["timecycle"])
    pools = TuplePools()

    if "STOP ID" in df.columns and "AFT_MON" in df.columns and "AFT_WK_CD" in df.columns:
        ALL_DAYS_7 = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        p: Dict[int, SchedTuple] = {}
        for _, r in df.iterrows():
            stop_id = int(r["STOP ID"])
            week_t = _dowcd_to_weektuple(r["AFT_WK_CD"], pools, timecycle)
            bits = tuple(int(r[f"AFT_{d}"]) for d in ALL_DAYS_7)
            day_b = pools.day(bits)
            p[stop_id] = pools.schedule(week_t, day_b)
        return p

    elif {"custno", "week_AFT", "day_AFT"}.issubset(df.columns):
        p = {}
        for _, r in df.iterrows():
            week_t = parse_week_value(r["week_AFT"])
            if timecycle == 1:
                week_t = pools.week([1])
            p[int(r["custno"])] = (
                week_t,
                parse_daybits_tuple_str(r["day_AFT"]),
            )
        return p

    elif {"stop_id", "chosen_week", "chosen_daybits"}.issubset(df.columns):
        p = {}
        for _, r in df.iterrows():
            week_t = parse_week_value(r["chosen_week"])
            if timecycle == 1:
                week_t = pools.week([1])
            p[int(r["stop_id"])] = (
                week_t,
                parse_daybits_tuple_str(r["chosen_daybits"]),
            )
        return p

def compute_nn_objective(artifacts: dict, p: Dict[int, SchedTuple]) -> float:
    stops, Ddist = artifacts["stops"], artifacts["Ddist"]
    visited = _get_visited_by_cell(artifacts, p)

    total = 0.0
    for ids_cell in visited.values():
        if len(ids_cell) <= 1:
            continue
        for i in ids_cell:
            total += min(get_dist(i, j, Ddist, stops) for j in ids_cell if j != i)
    return total

def compute_mssc_objective(artifacts: dict, p: Dict[int, SchedTuple]) -> float:
    stops, Ddist = artifacts["stops"], artifacts["Ddist"]
    visited = _get_visited_by_cell(artifacts, p)

    total = 0.0
    for ids_cell in visited.values():
        if len(ids_cell) <= 1:
            continue
        total += min(
            sum(get_dist(i, j, Ddist, stops) for i in ids_cell)
            for j in ids_cell
        )
    return total

# Rectangle model metrics
def compute_rectangle_size_term(artifacts: dict, p: Dict[int, SchedTuple]) -> float:
    stops = artifacts["stops"]
    visited = _get_visited_by_cell(artifacts, p)

    total = 0.0
    for ids_cell in visited.values():
        if not ids_cell:
            continue

        xs = [float(stops[i].xcoord) for i in ids_cell]
        ys = [float(stops[i].ycoord) for i in ids_cell]

        x1_ld = min(xs)
        x2_ld = max(xs)
        y1_ld = min(ys)
        y2_ld = max(ys)

        total += (x2_ld - x1_ld) + (y2_ld - y1_ld)

    return total

def compute_rectangle_overlap_sides_term(artifacts: dict, p: Dict[int, SchedTuple]) -> float:
    stops = artifacts["stops"]
    visited = _get_visited_by_cell(artifacts, p)

    total = 0.0
    day_pairs = list(itertools.combinations(DAYS_5, 2))   # d < e only
    timecycle = int(artifacts["timecycle"])
    weeks_local = list(range(1, timecycle + 1))

    for l in weeks_local:
        rect = {}
        for d in DAYS_5:
            ids_cell = visited[(l, d)]
            if not ids_cell:
                rect[d] = None
                continue

            xs = [float(stops[i].xcoord) for i in ids_cell]
            ys = [float(stops[i].ycoord) for i in ids_cell]
            rect[d] = (min(xs), max(xs), min(ys), max(ys))   # x1, x2, y1, y2

        for d, e in day_pairs:
            rd = rect[d]
            re = rect[e]
            if rd is None or re is None:
                continue

            x1_ld, x2_ld, y1_ld, y2_ld = rd
            x1_le, x2_le, y1_le, y2_le = re

            wx = max(0.0, min(x2_ld, x2_le) - max(x1_ld, x1_le))
            wy = max(0.0, min(y2_ld, y2_le) - max(y1_ld, y1_le))

            wc = wx + wy if (wx > 0.0 and wy > 0.0) else 0.0
            total += wc

    return total

def compute_rectangle_objective(artifacts: dict, p: Dict[int, SchedTuple], w1: float = 1.0, w2: float = 1.0,) -> Tuple[float, float, float]:
    rect_term = compute_rectangle_size_term(artifacts, p)
    overlap_term = compute_rectangle_overlap_sides_term(artifacts, p)
    obj = w1 * rect_term + w2 * overlap_term

    return obj, rect_term, overlap_term

def compute_rectangle_overlap_nodes_term(artifacts: dict, p: Dict[int, SchedTuple]) -> float:

    stops = artifacts["stops"]
    visited = _get_visited_by_cell(artifacts, p)

    timecycle = int(artifacts["timecycle"])
    weeks_local = list(range(1, timecycle + 1))

    total = 0.0

    for l in weeks_local:
        visited_set = {d: set(visited[(l, d)]) for d in DAYS_5}
        rect = {}

        for e in DAYS_5:
            ids_cell = visited[(l, e)]

            if not ids_cell:
                rect[e] = None
                continue

            xs = [float(stops[i].xcoord) for i in ids_cell]
            ys = [float(stops[i].ycoord) for i in ids_cell]
            rect[e] = (min(xs), max(xs), min(ys), max(ys))

        week_ids = set().union(*(visited_set[d] for d in DAYS_5))

        for i in week_ids:
            xi = float(stops[i].xcoord)
            yi = float(stops[i].ycoord)

            for e in DAYS_5:
                if i in visited_set[e]:
                    continue

                if rect[e] is None:
                    continue

                x1_e, x2_e, y1_e, y2_e = rect[e]

                if x1_e <= xi <= x2_e and y1_e <= yi <= y2_e:
                    total += 1.0

    return total

# NHO/Convex Hull metrics
def compute_nho_metric(artifacts: dict, p: Dict[int, SchedTuple]) -> float:
    stops = artifacts["stops"]
    visited = _get_visited_by_cell(artifacts, p)
    timecycle = int(artifacts["timecycle"])
    weeks_local = list(range(1, timecycle + 1))
    day_pairs = list(itertools.combinations(DAYS_5, 2))   # d < e only

    week_nho_values = []

    for l in weeks_local:
        hulls = {}
        active_days = []

        for d in DAYS_5:
            ids_cell = visited[(l, d)]
            if len(ids_cell) < 3: # 점이 2개 이하이면 polygon area가 0, NHO 계산 대상에서 제외
                hulls[d] = None
                continue

            pts = [(float(stops[i].xcoord), float(stops[i].ycoord)) for i in ids_cell]
            hull = MultiPoint(pts).convex_hull

            if hull.geom_type != "Polygon" or hull.area <= 0:
                hulls[d] = None
                continue

            hulls[d] = hull
            active_days.append(d)

        n_active = len(active_days)
        if n_active < 2:
            week_nho_values.append(0.0)
            continue

        directed_overlap_sum = 0.0

        for d, e in day_pairs:
            hd = hulls.get(d)
            he = hulls.get(e)
            if hd is None or he is None:
                continue

            inter_area = hd.intersection(he).area
            if inter_area <= 0:
                continue

            directed_overlap_sum += inter_area / hd.area
            directed_overlap_sum += inter_area / he.area

        nho_l = directed_overlap_sum / (n_active * (n_active - 1))
        week_nho_values.append(nho_l)

    if not week_nho_values:
        return 0.0

    return sum(week_nho_values) / len(week_nho_values)

def compute_pairwise_nho(artifacts: dict, p: Dict[int, SchedTuple], label: str) -> pd.DataFrame:
    """
    For each week and day pair (d,e), compute:
        - convex hull area of day d/e
        - intersection area between hull d and hull e
        - directed NHO terms:
            Area(C_d ∩ C_e) / Area(C_d)
            Area(C_d ∩ C_e) / Area(C_e)
        - number of nodes inside the intersection polygon
            * nodes from day d
            * nodes from day e
            * nodes from all days in the same week
    """
    stops = artifacts["stops"]
    visited = _get_visited_by_cell(artifacts, p)

    timecycle = int(artifacts["timecycle"])
    weeks_local = list(range(1, timecycle + 1))
    day_pairs = list(itertools.combinations(DAYS_5, 2))

    rows = []

    for l in weeks_local:
        hulls = {}
        hull_areas = {}
        hull_geom_types = {}
        day_counts = {}

        # Build convex hull for each day
        for d in DAYS_5:
            ids_cell = visited[(l, d)]
            day_counts[d] = len(ids_cell)

            if len(ids_cell) < 3:
                hulls[d] = None
                hull_areas[d] = 0.0
                hull_geom_types[d] = "None_less_than_3_points"
                continue

            pts = [
                (float(stops[i].xcoord), float(stops[i].ycoord))
                for i in ids_cell
            ]

            hull = MultiPoint(pts).convex_hull
            hull_geom_types[d] = hull.geom_type

            if hull.geom_type != "Polygon" or hull.area <= 0:
                hulls[d] = None
                hull_areas[d] = 0.0
                continue

            hulls[d] = hull
            hull_areas[d] = float(hull.area)

        # All nodes visited in this week, regardless of day
        week_ids = []
        for d in DAYS_5:
            week_ids.extend(visited[(l, d)])
        week_ids = list(set(week_ids))

        # Pairwise hull overlap diagnostics
        for d, e in day_pairs:
            hd = hulls.get(d)
            he = hulls.get(e)

            area_d = hull_areas.get(d, 0.0)
            area_e = hull_areas.get(e, 0.0)

            inter_area = 0.0
            n_d_in_inter = 0
            n_e_in_inter = 0
            n_all_in_inter = 0
            nho_d_to_e = 0.0
            nho_e_to_d = 0.0

            if hd is not None and he is not None:
                inter = hd.intersection(he)

                if not inter.is_empty and inter.area > 0:
                    inter_area = float(inter.area)

                    if area_d > 0:
                        nho_d_to_e = inter_area / area_d

                    if area_e > 0:
                        nho_e_to_d = inter_area / area_e

                    # Count day-d nodes inside intersection area
                    for i in visited[(l, d)]:
                        pt = Point(float(stops[i].xcoord), float(stops[i].ycoord))
                        if inter.covers(pt):
                            n_d_in_inter += 1

                    # Count day-e nodes inside intersection area
                    for i in visited[(l, e)]:
                        pt = Point(float(stops[i].xcoord), float(stops[i].ycoord))
                        if inter.covers(pt):
                            n_e_in_inter += 1

                    # Count all nodes in this week inside intersection area
                    for i in week_ids:
                        pt = Point(float(stops[i].xcoord), float(stops[i].ycoord))
                        if inter.covers(pt):
                            n_all_in_inter += 1

            rows.append({
                "METHOD": label,
                "WEEK": l,
                "DAY_D": d,
                "DAY_E": e,

                "N_D": day_counts[d],
                "N_E": day_counts[e],

                "HULL_D_TYPE": hull_geom_types[d],
                "HULL_E_TYPE": hull_geom_types[e],

                "HULL_AREA_D": area_d,
                "HULL_AREA_E": area_e,
                "HULL_INTERSECTION_AREA": inter_area,

                "NHO_D_TO_E": nho_d_to_e,
                "NHO_E_TO_D": nho_e_to_d,
                "NHO_DIRECTED_SUM": nho_d_to_e + nho_e_to_d,

                "N_D_IN_INTERSECTION": n_d_in_inter,
                "N_E_IN_INTERSECTION": n_e_in_inter,
                "N_D_PLUS_E_IN_INTERSECTION": n_d_in_inter + n_e_in_inter,
                "N_ALL_IN_INTERSECTION": n_all_in_inter,
            })

    return pd.DataFrame(rows)

def compute_hull_overlap_nodes_term(artifacts: dict, p: Dict[int, SchedTuple]) -> float:

    stops = artifacts["stops"]
    visited = _get_visited_by_cell(artifacts, p)

    timecycle = int(artifacts["timecycle"])
    weeks_local = list(range(1, timecycle + 1))

    total = 0.0

    for l in weeks_local:
        visited_set = {d: set(visited[(l, d)]) for d in DAYS_5}
        hulls = {}

        for e in DAYS_5:
            ids_cell = visited[(l, e)]

            if len(ids_cell) < 3:
                hulls[e] = None
                continue

            pts = [(float(stops[i].xcoord), float(stops[i].ycoord)) for i in ids_cell]
            hull = MultiPoint(pts).convex_hull
            hulls[e] = hull if hull.geom_type == "Polygon" and hull.area > 0 else None

        week_ids = set().union(*(visited_set[d] for d in DAYS_5))

        for i in week_ids:
            pt = Point(float(stops[i].xcoord), float(stops[i].ycoord))

            for e in DAYS_5:
                if i in visited_set[e]:
                    continue

                if hulls[e] is not None and hulls[e].covers(pt):
                    total += 1.0

    return total

# Main
def main():
    ARTIFACTS_PATH = Path("baseline_data_store/1004940_artifacts.pkl")
    RESULT_CSV = Path("./results_optimal/1004940_20260527_184639_resultdetail.csv")

    W1 = 1.0
    W2 = 1.0

    artifacts = load_artifacts(ARTIFACTS_PATH)
    baseline_sched: Dict[int, SchedTuple] = artifacts["baseline_sched"]
    p_optimal = build_p_from_result_csv(RESULT_CSV, artifacts)

    # Compute each metric only once
    """nn_b = compute_nn_objective(artifacts, baseline_sched)
    nn_o = compute_nn_objective(artifacts, p_optimal)

    mssc_b = compute_mssc_objective(artifacts, baseline_sched)
    mssc_o = compute_mssc_objective(artifacts, p_optimal)"""

    nho_b = compute_nho_metric(artifacts, baseline_sched)
    nho_o = compute_nho_metric(artifacts, p_optimal)

    rect_b, rect_size_b, overlap_b = compute_rectangle_objective(
        artifacts, baseline_sched, w1=W1, w2=W2
    )
    rect_o, rect_size_o, overlap_o = compute_rectangle_objective(
        artifacts, p_optimal, w1=W1, w2=W2
    )

    rect_overlap_nodes_b = compute_rectangle_overlap_nodes_term(artifacts, baseline_sched)
    rect_overlap_nodes_o = compute_rectangle_overlap_nodes_term(artifacts, p_optimal)

    hull_overlap_nodes_b = compute_hull_overlap_nodes_term(artifacts, baseline_sched)
    hull_overlap_nodes_o = compute_hull_overlap_nodes_term(artifacts, p_optimal)

    # metric-wise comparison
    print("\n[Baseline vs Algorithm Metrics]\n")

    def print_row(metric_name, baseline_value, algorithm_value, fmt):
        if baseline_value != 0:
            improvement = (baseline_value - algorithm_value) / baseline_value * 100
            improvement_str = f"{improvement:.2f}%"
        else:
            improvement_str = "N/A"

        print(
            f"{metric_name:<32} "
            f"Baseline = {format(baseline_value, fmt):>15}   "
            f"Algorithm = {format(algorithm_value, fmt):>15}   "
            f"Improvement = {improvement_str:>10}"
        )

    """print_row("NN", nn_b, nn_o, ",.2f")
    print_row("MSSC", mssc_b, mssc_o, ",.2f")"""
    print_row("NHO", nho_b, nho_o, ",.6f")
    print_row("RECT_SIZE", rect_size_b, rect_size_o, ",.6f")
    """print_row("OVERLAP", overlap_b, overlap_o, ",.6f")
    print_row("RECT_OBJ", rect_b, rect_o, ",.6f")"""
    print_row("#RECT_NODE_OVERLAP", rect_overlap_nodes_b, rect_overlap_nodes_o, ",.0f")
    print_row("#HULL_NODE_OVERLAP", hull_overlap_nodes_b, hull_overlap_nodes_o, ",.0f")

    pairwise_b = compute_pairwise_nho(
        artifacts,
        baseline_sched,
        label="Baseline",
    )

    pairwise_o = compute_pairwise_nho(
        artifacts,
        p_optimal,
        label="Algorithm",
    )

    pairwise_df = pd.concat(
        [pairwise_b, pairwise_o],
        ignore_index=True,
    )

    pairwise_df["METHOD_ORDER"] = pairwise_df["METHOD"].map({
        "Baseline": 0,
        "Algorithm": 1,
    })

    pairwise_df = pairwise_df.sort_values(
        ["WEEK", "DAY_D", "DAY_E", "METHOD_ORDER"]
    ).drop(columns=["METHOD_ORDER"])

    pairwise_df = pairwise_df.reset_index(drop=True)

    out_dir = Path("convex_hull_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    diag_path = out_dir / f"{ARTIFACTS_PATH.stem}_pairwise_nho_again.csv"
    pairwise_df.to_csv(diag_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved pairwise NHO diagnostics: {diag_path}")

if __name__ == "__main__":
    main()