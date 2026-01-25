from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import csv

from helper_funcs import compute_V
from heuristic_aggr import load_artifacts, DATA_SET
from heuristic_phase1 import phase_1
from heuristic_phase2 import phase_2
from data_type import Stop, SchedTuple
from heuristic_objective import compute_objective

def _daybits_to_str(daybits: Tuple[int, ...]) -> str:
    # (mon,tue,wed,thu,fri,sat,sun) -> "1100100"
    return "".join(str(int(x)) for x in daybits)


def _weektuple_to_str(weekt: Tuple[int, ...]) -> str:
    # (1,3,4) -> "1,3,4"
    return ",".join(str(int(x)) for x in weekt)


def save_stops_result_csv(out_path: Path,
                          stops: Dict[int, Stop],
                          p: Dict[int, SchedTuple],
                          baseline_sched: Dict[int, SchedTuple],
                          changed: Dict[int, int]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "stop_id",
        "xcoord", "ycoord",
        "volume", "frequency", "dowcd",
        "baseline_week", "baseline_daybits",
        "chosen_week", "chosen_daybits",
        "changed",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for stop_id, s in stops.items():
            base = baseline_sched[stop_id]
            chosen = p[stop_id]

            base_w, base_d = base
            ch_w, ch_d = chosen

            w.writerow({
                "stop_id": stop_id,
                "xcoord": float(s.xcoord),
                "ycoord": float(s.ycoord),
                "volume": float(s.volume),
                "frequency": int(s.frequency),
                "dowcd": str(s.dowcd),
                "baseline_week": _weektuple_to_str(base_w),
                "baseline_daybits": _daybits_to_str(base_d),
                "chosen_week": _weektuple_to_str(ch_w),
                "chosen_daybits": _daybits_to_str(ch_d),
                "changed": int(changed.get(stop_id, 0)),
            })


def save_objective_csv(out_path: Path,
                       obj_dict: Dict[str, float],
                       w1: float,
                       w2: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["data_set", DATA_SET])
        w.writerow(["w1", w1])
        w.writerow(["w2", w2])
        w.writerow(["density_term", obj_dict["density"]])
        w.writerow(["volume_balance_term", obj_dict["vol_balance"]])
        w.writerow(["objective", obj_dict["obj"]])


def main():
    ARTIFACTS_PATH = Path("baseline_data_store/artifacts.pkl")
    artifacts = load_artifacts(ARTIFACTS_PATH)

    # inputs from artifacts
    stops: Dict[int, Stop] = artifacts["stops"]
    baseline_sched: Dict[int, SchedTuple] = artifacts["baseline_sched"]
    C_max: int = int(artifacts["C_max"])
    Ddist = artifacts["Ddist"]
    priority_map = artifacts["priority_map"]

    # Phase 0: basic state
    p: Dict[int, SchedTuple] = dict(baseline_sched)   # current schedule (baseline start)
    changed: Dict[int, int] = {stop_id: 0 for stop_id in stops.keys()}
    C_used: int = 0
    V = compute_V(stops, p)

    # Phase 1, 2: clustering & relocation
    clusters, nucleus, p, changed, C_used = phase_1(artifacts, p, changed, C_used)
    p, changed, C_used = phase_2(artifacts, p, changed, C_used, clusters, nucleus)

    obj = compute_objective(artifacts, p, w1=1.0, w2=1.0)
    print(f"[OBJ] total={obj['obj']:.6f}")

    out_dir = Path("results_heuristic")
    save_stops_result_csv(out_dir / f"{DATA_SET}_result_stops.csv", stops, p, baseline_sched, changed)
    save_objective_csv(out_dir / f"{DATA_SET}_result_objective.csv", obj, w1=1.0, w2=1.0)

    print(f"{DATA_SET} Saved results to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()