from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import time
import csv
from datetime import datetime
import pickle

from optimal_utils import compute_V, make_run_prefix
from heuristic_phase1 import phase_1
from heuristic_phase2 import phase_2
from data_type import Stop, SchedTuple
from heuristic_objective import compute_objective

W1, W2 = 1.0, 1.0
DATA_SET = "1004940" #"1042199" #"1004812"

def load_artifacts(pkl_path: Path) -> dict:
    with pkl_path.open("rb") as f:
        return pickle.load(f)

def _weektuple_to_str(weekt: Tuple[int, ...]) -> str:
    # (1,3,4) -> "1,3,4"
    return ",".join(str(int(x)) for x in weekt)

def save_heuristic_onefile_csv(
    out_path: Path,
    *,
    meta: Dict[str, object],          # 초록 영역 1줄짜리 값들
    stops: Dict[int, Stop],
    p: Dict[int, SchedTuple],
    baseline_sched: Dict[int, SchedTuple],
    changed: Dict[int, int],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ===== green header (초록 영역) =====
    meta_header = [
        "Date",
        "Data set",
        "Objective",
        "Execution Time(s)",
        "w1",
        "w2",
        "kNN_k",
        "Objective_kNN",
    ]

    meta_values = [
        meta["date"],
        meta["dataset"],
        meta["objective"],
        meta["exec_time_s"],
        meta["w1"],
        meta["w2"],
        meta["k"],
        meta["obj_knn"],
    ]

    # ===== stop table header =====
    fieldnames = [
        "stop_id",
        "xcoord", "ycoord",
        "volume", "frequency", "baseline_dowcd",
        "baseline_week", "wccd_flag",
        "baseline_daybits", "dowlockcd",
        "chosen_week", "chosen_daybits",
        "changed",
    ]

    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(meta_header)
        w.writerow(meta_values)
        w.writerow([])  # 빈 줄(선택)

        dw = csv.DictWriter(f, fieldnames=fieldnames)
        dw.writeheader()

        for stop_id, s in stops.items():
            base = baseline_sched[stop_id]
            chosen = p[stop_id]
            base_w, base_d = base
            ch_w, ch_d = chosen

            dw.writerow({
                "stop_id": stop_id,
                "xcoord": float(s.xcoord),
                "ycoord": float(s.ycoord),
                "volume": float(s.volume),
                "frequency": int(s.frequency),
                "baseline_dowcd": str(s.dowcd),
                "baseline_week": _weektuple_to_str(base_w),
                "wccd_flag": int(s.wccd_flag) if s.wccd_flag is not None else 0,
                "baseline_daybits": tuple(int(x) for x in base_d),
                "dowlockcd": int(s.dowlockcd) if s.dowlockcd is not None else 0,
                "chosen_week": _weektuple_to_str(ch_w),
                "chosen_daybits": tuple(int(x) for x in ch_d),
                "changed": int(changed.get(stop_id, 0)),
            })


def main():
    start_time = time.perf_counter()

    ARTIFACTS_PATH = Path(f"baseline_data_store/{DATA_SET}_artifacts.pkl")
    artifacts = load_artifacts(ARTIFACTS_PATH)

    # inputs from artifacts
    stops: Dict[int, Stop] = artifacts["stops"]
    baseline_sched: Dict[int, SchedTuple] = artifacts["baseline_sched"]

    # Phase 0: basic state
    p: Dict[int, SchedTuple] = dict(baseline_sched)   # current schedule (baseline start)
    changed: Dict[int, int] = {stop_id: 0 for stop_id in stops.keys()}
    C_used: int = 0

    # Phase 1, 2: clustering & relocation
    clusters, nucleus, p, changed, C_used = phase_1(artifacts, p, changed, C_used)

    p, changed, C_used = phase_2(artifacts, p, changed, C_used)

    print("[Objective scored by heuristic_objective]")
    obj = compute_objective(artifacts, p, w1=W1, w2=W2)
    print("[Baseline Objective]")
    baseline_obj = compute_objective(artifacts, baseline_sched, W1, W2)

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"[Total Execution Time] ={elapsed:.3f}s")

    run_prefix = make_run_prefix(DATA_SET)
    out_path = Path("../results_heuristic") / f"{run_prefix}_heuristic_resultdetail.csv"

    meta = {
        "date": datetime.now().strftime("%y/%m/%d"),
        "dataset": DATA_SET,
        "solver_name": "Heuristic",
        "objective": float(obj["obj"]),
        "exec_time_s": float(elapsed),
        "w1": W1,
        "w2": W2,
        "k": int(obj["k"]),
        "obj_knn": float(obj["obj_knn"])
    }

    save_heuristic_onefile_csv(
        out_path,
        meta=meta,
        stops=stops,
        p=p,
        baseline_sched=baseline_sched,
        changed=changed,
    )

    print(f"{DATA_SET} Saved")

if __name__ == "__main__":
    main()