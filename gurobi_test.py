from pathlib import Path
from typing import Dict, Tuple
import ast
import pandas as pd

from heuristic_aggr import load_artifacts
from heuristic_objective import compute_objective

WeekTuple = Tuple[int, ...]
DayBits = Tuple[int, int, int, int, int, int, int]
SchedTuple = Tuple[WeekTuple, DayBits]


def parse_tuple_str(x) -> tuple:
    """
    CSV에 "(1, 3)" / "(0, 0, 1, 0, 0, 0, 0)" 같은 형태로 들어있는 걸 tuple로 변환
    """
    if x is None:
        return tuple()
    s = str(x).strip()
    if not s:
        return tuple()
    return ast.literal_eval(s)  # 안전하게 파싱(문자열을 파이썬 리터럴로)


def to_daybits7(t: tuple) -> DayBits:
    """
    (1,0,1,0,1) 처럼 5개면 (sat,sun)=0 붙여서 7개로 맞춤
    """
    if len(t) == 7:
        return tuple(int(v) for v in t)  # type: ignore
    if len(t) == 5:
        return tuple(int(v) for v in t) + (0, 0)  # type: ignore
    raise ValueError(f"daybits length must be 5 or 7, got {len(t)}: {t}")


def build_p_from_gurobi_csv(csv_path: Path) -> Dict[int, SchedTuple]:
    df = pd.read_csv(csv_path, skiprows=3)

    required = {"custno", "week_AFT", "day_AFT"}
    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"CSV missing columns: {missing}")

    p: Dict[int, SchedTuple] = {}
    for _, r in df.iterrows():
        stop_id = int(r["custno"])
        w = parse_tuple_str(r["week_AFT"])
        d_raw = parse_tuple_str(r["day_AFT"])
        week_t: WeekTuple = tuple(int(v) for v in w)
        day_b: DayBits = to_daybits7(tuple(d_raw))
        p[stop_id] = (week_t, day_b)

    return p


def main():
    # 너 프로젝트에서 artifacts.pkl 저장된 위치로만 맞춰주면 됨
    ARTIFACTS_PATH = Path("baseline_data_store/artifacts.pkl")
    GUROBI_CSV = Path("./results_optimal/1042199_20260127_171102_optimal_resultdetail.csv")

    artifacts = load_artifacts(ARTIFACTS_PATH)
    p = build_p_from_gurobi_csv(GUROBI_CSV)

    obj = compute_objective(artifacts, p, w1=1.0, w2=1.0)

    print("[Gurobi solution scored by heuristic_objective]")
    print(f"density      = {obj['density']:.6f}")
    print(f"vol_balance  = {obj['vol_balance']:.6f}")
    print(f"TOTAL        = {obj['obj']:.6f}")


if __name__ == "__main__":
    main()
