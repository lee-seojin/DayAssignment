from pathlib import Path
from typing import Dict, Tuple

from heuristic_solver import load_artifacts
from heuristic_objective import compute_objective

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
