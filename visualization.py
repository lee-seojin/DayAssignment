from __future__ import annotations

import os
import re
import ast
from io import StringIO
from typing import Any, List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt


DAYS_5 = ["MON", "TUE", "WED", "THU", "FRI"]
WEEKS = [1, 2, 3, 4]
DAY_TO_IDX = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}


# 1) Robust CSV reader
def read_resultdetail_csv(csv_path: str) -> pd.DataFrame:
    """
    resultdetail.csv 파일 상단에 메타데이터 줄이 있고,
    그 아래에 'stop_id,...' 헤더가 있는 구조를 안전하게 읽는다.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        # 헤더는 보통 stop_id로 시작
        if line.strip().startswith("stop_id"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not find header line starting with 'stop_id' in {csv_path}")

    data = "".join(lines[header_idx:])
    df = pd.read_csv(StringIO(data))

    # 기본 컬럼 체크
    required = ["stop_id", "xcoord", "ycoord", "baseline_week", "baseline_daybits", "chosen_week", "chosen_daybits"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    return df


# 2) Parsers for week/daybits
def _parse_int_list(val: Any) -> List[int]:
    """
    baseline_week / chosen_week가
    - int
    - "1,3"
    - "(1, 3)"
    - "[1, 3]"
    - '"1,3"' (CSV에서 따옴표 중첩)
    등으로 들어와도 [1,3] 형태로 통일.
    """
    if val is None:
        return []
    if isinstance(val, (int, float)) and not pd.isna(val):
        return [int(val)]
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]

    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return []

    # 바깥 따옴표 제거
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # tuple/list literal이면 literal_eval 시도
    if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return [int(x) for x in obj]
            if isinstance(obj, (int, float)):
                return [int(obj)]
        except Exception:
            pass

    # "1,3" / "1,2,3,4"
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    out = []
    for p in parts:
        # 혹시 W1 이런 꼴 있으면 숫자만 추출
        m = re.search(r"\d+", p)
        if m:
            out.append(int(m.group(0)))
    return out


def _parse_daybits_tuple(val: Any) -> Tuple[int, ...]:
    """
    daybits는 (0,1,0,0,1,0,0) 형태.
    CSV에서 문자열로 들어오므로 literal_eval로 튜플/리스트 복원.
    """
    if val is None:
        return (0, 0, 0, 0, 0, 0, 0)
    if isinstance(val, (list, tuple)):
        bits = tuple(int(x) for x in val)
        return bits if len(bits) == 7 else (0, 0, 0, 0, 0, 0, 0)

    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return (0, 0, 0, 0, 0, 0, 0)

    # 바깥 따옴표 제거
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)):
            bits = tuple(int(x) for x in obj)
            return bits if len(bits) == 7 else (0, 0, 0, 0, 0, 0, 0)
    except Exception:
        pass

    # 혹시 "0100100" 같은 문자열이면
    if re.fullmatch(r"[01]{7}", s):
        return tuple(int(ch) for ch in s)

    return (0, 0, 0, 0, 0, 0, 0)


def add_parsed_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["baseline_week_list"] = df["baseline_week"].apply(_parse_int_list)
    df["chosen_week_list"] = df["chosen_week"].apply(_parse_int_list)

    # 컬럼명이 baseline_daybits / chosen_daybits 로 들어온다고 가정
    df["baseline_day_tuple"] = df["baseline_daybits"].apply(_parse_daybits_tuple)
    df["chosen_day_tuple"] = df["chosen_daybits"].apply(_parse_daybits_tuple)

    # changed 없으면 만들어주기(혹시 최적 결과처럼 없을 수 있음)
    if "changed" not in df.columns:
        df["changed"] = 0
    df["changed"] = df["changed"].fillna(0).astype(int)

    return df


# 3) Visit mask builders
def visit_mask(df: pd.DataFrame, week_col: str, day_col: str, week: int, day: str) -> pd.Series:
    """
    df[week_col] = list[int]
    df[day_col]  = tuple[int,...] len 7
    -> (week, day) 방문 여부 boolean mask
    """
    di = DAY_TO_IDX[day]

    m_week = df[week_col].apply(lambda ws: week in ws)
    m_day = df[day_col].apply(lambda bits: len(bits) >= di + 1 and int(bits[di]) == 1)
    return m_week & m_day


# 4) Plotting (baseline left, result right with changed split)
def plot_weekday_pair(
    *,
    df_all: pd.DataFrame,
    df_base_vis: pd.DataFrame,
    df_res_vis: pd.DataFrame,
    solver_name: str,
    week: int,
    day: str,
    out_path: str,
):
    # 전체 bbox 고정
    xmin, xmax = df_all["xcoord"].min(), df_all["xcoord"].max()
    ymin, ymax = df_all["ycoord"].min(), df_all["ycoord"].max()
    xpad = (xmax - xmin) * 0.03 if xmax > xmin else 0.01
    ypad = (ymax - ymin) * 0.03 if ymax > ymin else 0.01
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    ax_l, ax_r = axes

    # LEFT: baseline
    ax_l.scatter(
        df_base_vis["xcoord"],
        df_base_vis["ycoord"],
        s=14,
        alpha=0.70,
        marker="o",
        label="baseline",
    )
    ax_l.set_title(f"BASELINE | Week {week} - {day}")
    ax_l.set_xlabel("Longitude")
    ax_l.set_ylabel("Latitude")

    # RIGHT: solver (changed split)
    df_unch = df_res_vis[df_res_vis["changed"].astype(int) == 0]
    df_chg = df_res_vis[df_res_vis["changed"].astype(int) == 1]

    ax_r.scatter(
        df_unch["xcoord"],
        df_unch["ycoord"],
        s=12,
        alpha=0.55,
        marker="o",
        label="unchanged",
    )
    ax_r.scatter(
        df_chg["xcoord"],
        df_chg["ycoord"],
        s=18,
        alpha=0.90,
        marker="x",
        label="changed",
    )
    ax_r.legend(loc="upper right", frameon=True)
    ax_r.set_title(f"{solver_name} | Week {week} - {day}")
    ax_r.set_xlabel("Longitude")

    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# 5) Main API
def visualize_solver_result(
    csv_path: str,
    out_dir: str,
    solver_name: Optional[str] = None,
):
    df = read_resultdetail_csv(csv_path)
    df = add_parsed_columns(df)

    if solver_name is None:
        # 파일명에서 적당히 뽑기
        base = os.path.basename(csv_path)
        solver_name = os.path.splitext(base)[0]

    # 20장 생성
    for week in WEEKS:
        for day in DAYS_5:
            m_base = visit_mask(df, "baseline_week_list", "baseline_day_tuple", week, day)
            m_res = visit_mask(df, "chosen_week_list", "chosen_day_tuple", week, day)

            df_base_vis = df[m_base]
            df_res_vis = df[m_res]

            out_path = os.path.join(out_dir, f"week{week}_{day}_baseline_vs_{solver_name}.png")
            plot_weekday_pair(
                df_all=df,
                df_base_vis=df_base_vis,
                df_res_vis=df_res_vis,
                solver_name=solver_name,
                week=week,
                day=day,
                out_path=out_path,
            )

    print(f"[Saved] {out_dir} (20 images)")


if __name__ == "__main__":

    CSV_PATH = "results_heuristic/1042199_20260130_092518_heuristic_resultdetail.csv"
    OUT_DIR = "viz_1042199_heuristic"

    visualize_solver_result(
        csv_path=CSV_PATH,
        out_dir=OUT_DIR,
        solver_name="HEURISTIC_1042199",
    )
