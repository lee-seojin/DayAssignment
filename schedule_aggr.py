from __future__ import annotations

from pathlib import Path
import pandas as pd


# -----------------------------
# Config
# -----------------------------
DATASETS = [
    ("1027633", "1027633_stops_aggregated.csv"),
    ("1042199", "1042199_stops_aggregated.csv"),
]

# dowcd / frequency 정렬 순서
DOWCD_ORDER = ["1", "2", "3", "4", "O", "E", "A"]
FREQ_ORDER = [1, 2, 3, 4, 5]

# 그룹 파일명 안전하게 만들기 위한 매핑
SAFE_DOWCD = {"O": "O", "E": "E", "A": "A", "1": "1", "2": "2", "3": "3", "4": "4"}


# -----------------------------
# Helpers
# -----------------------------
def detect_day_cols(df: pd.DataFrame) -> list[str]:
    """
    aggregated 파일에 들어있는 요일 패턴 컬럼(0/1)을 자동으로 찾는다.
    우선순위:
      1) BEF_MON..BEF_SUN
      2) mon..sun
      3) MON..SUN
    """
    cand_sets = [
        [f"BEF_{d}" for d in ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]],
        ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
        ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"],
    ]
    cols = set(df.columns)
    for cands in cand_sets:
        if all(c in cols for c in cands):
            return cands
    return []


def dayrule_sort_key(df: pd.DataFrame, day_cols: list[str]) -> pd.DataFrame:
    """
    day_cols 기준으로 같은 dayrule끼리 모이도록 정렬.
    (예: freq=1에서 MON-only들이 먼저 쭉, 그 다음 TUE-only들이 쭉…)
    """
    if not day_cols:
        return df
    # 0/1이 아닌 값(예: NaN)이 있을 수 있어서 int로 강제하기 전에 fillna
    for c in day_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df.sort_values(day_cols, ascending=False, kind="mergesort")


def make_order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """dowcd/frequency 정렬을 위해 categorical 지정."""
    df["dowcd"] = df["dowcd"].astype(str).str.strip().str.upper()
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce").astype("Int64")

    df["dowcd"] = pd.Categorical(df["dowcd"], categories=DOWCD_ORDER, ordered=True)
    df["frequency"] = pd.Categorical(df["frequency"], categories=FREQ_ORDER, ordered=True)
    return df


# -----------------------------
# Main
# -----------------------------
def main():
    BASE_DIR = Path.cwd()

    # 결과는 이 폴더 한 곳에만 저장
    out_dir = BASE_DIR / "grouped_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 두 데이터 로드 + 컬럼 기준 concat
    frames = []
    for folder_name, filename in DATASETS:
        path = BASE_DIR / folder_name / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)
        df["dataset_id"] = folder_name  # 원본 추적용 (원하면 지워도 됨)
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True, sort=False)

    # 필수 컬럼 체크
    required = {"dowcd", "frequency"}
    missing = required - set(df_all.columns)
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    # 2) dowcd/frequency 정렬 타입 지정
    df_all = make_order_columns(df_all)

    # dayrule 정렬용 요일컬럼 찾기
    day_cols = detect_day_cols(df_all)

    # 3) (dowcd, frequency) 그룹별 CSV 저장
    total_n = len(df_all)
    summary_rows = []

    grouped = df_all.groupby(["dowcd", "frequency"], observed=True, dropna=False)

    for (dowcd, freq), g in grouped:
        dowcd_str = str(dowcd)
        freq_int = int(freq)

        g2 = g.copy()

        # dayrule이 같은 것끼리 모이도록 정렬 (있으면)
        g2 = dayrule_sort_key(g2, day_cols)

        # 파일명 생성
        dowcd_tag = SAFE_DOWCD.get(dowcd_str, dowcd_str)
        fname = f"dowcd_{dowcd_tag}__freq_{freq_int}.csv"
        out_path = out_dir / fname

        g2.to_csv(out_path, index=False, encoding="utf-8-sig")

        # summary 기록
        cnt = len(g2)
        pct = cnt / total_n if total_n else 0.0
        summary_rows.append(
            {
                "dowcd": dowcd_str,
                "frequency": freq_int,
                "count": cnt,
                "pct_of_total": pct * 100,
                "output_file": out_path.name,
            }
        )

    # 4) summary csv 저장 (정렬 포함)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["dowcd"] = pd.Categorical(summary["dowcd"], categories=DOWCD_ORDER, ordered=True)
        summary["frequency"] = pd.Categorical(summary["frequency"], categories=FREQ_ORDER, ordered=True)
        summary = summary.sort_values(["dowcd", "frequency"]).reset_index(drop=True)

    summary_path = out_dir / "schedule_aggregated.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"[OK] merged rows: {total_n}")
    print(f"[OK] group csvs + summary saved to: {out_dir}")
    if day_cols:
        print(f"[INFO] dayrule sort columns used: {day_cols}")
    else:
        print("[INFO] dayrule columns not found; group files saved without dayrule ordering.")


if __name__ == "__main__":
    main()
