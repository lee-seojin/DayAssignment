import pandas as pd
from pathlib import Path

# =========================
# Paths
# =========================

DATA_SET = '1042199'

INPUT_PATH = Path(f"./{DATA_SET}/stops.txt")
OUTPUT_PATH = Path(f"./{DATA_SET}/{DATA_SET}_stops_aggregated.csv")

# =========================
# Key & Columns
# =========================
KEY_COLS = ["custno", "volume", "material_typ"]

DOW_COL = "dow"
WEEK_PATTERN_COL = "dowcd"

# 원래 stops.csv의 컬럼 순서 (dow 위치 중요)
ORIGINAL_ORDER = [
    "custno", "xcoord", "ycoord", "edgeid", "qty",
    "volume", "weight", "sos", "dow", "dowcd",
    "dowlockcd", "wccd_flag", "material_typ", "clusterid"
]

DAY_MAP = {
    "MONDAY": "MON", "TUESDAY": "TUE", "WEDNESDAY": "WED",
    "THURSDAY": "THU", "FRIDAY": "FRI",
    "SATURDAY": "SAT", "SUNDAY": "SUN"
}
ALL_DAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]


def main():

    df = pd.read_csv(INPUT_PATH, sep="\t", engine="python", encoding="utf-8-sig")

    # ---- normalize dow
    df[DOW_COL] = (
        df[DOW_COL]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(DAY_MAP)
    )

    # ---- indicator for visit
    df["_one"] = 1

    # ---- pivot: 요일 방문 여부
    pv = pd.pivot_table(
        df,
        index=KEY_COLS,
        columns=DOW_COL,
        values="_one",
        aggfunc="max",
        fill_value=0
    )

    # ---- 모든 요일 컬럼 보장
    for d in ALL_DAYS:
        if d not in pv.columns:
            pv[d] = 0
    pv = pv[ALL_DAYS]

    pv.columns = [f"BEF_{d}" for d in pv.columns]
    pv = pv.reset_index()

    # ---- frequency (총 방문 횟수)
    freq = (
        df.groupby(KEY_COLS, as_index=False)
        .size()
        .rename(columns={"size": "frequency"})
    )

    # ---- 나머지 메타데이터: first 유지
    meta_cols = [c for c in ORIGINAL_ORDER if c not in ["dow"]]
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta_cols = [c for c in meta_cols if c not in KEY_COLS]

    meta = (
        df[KEY_COLS + meta_cols]
        .groupby(KEY_COLS, as_index=False)
        .first()
    )

    # ---- merge
    out = (
        pv.merge(meta, on=KEY_COLS, how="left")
          .merge(freq, on=KEY_COLS, how="left")
    )

    # =========================
    # 컬럼 순서 재구성 (핵심)
    # =========================
    final_cols = []

    for col in ORIGINAL_ORDER:
        if col == "dow":
            if "frequency" in out.columns:
                final_cols.append("frequency")
            # dow 자리 → BEF_* 컬럼들
            final_cols.extend([f"BEF_{d}" for d in ALL_DAYS])
        elif col in out.columns:
            final_cols.append(col)

    # 나머지 컬럼 (frequency 등)
    for col in out.columns:
        if col not in final_cols:
            final_cols.append(col)

    out = out[final_cols]

    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"{DATA_SET} Saved: {OUTPUT_PATH}")
    print("Final column order:")
    print(out.columns.tolist())


if __name__ == "__main__":
    main()

