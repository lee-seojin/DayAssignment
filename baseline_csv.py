from pathlib import Path
import pandas as pd

from helper_funcs import load_artifacts
from data_type import ALL_DAYS_7

DATA_SET = "1027633"
DOWCD_FILTER = ["A"] #None   # 예: None, ["A"], ["A", "O"]

def export_baseline_resultdetail(dataset=DATA_SET, dowcd_filter=DOWCD_FILTER):
    artifacts_path = Path("baseline_data_store") / f"{dataset}_artifacts.pkl"
    artifacts = load_artifacts(artifacts_path)

    stops = artifacts["stops"]
    baseline_sched = artifacts["baseline_sched"]

    if dowcd_filter is not None:
        allowed = {str(x).strip().upper() for x in dowcd_filter}
        stops = {i: s for i, s in stops.items() if str(s.dowcd).strip().upper() in allowed}

    rows = []

    for s in stops.values():
        _, base_day = baseline_sched[s.custno]

        row = {
            "STOP ID": s.custno,
            "PIECES": float(s.qty) if s.qty is not None else None,
            "VOLUME": float(s.volume),
            "BEF_WK_CD": s.dowcd,
            "AFT_WK_CD": s.dowcd,
            "WK_FREQUENCY": int(s.frequency),
            "CHANGED": 0,
            "XCOORD": float(s.xcoord),
            "YCOORD": float(s.ycoord),
        }

        for idx, d in enumerate(ALL_DAYS_7):
            val = int(base_day[idx])
            row[f"BEF_{d}"] = val
            row[f"AFT_{d}"] = val

        rows.append(row)

    df = pd.DataFrame(rows)

    cols = [
        "STOP ID", "PIECES", "VOLUME",
        "BEF_WK_CD", "AFT_WK_CD", "WK_FREQUENCY"
    ] + [f"BEF_{d}" for d in ALL_DAYS_7] \
      + [f"AFT_{d}" for d in ALL_DAYS_7] \
      + ["CHANGED", "XCOORD", "YCOORD"]

    df = df[cols]

    out_dir = Path("baseline_data_store")
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = ""
    if dowcd_filter is not None:
        suffix = "_" + "_".join(sorted({str(x).strip().upper() for x in dowcd_filter}))

    out_path = out_dir / f"{dataset}_baseline_resultdetail{suffix}.csv"

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Loaded artifacts: {artifacts_path}")
    print(f"Stops: {len(stops)}")
    print(f"Saved: {out_path}")

    return out_path


if __name__ == "__main__":
    export_baseline_resultdetail()