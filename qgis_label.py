from pathlib import Path
import pandas as pd

from data_type import DAYS_5

from pathlib import Path
import pandas as pd

def make_qgis_label(input_csv, output_csv=None):
    input_path = Path(input_csv)
    df = pd.read_csv(input_path)

    if "WEEK#" in df.columns:
        df = df.drop(columns=["WEEK#"])

    df = df.drop_duplicates()
    out_rows = []

    for _, row in df.iterrows():
        aft_days = [
            d for d in DAYS_5
            if int(row[f"AFT_{d}"]) == 1
        ]

        aft_schedule = f"({row['AFT_WK_CD']}, {'/'.join(aft_days)})"

        for d in aft_days:
            new_row = row.to_dict()
            new_row["DAY"] = d
            new_row["LABEL"] = f"({row['AFT_WK_CD']}, {d})"
            new_row["AFT_SCHEDULE"] = aft_schedule
            out_rows.append(new_row)

    out_df = pd.DataFrame(out_rows)

    out_dir = Path("results_qgis")
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_csv is None:
        output_path = out_dir / f"{input_path.stem}_qgis_ready.csv"
    else:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


if __name__ == "__main__":

    input_csv = Path("results_optimal/1042199_20260505_141711_resultdetail.csv")
    output_path = make_qgis_label(input_csv)
    print(f"Saved: {output_path}")