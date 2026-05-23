from pathlib import Path
import json
import pandas as pd

from data_type import DAYS_5


def make_rectangle_geojson(input_csv, output_geojson=None):
    input_path = Path(input_csv)
    df = pd.read_csv(input_path)

    if output_geojson is None:
        out_dir = Path("results_qgis")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_geojson = out_dir / f"{input_path.stem}_rectangles.geojson"
    else:
        output_geojson = Path(output_geojson)
        output_geojson.parent.mkdir(parents=True, exist_ok=True)

    features = []

    for d in DAYS_5:
        day_df = df[df[f"AFT_{d}"] == 1].copy()

        if day_df.empty:
            continue

        x_min = float(day_df["XCOORD"].min())
        x_max = float(day_df["XCOORD"].max())
        y_min = float(day_df["YCOORD"].min())
        y_max = float(day_df["YCOORD"].max())

        polygon = [
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min],
            ]
        ]

        features.append({
            "type": "Feature",
            "properties": {
                "DAY": d,
                "N_STOPS": int(len(day_df)),
                "X_MIN": x_min,
                "X_MAX": x_max,
                "Y_MIN": y_min,
                "Y_MAX": y_max,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": polygon,
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(output_geojson, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    return output_geojson


if __name__ == "__main__":
    input_csv = Path("baseline_data_store/1027633_baseline_resultdetail_A.csv")
    output_path = make_rectangle_geojson(input_csv)
    print(f"Saved: {output_path}")