from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Iterable, Tuple
from data_type import SchedTuple, WeekTuple, DayBits, ALL_DAYS_7, KEY_COLS, TuplePools, DAY_MAP_MIN, ScheduleView, Stop
from outdated.heuristic_solver import DATA_SET
import itertools
import pickle

import pandas as pd

# loading params.txt
def load_params(params_path: Path) -> dict:
    p = pd.read_csv(params_path, sep="\t").iloc[0]
    base = params_path.parent

    def _resolve(x: str) -> str:
        x = str(x)
        return str((base / x).resolve())

    return {
        "timecycle": int(p["timecycle"]),
        "V_MAX": float(p["maxvolperday"]),
        "G_MAX": float(p["maxweightperday"]),
        "MAX_PCT_DAY_CHANGES": float(p["maxpctdaychanges"]),
        #"stop_file": _resolve(p["stop_file"]),
        #"da_rules_file": _resolve(p["da_rules_file"]),
    }


# loading darules.txt -> dict[freq] = list[DayBits]
def load_darules(darules_path: Path, pools: TuplePools) -> Dict[int, List[DayBits]]:
    df = pd.read_csv(
        darules_path,
        sep="\t",
        usecols=["freq", "mon", "tue", "wed", "thu", "fri", "sat", "sun"],
    )
    df.columns = [c.strip().lower() for c in df.columns]

    out: Dict[int, List[DayBits]] = {}
    for _, r in df.iterrows():
        f = int(r["freq"])
        bits: DayBits = (
            int(r["mon"]), int(r["tue"]), int(r["wed"]), int(r["thu"]),
            int(r["fri"]), int(r["sat"]), int(r["sun"])
        )
        out.setdefault(f, []).append(bits)
    return out


# loading stops: stops.txt -> aggregated DF
def aggregate_stops(stops_path: Path) -> pd.DataFrame:
    df = pd.read_csv(stops_path, sep="\t", encoding="utf-8-sig")

    # dow 정규화 - 공백 지우기, string 타입으로 강제, 대문자로 치환, abbreviation으로 표현 등
    df["dow"] = df["dow"].astype(str).str.strip().str.upper().replace(DAY_MAP_MIN)
    rows = []

    # 같은 stop 단위로 묶기
    for key, g in df.groupby(KEY_COLS):
        custno, volume, material_typ = key

        dow_set = set(g["dow"])
        day_bits = { f"BEF_{d}": int(d in dow_set) for d in ALL_DAYS_7 } #daybits를 dictionary로 만들기
        frequency = len(g)

        first = g.iloc[0] # meta 데이터에서 변하지 않는 xcoord, ycoord, weight 등의 값 저장
        row = {
            "custno": custno,
            "volume": volume,
            "material_typ": material_typ,
            "frequency": frequency,
            **day_bits,
        } # key cols 값과 day, frequency 값을 넣어 기본적인 한 행(row)를 위한 dictionary 만들기

        # 필요한 meta 컬럼만 명시적으로 추가
        for col in ["xcoord", "ycoord", "weight", "dowcd", "qty", "sos", "clusterid", "wccd_flag"]:
            if col in first:
                row[col] = first[col] # meta 데이터에 해당하는 칼럼의 값 row에 add
        rows.append(row)

    return pd.DataFrame(rows)

# dowcd -> week pattern (TimeCycle=4 only) (baseline)
def dowcd_to_week_baseline(dowcd: str, pools: TuplePools) -> WeekTuple:
    s = str(dowcd).strip().upper()
    if s in {"1", "2", "3", "4"}:
        return pools.week([int(s)])
    if s == "O":
        return pools.week([1, 3])
    if s == "E":
        return pools.week([2, 4])
    if s == "A":
        return pools.week([1, 2, 3, 4])
    raise ValueError(f"Unknown dowcd: {dowcd}")


def daybits_from_bef_row(r: pd.Series, pools: TuplePools) -> DayBits:
    bits: DayBits = tuple(int(r[f"BEF_{d}"]) for d in ALL_DAYS_7)  # type: ignore
    return pools.day(bits)


# aggregated DF -> stops dict + baseline_sched
def build_stops_and_baseline(aggr: pd.DataFrame, pools: TuplePools) -> Tuple[Dict[int, Stop], Dict[int, SchedTuple]]:
    stops: Dict[int, Stop] = {}
    baseline_sched: Dict[int, SchedTuple] = {}

    for _, r in aggr.iterrows():
        custno = int(r["custno"])
        freq = int(r["frequency"])
        baseline_day = daybits_from_bef_row(r, pools)
        baseline_week = dowcd_to_week_baseline(str(r["dowcd"]), pools)
        base = pools.schedule(baseline_week, baseline_day)

        s = Stop(
            custno=custno,
            xcoord=float(r["xcoord"]),
            ycoord=float(r["ycoord"]),
            qty=float(r["qty"]) if "qty" in aggr.columns and pd.notna(r.get("qty")) else None,
            volume=float(r["volume"]),
            weight=float(r["weight"]),
            sos=int(r["sos"]) if "sos" in aggr.columns and pd.notna(r.get("sos")) else None,
            dowcd=str(r["dowcd"]).strip().upper(),
            dowlockcd=int(r["dowlockcd"]) if "dowlockcd" in aggr.columns and pd.notna(r.get("dowlockcd")) else None,
            wccd_flag=int(r["wccd_flag"]) if "wccd_flag" in aggr.columns and pd.notna(r.get("wccd_flag")) else None,
            material_typ=str(r["material_typ"]) if "material_typ" in aggr.columns and pd.notna(r.get("material_typ")) else None,
            clusterid=int(r["clusterid"]) if "clusterid" in aggr.columns and pd.notna(r.get("clusterid")) else None,
            frequency=freq,
            baseline=ScheduleView(baseline_week, baseline_day),
        )

        stops[custno] = s
        baseline_sched[custno] = base # 각 stop의 baseline 빠르게 참조하기 위해

    return stops, baseline_sched


# 같은 (dowcd, frequency)를 가진 stop이 feasible schedule option을 매번 다시 생성하지 않도록
def dowcd_to_week_options(dowcd: str, pools: TuplePools) -> List[WeekTuple]:
    s = str(dowcd).strip().upper()
    if s in {"1", "2", "3", "4"}:
        return [pools.week([1]), pools.week([2]), pools.week([3]), pools.week([4])]
    if s in {"O", "E"}:
        return [pools.week([1, 3]), pools.week([2, 4])]
    if s == "A":
        return [pools.week([1, 2, 3, 4])]
    raise ValueError(f"Unknown dowcd: {dowcd}")

def build_feasible_sched(
    pools: TuplePools,
    darules_map: Dict[int, List[DayBits]],
    dowcd_values: Iterable[str],  # <- dowcd_values = set([s.dowcd for s in stops.values()])
    freq_values: Iterable[int],  # <- freq_values = set([s.frequency for s in stops.values()])
) -> Dict[Tuple[str, int], List[SchedTuple]]:

    feasible_sched: Dict[Tuple[str, int], List[SchedTuple]] = {}

    week_opts: Dict[str, List[WeekTuple]] = {}
    for dowcd in [str(x).strip() for x in dowcd_values]:
        week_opts[dowcd] = dowcd_to_week_options(dowcd, pools)

    for dowcd in week_opts:
        for freq in [int(f) for f in freq_values]:
            day_opts = darules_map[freq]
            feasible_sched[(dowcd, freq)] = [
                pools.schedule(w, d) for (w, d) in itertools.product(week_opts[dowcd], day_opts) # 가능한 경우의 수 combination
            ]

    return feasible_sched

def build_priority_map(
    pools: TuplePools,
    darules_map: Dict[int, List[DayBits]],
    dowcd_set: Iterable[str],
    freq_set: Iterable[int],
) -> Dict[Tuple[str, int], int]:
    # priority: (dowcd, freq) 타입별 feasible 옵션 수
    # = (#week options) * (#day options) - 옵션 수가 적을수록 priority 높음 (priority=1이 가장 높음)

    rows = []
    for dowcd in dowcd_set:
        week_opts = dowcd_to_week_options(dowcd, pools)
        week_cnt = len(week_opts)

        for freq in freq_set:
            day_opts = darules_map.get(freq, [])
            day_cnt = len(day_opts)

            total = week_cnt * day_cnt
            rows.append({
                "dowcd": dowcd,
                "frequency": freq,
                "week_options": week_cnt,
                "day_options": day_cnt,
                "total_options": total,
            })

    df = pd.DataFrame(rows)

    df["total_for_sort"] = df["total_options"]
    df = df.sort_values(["total_for_sort", "dowcd", "frequency"]).reset_index(drop=True)
    df["priority"] = range(1, len(df) + 1)

    df["priority"] = df["total_for_sort"].rank(method="dense").astype(int) # 옵션 수 기준으로 동점이면 같은 priority가 되도록 그룹 생성

    priority_map = {
        (row["dowcd"], int(row["frequency"])): int(row["priority"])
        for _, row in df.iterrows()
    }

    return priority_map

# Save / Load Data
def save_artifacts(out_dir: Path, artifacts: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / f"{DATA_SET}_artifacts.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)
    return pkl_path


# main builder
def build_inputs_and_cache(
    params_path: Path,
    stops_path: Path,
    darules_path: Path,
    od_source: Path,
    out_dir: Path,
) -> dict:

    pools = TuplePools()
    params = load_params(params_path)

    """if params["timecycle"] != 4:
        raise ValueError("This pipeline assumes timecycle=4.")"""

    darules_map = load_darules(darules_path, pools)

    aggr = aggregate_stops(stops_path)
    stops, baseline_sched = build_stops_and_baseline(aggr, pools)

    dowcd_set = set([s.dowcd for s in stops.values()])
    freq_set = set([s.frequency for s in stops.values()])
    sched_cache = build_feasible_sched(pools, darules_map, dowcd_set, freq_set)
    priority_map = build_priority_map(pools, darules_map, aggr["dowcd"], aggr["frequency"])

    Ddist = load_od_matrix_flexible(od_source)

    n = len(stops)
    C_max = int(round(params["MAX_PCT_DAY_CHANGES"] / 100.0 * n))

    artifacts = {
        "params": params,
        "timecycle": params["timecycle"],
        "C_max": C_max,
        "pools": pools,
        "darules_map": darules_map,
        "aggr": aggr,
        "stops": stops,
        "baseline_sched": baseline_sched,
        "sched_cache": sched_cache,
        "Ddist": Ddist,
        "priority_map": priority_map
    }

    pkl_path = save_artifacts(out_dir, artifacts)

    # quick sanity info
    print(f"[OK] saved: {pkl_path}")
    print(f"[OK] stops: {len(stops)}")
    print(f"[OK] unique (dowcd,freq): {len(sched_cache)}")
    print(f"[OK] od entries: {len(Ddist)}")
    print(f"[OK] C_max: {C_max}")

    return artifacts

from pathlib import Path
from typing import Dict, Tuple
import math


def resolve_dataset_paths(dataset_name: str) -> dict:
    dataset_dir = Path(f"./{dataset_name}")
    od_file = dataset_dir / "od.txt"
    od_dir = Path(f"./{dataset_name}_OD")

    stops_path = dataset_dir / "stops.txt"
    params_path = dataset_dir / "params.txt"
    darules_path = dataset_dir / "darules.txt"

    if not stops_path.exists():
        raise FileNotFoundError(f"stops.txt not found: {stops_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"params.txt not found: {params_path}")
    if not darules_path.exists():
        raise FileNotFoundError(f"darules.txt not found: {darules_path}")

    if od_file.exists():
        od_source = od_file
    elif od_dir.exists():
        od_source = od_dir
    else:
        raise FileNotFoundError(f"No OD source found for dataset {dataset_name}")

    return {
        "dataset_dir": dataset_dir,
        "stops_path": stops_path,
        "params_path": params_path,
        "darules_path": darules_path,
        "od_source": od_source,
    }


def load_od_matrix_flexible(
    od_source: Path,
    value: str = "dist",
    assume_symmetric: bool = False,
) -> Dict[Tuple[int, int], float]:

    od: Dict[Tuple[int, int], float] = {}

    def _read_one_file(path: Path):
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if "Origin ID" in line or "Route Type" in line:
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue

                try:
                    origin = int(parts[-4])
                    dest = int(parts[-3])
                    time_s = float(parts[-2])
                    dist_c = float(parts[-1])
                except ValueError:
                    continue

                v = dist_c if value == "dist" else time_s
                if not math.isfinite(v):
                    continue

                od[(origin, dest)] = v
                if assume_symmetric:
                    od[(dest, origin)] = v

    od_source = Path(od_source)

    if od_source.is_file():
        _read_one_file(od_source)

    elif od_source.is_dir():
        files = sorted(od_source.glob("od_NoPenalty*.txt"))
        if not files:
            raise FileNotFoundError(f"No od_NoPenalty*.txt files found in {od_source}")
        for f in files:
            _read_one_file(f)

    else:
        raise FileNotFoundError(f"Invalid OD source: {od_source}")

    return od


if __name__ == "__main__":

    paths = resolve_dataset_paths(DATA_SET)
    OUT_DIR = Path("baseline_data_store")

    build_inputs_and_cache(
        paths["params_path"],
        paths["stops_path"],
        paths["darules_path"],
        paths["od_source"],
        OUT_DIR,
    )
