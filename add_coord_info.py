import pandas as pd

# ── 1. 파일 읽기 ──────────────────────────────────────────────
stops = pd.read_csv(
    "./1027633/stops.txt",
    sep="\t",          # 탭 구분자
    dtype={"custno": str},
)

result = pd.read_csv(
    "./results_optimal/1027633_20260408_144517_resultdetail.csv",
    dtype={"STOP ID": str},
    encoding="utf-8-sig",   # BOM 처리
)

# ── 2. stops에서 필요한 컬럼만 추출 (중복 custno 제거) ────────
coords = (
    stops[["custno", "xcoord", "ycoord"]]
    .drop_duplicates(subset="custno")
)

# ── 3. LEFT JOIN : result 기준으로 좌표 붙이기 ────────────────
merged = result.merge(
    coords,
    left_on="STOP ID",
    right_on="custno",
    how="left",
)

# custno 컬럼은 STOP ID와 동일하므로 제거
merged.drop(columns=["custno"], inplace=True)

# ── 4. 컬럼 순서 조정 : STOP ID 바로 뒤에 xcoord, ycoord 삽입
cols = list(merged.columns)
stop_idx = cols.index("STOP ID") + 1
cols.remove("xcoord")
cols.remove("ycoord")
cols = cols[:stop_idx] + ["xcoord", "ycoord"] + cols[stop_idx:]
merged = merged[cols]

# ── 5. 매칭 결과 요약 출력 ────────────────────────────────────
total      = len(merged)
matched    = merged["xcoord"].notna().sum()
unmatched  = total - matched

print(f"전체 행 수     : {total}")
print(f"좌표 매칭 성공 : {matched}")
print(f"좌표 없음 (NaN): {unmatched}")

if unmatched > 0:
    missing_ids = merged.loc[merged["xcoord"].isna(), "STOP ID"].unique()
    print(f"  → 미매칭 STOP ID 예시: {missing_ids[:10]}")

# ── 6. 저장 ──────────────────────────────────────────────────
output_path = "./results_optimal/1027633_20260408_144517_resultdetail_coord_aggr.csv"
merged.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n저장 완료 → {output_path}")