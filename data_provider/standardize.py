from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path(__file__).resolve().parents[1] / "dataset"
OUT_DIR = Path(__file__).resolve().parents[1] / "dataset/standardized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 和之前实验保持一致的 7 个站点顺序
STATIONS = [
    ("dinghai.xlsx", "dinghai", 0),
    ("kaihua.xlsx", "kaihua", 1),
    ("tonglu.xlsx", "tonglu", 2),
    ("chunan.xlsx", "chunan", 3),
    ("panan.xlsx", "panan", 4),
    ("xihu.xlsx", "xihu", 5),
    ("longyou.xlsx", "longyou", 6),
]

# 注意：solar_radiation 使用旧实验确认过的“太阳辐射量(mj/m2)”
COLUMN_MAP = {
    "大气温度(℃)": "air_temperature",
    "降雨量(mm)": "rainfall",
    "风速": "wind_speed",
    "土温5cm(℃)": "soil_temp_5cm",
    "土温20cm(℃)": "soil_temp_20cm",
    "土湿5cm(%)": "soil_moisture_5cm",
    "土湿20cm(%)": "soil_moisture_20cm",
    "太阳辐射量(mj/m2)": "solar_radiation",
    "空气相对湿度(%)": "relative_humidity",
    "日照时间(h)": "sunshine_duration",
}

COLUMNS = list(COLUMN_MAP.values())

# 基础物理范围 QC：超出范围置 NaN，但不做插值
QC_RANGES = {
    "air_temperature": (-50, 60),
    "rainfall": (0, 500),
    "wind_speed": (0, 80),
    "soil_temp_5cm": (-30, 80),
    "soil_temp_20cm": (-30, 80),
    "soil_moisture_5cm": (0, 100),
    "soil_moisture_20cm": (0, 100),
    "solar_radiation": (0, 40),
    "relative_humidity": (0, 100),
    "sunshine_duration": (0, 24),
}

summary_rows = []

for fname, key, sid in STATIONS:
    path = RAW_DIR / fname
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_excel(path)

    missing_raw_cols = [c for c in COLUMN_MAP if c not in df.columns]
    if missing_raw_cols:
        raise ValueError(f"{path} missing raw columns: {missing_raw_cols}")

    out = pd.DataFrame()

    # 日期列保留，方便后续分析
    if all(c in df.columns for c in ["年份", "月份", "日期"]):
        out["date"] = pd.to_datetime(
            dict(year=df["年份"], month=df["月份"], day=df["日期"]),
            errors="coerce"
        )

    for raw_col, eng_col in COLUMN_MAP.items():
        x = pd.to_numeric(df[raw_col], errors="coerce")

        if eng_col in QC_RANGES:
            lo, hi = QC_RANGES[eng_col]
            x = x.where((x >= lo) & (x <= hi), np.nan)

        out[eng_col] = x

    if "date" in out.columns:
        out = out.sort_values("date").reset_index(drop=True)

    out_csv = OUT_DIR / f"{key}.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n==============================")
    print(f"{key} -> {out_csv}")
    print("==============================")
    print("rows =", len(out))
    miss = out[COLUMNS].isna().sum()
    print(miss.to_string())
    print("TOTAL_MISSING =", int(miss.sum()))

    for c in COLUMNS:
        summary_rows.append({
            "station_key": key,
            "station_id": sid,
            "variable": c,
            "rows": len(out),
            "missing": int(out[c].isna().sum()),
            "missing_rate": float(out[c].isna().mean()),
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "raw_standard_missing_summary.csv", index=False, encoding="utf-8-sig")
print("\nSaved summary to:", OUT_DIR / "raw_standard_missing_summary.csv")
