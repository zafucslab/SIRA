#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


STATIONS = [
    "dinghai",
    "kaihua",
    "tonglu",
    "chunan",
    "panan",
    "xihu",
    "longyou",
]

VALUE_COLUMNS = [
    "air_temperature",
    "rainfall",
    "wind_speed",
    "soil_temp_5cm",
    "soil_temp_20cm",
    "soil_moisture_5cm",
    "soil_moisture_20cm",
    "solar_radiation",
    "relative_humidity",
    "sunshine_duration",
]

SOIL_MOISTURE_COLUMNS = [
    "soil_moisture_5cm",
    "soil_moisture_20cm",
]

NONNEGATIVE_COLUMNS = {
    "rainfall",
    "wind_speed",
    "soil_moisture_5cm",
    "soil_moisture_20cm",
    "solar_radiation",
    "relative_humidity",
    "sunshine_duration",
}

TARGET = "soil_moisture_5cm"


def clip_value(column: str, value: float) -> float:
    value = float(value)

    if column in NONNEGATIVE_COLUMNS:
        value = max(0.0, value)

    if column in {
        "soil_moisture_5cm",
        "soil_moisture_20cm",
        "relative_humidity",
    }:
        value = min(100.0, value)

    if column == "sunshine_duration":
        value = min(24.0, value)

    return value


def apply_soil_moisture_qc(df: pd.DataFrame) -> dict:
    report = {}

    for column in SOIL_MOISTURE_COLUMNS:
        values = df[column]

        fraction_mask = (
            values.notna()
            & (values > 0)
            & (values <= 1)
        )
        nonpositive_mask = (
            values.notna()
            & (values <= 0)
        )

        df.loc[fraction_mask, column] = (
            df.loc[fraction_mask, column] * 100.0
        )
        df.loc[nonpositive_mask, column] = np.nan

        below_min_mask = (
            df[column].notna()
            & (df[column] < 5.0)
        )
        df.loc[below_min_mask, column] = np.nan

        report[f"{column}_fraction_to_percent"] = int(
            fraction_mask.sum()
        )
        report[f"{column}_nonpositive_to_nan"] = int(
            nonpositive_mask.sum()
        )
        report[f"{column}_below_5_to_nan"] = int(
            below_min_mask.sum()
        )

    return report


def prepare_station(
    input_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(input_path)

    missing_columns = [
        column
        for column in ["date", *VALUE_COLUMNS]
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{input_path} 缺少列：{missing_columns}"
        )

    original_rows = len(df)

    df = df[["date", *VALUE_COLUMNS]].copy()
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    invalid_dates = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"]).copy()

    for column in VALUE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    qc_report = apply_soil_moisture_qc(df)

    duplicate_rows = int(
        df["date"].duplicated(keep=False).sum()
    )

    raw_mask = df[VALUE_COLUMNS].notna().astype(np.int8)
    raw_mask.insert(0, "date", df["date"])

    values = (
        df.groupby(
            "date",
            as_index=False,
            sort=True,
        )[VALUE_COLUMNS]
        .mean()
        .sort_values("date")
        .reset_index(drop=True)
    )

    mask = (
        raw_mask.groupby(
            "date",
            as_index=False,
            sort=True,
        )[VALUE_COLUMNS]
        .max()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if not values["date"].equals(mask["date"]):
        raise RuntimeError(
            "重复日期聚合后 value 与 mask 日期不一致"
        )

    segment_id = (
        values["date"]
        .diff()
        .dt.days
        .ne(1)
        .fillna(True)
        .cumsum()
    )

    segment_sizes = segment_id.value_counts(
        sort=False
    )
    longest_segment = segment_sizes.idxmax()
    keep = segment_id == longest_segment

    segment_count = int(segment_id.nunique())

    values = values.loc[keep].reset_index(drop=True)
    mask = mask.loc[keep].reset_index(drop=True)

    report = {
        "original_rows": original_rows,
        "invalid_dates": invalid_dates,
        "duplicate_rows": duplicate_rows,
        "contiguous_segments": segment_count,
        "rows_after": len(values),
        "date_start": values["date"].iloc[0],
        "date_end": values["date"].iloc[-1],
        **qc_report,
    }

    return values, mask, report


def split_bounds(n_rows: int) -> dict:
    n_train = int(n_rows * 0.7)
    n_test = int(n_rows * 0.2)
    n_val = n_rows - n_train - n_test

    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_rows),
    }


def training_fallbacks(
    values: pd.DataFrame,
    train_end: int,
) -> tuple[dict, list]:
    fallbacks = {}
    warnings = []

    train = values.iloc[:train_end]

    for column in VALUE_COLUMNS:
        observed = train[column].dropna()

        if column == "rainfall":
            fallback = 0.0
        elif len(observed):
            fallback = float(observed.median())
        else:
            fallback = 0.0
            warnings.append(
                f"{column}: 训练区间没有真实观测，fallback=0"
            )

        fallbacks[column] = clip_value(
            column,
            fallback,
        )

    return fallbacks, warnings


def causal_base_fill(
    values: pd.DataFrame,
    train_end: int,
) -> tuple[pd.DataFrame, dict, list]:
    output = values.copy()
    fallbacks, warnings = training_fallbacks(
        values,
        train_end,
    )

    counts = {}

    for column in VALUE_COLUMNS:
        last_value = None
        count = 0

        for t in range(len(output)):
            value = values.at[t, column]

            if pd.notna(value):
                value = clip_value(column, value)
                output.at[t, column] = value
                last_value = value
                continue

            if column == "rainfall":
                filled_value = 0.0
            elif last_value is not None:
                filled_value = last_value
            else:
                filled_value = fallbacks[column]

            filled_value = clip_value(
                column,
                filled_value,
            )

            output.at[t, column] = filled_value
            last_value = filled_value
            count += 1

        counts[column] = count

    if output[VALUE_COLUMNS].isna().any().any():
        raise RuntimeError(
            "Causal Base 填充后仍存在 NaN"
        )

    return output, counts, warnings


# Preserve the original ordering at equal prior weights across pandas versions.
TIE_ORDER = {
    "air_temperature": ('solar_radiation', 'rainfall', 'wind_speed', 'soil_moisture_5cm'),
    "rainfall": ('air_temperature', 'wind_speed', 'solar_radiation', 'relative_humidity'),
    "wind_speed": ('air_temperature', 'rainfall', 'solar_radiation', 'relative_humidity'),
    "soil_temp_5cm": ('air_temperature', 'soil_temp_20cm', 'solar_radiation', 'rainfall'),
    "soil_temp_20cm": ('air_temperature', 'soil_temp_5cm', 'solar_radiation', 'rainfall'),
    "soil_moisture_5cm": ('rainfall', 'soil_moisture_20cm', 'air_temperature', 'wind_speed'),
    "soil_moisture_20cm": ('rainfall', 'soil_moisture_5cm', 'air_temperature', 'wind_speed'),
    "solar_radiation": ('sunshine_duration', 'air_temperature', 'rainfall', 'wind_speed'),
    "relative_humidity": ('air_temperature', 'rainfall', 'wind_speed', 'soil_temp_5cm'),
    "sunshine_duration": ('solar_radiation', 'air_temperature', 'rainfall', 'wind_speed'),
}


def choose_parents(
    graph: pd.DataFrame,
    target: str,
    top_parents: int,
    min_weight: float,
) -> list[str]:
    scores = (
        graph[target]
        .reindex(VALUE_COLUMNS)
        .fillna(0.0)
        .copy()
    )

    scores.loc[target] = 0.0
    preferred = list(TIE_ORDER[target])
    order = preferred + [name for name in VALUE_COLUMNS if name not in preferred]
    scores = scores.reindex(order).sort_values(ascending=False, kind="stable")

    parents = [
        column
        for column, weight in scores.items()
        if float(weight) >= min_weight
    ][:top_parents]

    if target not in parents:
        parents = [target, *parents]

    return parents


def fit_lorm_models(
    original: pd.DataFrame,
    observed_mask: pd.DataFrame,
    base_filled: pd.DataFrame,
    graph: pd.DataFrame,
    train_end: int,
    top_parents: int,
    min_weight: float,
    lag: int,
    ridge_alpha: float,
) -> tuple[dict, list]:
    models = {}
    report = []

    for target in VALUE_COLUMNS:
        parents = choose_parents(
            graph,
            target,
            top_parents,
            min_weight,
        )

        # rainfall 是事件型变量，不使用平滑回归插补。
        if target == "rainfall":
            models[target] = None
            report.append({
                "variable": target,
                "parents": ",".join(parents),
                "train_samples": 0,
                "model_used": False,
                "reason": "event_variable_zero_fallback",
            })
            continue

        x_rows = []
        y_rows = []

        for t in range(lag, train_end):
            if observed_mask.at[t, target] <= 0:
                continue

            y = original.at[t, target]
            if pd.isna(y):
                continue

            features = []
            valid = True

            for parent in parents:
                for current_lag in range(1, lag + 1):
                    value = base_filled.at[
                        t - current_lag,
                        parent,
                    ]

                    if not np.isfinite(value):
                        valid = False
                        break

                    features.append(float(value))

                if not valid:
                    break

            if valid:
                x_rows.append(features)
                y_rows.append(float(y))

        minimum_samples = max(
            10,
            len(parents) * lag + 2,
        )

        if len(y_rows) < minimum_samples:
            models[target] = None
            report.append({
                "variable": target,
                "parents": ",".join(parents),
                "train_samples": len(y_rows),
                "model_used": False,
                "reason": "insufficient_train_samples",
            })
            continue

        x_train = np.asarray(
            x_rows,
            dtype=np.float64,
        )
        y_train = np.asarray(
            y_rows,
            dtype=np.float64,
        )

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_train)

        model = Ridge(alpha=ridge_alpha)
        model.fit(x_scaled, y_train)

        residual = (
            y_train
            - model.predict(x_scaled)
        )

        models[target] = {
            "parents": parents,
            "scaler": scaler,
            "model": model,
            "lag": lag,
        }

        report.append({
            "variable": target,
            "parents": ",".join(parents),
            "train_samples": len(y_rows),
            "model_used": True,
            "reason": "",
            "train_residual_std": float(
                np.std(residual)
            ),
        })

    return models, report


def lorm_causal_fill(
    original: pd.DataFrame,
    observed_mask: pd.DataFrame,
    base_filled: pd.DataFrame,
    models: dict,
) -> tuple[pd.DataFrame, dict]:
    output = base_filled.copy()
    counts = {
        column: 0
        for column in VALUE_COLUMNS
    }

    for t in range(len(output)):
        for target in VALUE_COLUMNS:
            if observed_mask.at[t, target] > 0:
                output.at[t, target] = original.at[t, target]
                continue

            bundle = models.get(target)

            if bundle is None:
                continue

            lag = bundle["lag"]
            if t < lag:
                continue

            features = []
            valid = True

            for parent in bundle["parents"]:
                for current_lag in range(1, lag + 1):
                    value = output.at[
                        t - current_lag,
                        parent,
                    ]

                    if not np.isfinite(value):
                        valid = False
                        break

                    features.append(float(value))

                if not valid:
                    break

            if not valid:
                continue

            x = np.asarray(
                features,
                dtype=np.float64,
            ).reshape(1, -1)

            prediction = float(
                bundle["model"].predict(
                    bundle["scaler"].transform(x)
                )[0]
            )

            if not np.isfinite(prediction):
                continue

            output.at[t, target] = clip_value(
                target,
                prediction,
            )
            counts[target] += 1

    if output[VALUE_COLUMNS].isna().any().any():
        raise RuntimeError(
            "Safe LORM 填充后仍存在 NaN"
        )

    return output, counts


def save_station(
    root: Path,
    station: str,
    values: pd.DataFrame,
    mask: pd.DataFrame,
) -> None:
    root.mkdir(parents=True, exist_ok=True)

    value_output = values.copy()
    mask_output = mask.copy()

    value_output["date"] = pd.to_datetime(
        value_output["date"]
    ).dt.strftime("%Y-%m-%d")

    mask_output["date"] = pd.to_datetime(
        mask_output["date"]
    ).dt.strftime("%Y-%m-%d")

    mask_output[VALUE_COLUMNS] = (
        mask_output[VALUE_COLUMNS]
        .astype(np.int8)
    )

    value_output.to_csv(
        root / f"{station}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mask_output.to_csv(
        root / f"{station}_mask.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main():
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parents[1]

    parser.add_argument(
        "--source-root",
        default=str(project_root / "dataset/standardized"),
    )
    parser.add_argument(
        "--graph-path",
        default=str(
            project_root
            / "graphs/agriculture_prior.csv"
        ),
    )
    parser.add_argument(
        "--base-root",
        default=str(project_root / "dataset/base"),
    )
    parser.add_argument(
        "--lorm-root", "--retime-root",
        default=str(project_root / "dataset/processed"),
    )
    parser.add_argument(
        "--top-parents",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    source_root = Path(args.source_root)
    base_root = Path(args.base_root)
    lorm_root = Path(args.lorm_root)

    graph = pd.read_csv(
        args.graph_path,
        index_col=0,
    )
    graph = (
        graph.loc[
            VALUE_COLUMNS,
            VALUE_COLUMNS,
        ]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    split_rows = []
    imputation_rows = []
    qc_rows = []

    for station in STATIONS:
        input_path = (
            source_root
            / f"{station}.csv"
        )

        print("\n" + "=" * 100)
        print("站点：", station)
        print("输入：", input_path)

        values, mask, qc_report = prepare_station(
            input_path
        )

        bounds = split_bounds(len(values))
        train_end = bounds["train"][1]

        base_values, base_counts, warnings = (
            causal_base_fill(
                values,
                train_end,
            )
        )

        models, model_report = fit_lorm_models(
            original=values,
            observed_mask=mask,
            base_filled=base_values,
            graph=graph,
            train_end=train_end,
            top_parents=args.top_parents,
            min_weight=args.min_weight,
            lag=args.lag,
            ridge_alpha=args.ridge_alpha,
        )

        lorm_values, lorm_counts = (
            lorm_causal_fill(
                original=values,
                observed_mask=mask,
                base_filled=base_values,
                models=models,
            )
        )

        # 真实观测位置必须完全相同。
        observed = (
            mask[VALUE_COLUMNS]
            .to_numpy(dtype=bool)
        )

        base_array = base_values[
            VALUE_COLUMNS
        ].to_numpy(dtype=float)

        lorm_array = lorm_values[
            VALUE_COLUMNS
        ].to_numpy(dtype=float)

        if not np.allclose(
            base_array[observed],
            lorm_array[observed],
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(
                f"{station}: LORM 修改了真实观测值"
            )

        save_station(
            base_root,
            station,
            base_values,
            mask,
        )
        save_station(
            lorm_root,
            station,
            lorm_values,
            mask,
        )

        for split_name, (start, end) in bounds.items():
            split_mask = mask.iloc[start:end]

            split_rows.append({
                "station": station,
                "split": split_name,
                "rows": end - start,
                "date_start": values["date"].iloc[start],
                "date_end": values["date"].iloc[end - 1],
                "target_observed": int(
                    split_mask[TARGET].sum()
                ),
                "target_missing": int(
                    len(split_mask)
                    - split_mask[TARGET].sum()
                ),
                "all_observed": int(
                    split_mask[VALUE_COLUMNS]
                    .to_numpy()
                    .sum()
                ),
                "all_missing": int(
                    split_mask[VALUE_COLUMNS].size
                    - split_mask[VALUE_COLUMNS]
                    .to_numpy()
                    .sum()
                ),
            })

        model_report_by_variable = {
            row["variable"]: row
            for row in model_report
        }

        for column in VALUE_COLUMNS:
            row = {
                "station": station,
                "variable": column,
                "base_imputed": base_counts[column],
                "lorm_model_imputed": lorm_counts[column],
            }
            row.update(
                model_report_by_variable[column]
            )
            imputation_rows.append(row)

        qc_rows.append({
            "station": station,
            **qc_report,
            "warnings": " | ".join(warnings),
        })

        missing_positions = ~observed
        differing_missing = int(
            np.sum(
                np.abs(
                    base_array[missing_positions]
                    - lorm_array[missing_positions]
                ) > 1e-12
            )
        )

        print("行数：", len(values))
        print(
            "时间范围：",
            values["date"].iloc[0].date(),
            "至",
            values["date"].iloc[-1].date(),
        )
        print(
            "缺失单元格：",
            int(missing_positions.sum()),
        )
        print(
            "LORM 相对 Base 修改的缺失单元格：",
            differing_missing,
        )

    split_df = pd.DataFrame(split_rows)
    imputation_df = pd.DataFrame(imputation_rows)
    qc_df = pd.DataFrame(qc_rows)

    for root in [base_root, lorm_root]:
        split_df.to_csv(
            root / "_split_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        imputation_df.to_csv(
            root / "_imputation_report.csv",
            index=False,
            encoding="utf-8-sig",
        )
        qc_df.to_csv(
            root / "_qc_report.csv",
            index=False,
            encoding="utf-8-sig",
        )

        leakage_audit = {
            "split_before_model_fit": True,
            "split_rule": "70% train / 10% val / 20% test",
            "training_statistics_only": True,
            "ridge_fit_on_train_only": True,
            "scaler_fit_on_train_only": True,
            "bidirectional_interpolation": False,
            "backfill_used": False,
            "validation_used_for_fit": False,
            "test_used_for_fit": False,
            "mask_preserves_original_observation": True,
            "rainfall_policy": (
                "missing rainfall uses zero fallback; "
                "rainfall is not Ridge-imputed"
            ),
            "prior_graph": str(args.graph_path),
            "top_parents": args.top_parents,
            "min_weight": args.min_weight,
            "lag": args.lag,
            "ridge_alpha": args.ridge_alpha,
        }

        with open(
            root / "_leakage_audit.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                leakage_audit,
                file,
                ensure_ascii=False,
                indent=2,
            )

    print("\n生成完成：")
    print(base_root)
    print(lorm_root)


if __name__ == "__main__":
    main()
