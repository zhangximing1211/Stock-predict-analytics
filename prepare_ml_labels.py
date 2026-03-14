from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build forward-return targets and time-based train/valid splits."
    )
    parser.add_argument(
        "--input",
        default="stock_data_final.csv",
        help="Input CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="ml_data",
        help="Directory for labeled outputs.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Share of unique dates assigned to the train side before horizon filtering.",
    )
    return parser.parse_args()


def build_split_column(date_index: pd.Series, valid_start_idx: int, horizon: int) -> pd.Series:
    split = pd.Series("drop", index=date_index.index, dtype="object")

    train_max_idx = valid_start_idx - horizon - 1
    if train_max_idx >= 0:
        split.loc[date_index <= train_max_idx] = "train"

    valid_max_idx = int(date_index.max()) - horizon
    if valid_start_idx <= valid_max_idx:
        split.loc[(date_index >= valid_start_idx) & (date_index <= valid_max_idx)] = "valid"

    return split


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, parse_dates=["date"])
    original_columns = df.columns.tolist()
    df = df.sort_values(["Name", "date"]).reset_index(drop=True)

    grouped_close = df.groupby("Name")["close"]

    df["target_return_1d"] = grouped_close.shift(-1) / df["close"] - 1
    df["target_return_5d"] = grouped_close.shift(-5) / df["close"] - 1

    df["target_up_1d"] = np.where(
        df["target_return_1d"].notna(),
        (df["target_return_1d"] > 0).astype("int8"),
        np.nan,
    )

    unique_dates = pd.Index(sorted(df["date"].unique()))
    if len(unique_dates) < 10:
        raise ValueError("Not enough unique dates to create a stable time split.")

    valid_start_idx = int(len(unique_dates) * args.train_ratio)
    valid_start_idx = min(max(valid_start_idx, 1), len(unique_dates) - 1)
    valid_start_date = pd.Timestamp(unique_dates[valid_start_idx])

    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    df["date_idx"] = df["date"].map(date_to_idx).astype("int32")

    df["split_1d"] = build_split_column(df["date_idx"], valid_start_idx, horizon=1)
    df["split_5d"] = build_split_column(df["date_idx"], valid_start_idx, horizon=5)

    df["label_available_1d"] = df["target_return_1d"].notna()
    df["label_available_5d"] = df["target_return_5d"].notna()

    output_csv = output_dir / "stock_data_ml_labeled.csv"
    df.drop(columns=["date_idx"]).to_csv(output_csv, index=False)

    next_day_columns = original_columns + ["target_return_1d", "target_up_1d", "split_1d"]
    future_5d_columns = original_columns + ["target_return_5d", "split_5d"]

    next_day_train = output_dir / "stock_data_next_day_train.csv"
    next_day_valid = output_dir / "stock_data_next_day_valid.csv"
    future_5d_train = output_dir / "stock_data_future_5d_train.csv"
    future_5d_valid = output_dir / "stock_data_future_5d_valid.csv"

    df.loc[df["split_1d"] == "train", next_day_columns].to_csv(next_day_train, index=False)
    df.loc[df["split_1d"] == "valid", next_day_columns].to_csv(next_day_valid, index=False)
    df.loc[df["split_5d"] == "train", future_5d_columns].to_csv(future_5d_train, index=False)
    df.loc[df["split_5d"] == "valid", future_5d_columns].to_csv(future_5d_valid, index=False)

    summary = {
        "input_file": str(input_path),
        "output_file": str(output_csv),
        "scenario_files": {
            "next_day_train": str(next_day_train),
            "next_day_valid": str(next_day_valid),
            "future_5d_train": str(future_5d_train),
            "future_5d_valid": str(future_5d_valid),
        },
        "row_count": int(len(df)),
        "column_count": int(len(df.columns) - 1),
        "train_ratio": args.train_ratio,
        "date_min": str(unique_dates[0].date()),
        "date_max": str(unique_dates[-1].date()),
        "n_unique_dates": int(len(unique_dates)),
        "valid_start_date": str(valid_start_date.date()),
        "scenarios": {
            "next_day_after_close": {
                "label_columns": ["target_return_1d", "target_up_1d"],
                "target_definition": "close(t+1) / close(t) - 1 and its > 0 direction label",
                "train_last_feature_date": str(unique_dates[valid_start_idx - 2].date()),
                "valid_first_feature_date": str(unique_dates[valid_start_idx].date()),
                "counts": df["split_1d"].value_counts().sort_index().to_dict(),
            },
            "future_5d_return": {
                "label_columns": ["target_return_5d"],
                "target_definition": "close(t+5) / close(t) - 1",
                "train_last_feature_date": str(unique_dates[valid_start_idx - 6].date()),
                "valid_first_feature_date": str(unique_dates[valid_start_idx].date()),
                "counts": df["split_5d"].value_counts().sort_index().to_dict(),
            },
        },
    }

    summary_path = output_dir / "stock_data_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Labeled dataset written to: {output_csv}")
    print(f"Split summary written to: {summary_path}")
    print(f"Validation starts on: {valid_start_date.date()}")
    print("split_1d counts:")
    print(df["split_1d"].value_counts().sort_index().to_string())
    print("split_5d counts:")
    print(df["split_5d"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
