from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ENGINEERED_FEATURES = [
    "momentum_3d",
    "momentum_5d",
    "momentum_10d",
    "volatility_5",
    "volatility_10",
    "close_to_sma_5",
    "close_to_sma_10",
    "sma_5_over_10",
    "volume_ratio_5",
    "volume_trend_5",
    "price_range_mean_5",
    "price_range_mean_10",
    "rsi_5_mean",
    "macd_hist_mean_3",
    "obv_change_mean_5",
    "market_return_mean_1d",
    "market_volatility_mean_20",
    "market_up_ratio_1d",
    "relative_return_vs_market",
    "relative_momentum_vs_market",
    "cs_rank_close_to_sma",
    "cs_rank_rsi_14",
    "cs_rank_volatility_20",
    "cs_rank_momentum_20",
    "cs_rank_volume_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build no-leakage engineered features on top of the labeled stock dataset."
    )
    parser.add_argument(
        "--input",
        default="ml_data/stock_data_ml_labeled.csv",
        help="Input CSV produced by prepare_ml_labels.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="ml_data/features",
        help="Directory for feature-engineered outputs.",
    )
    return parser.parse_args()


def grouped_rolling_mean(df: pd.DataFrame, column: str, window: int) -> pd.Series:
    return (
        df.groupby("Name", sort=False)[column]
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )


def grouped_rolling_std(df: pd.DataFrame, column: str, window: int) -> pd.Series:
    return (
        df.groupby("Name", sort=False)[column]
        .rolling(window=window, min_periods=window)
        .std()
        .reset_index(level=0, drop=True)
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Name", "date"]).reset_index(drop=True)
    close_group = df.groupby("Name", sort=False)["close"]
    volume_group = df.groupby("Name", sort=False)["volume"]

    sma_5 = grouped_rolling_mean(df, "close", 5)
    sma_10 = grouped_rolling_mean(df, "close", 10)
    volume_ma_5 = grouped_rolling_mean(df, "volume", 5)
    price_range_mean_5 = grouped_rolling_mean(df, "price_range", 5)
    price_range_mean_10 = grouped_rolling_mean(df, "price_range", 10)
    rsi_5_mean = grouped_rolling_mean(df, "RSI_14", 5)
    macd_hist_mean_3 = grouped_rolling_mean(df, "MACD_hist", 3)
    obv_change_mean_5 = grouped_rolling_mean(df, "OBV_change", 5)

    df["momentum_3d"] = close_group.pct_change(3)
    df["momentum_5d"] = close_group.pct_change(5)
    df["momentum_10d"] = close_group.pct_change(10)
    df["volatility_5"] = grouped_rolling_std(df, "log_return", 5)
    df["volatility_10"] = grouped_rolling_std(df, "log_return", 10)
    df["close_to_sma_5"] = df["close"] / sma_5 - 1.0
    df["close_to_sma_10"] = df["close"] / sma_10 - 1.0
    df["sma_5_over_10"] = sma_5 / sma_10 - 1.0
    df["volume_ratio_5"] = df["volume"] / volume_ma_5
    df["volume_trend_5"] = volume_group.pct_change(5)
    df["price_range_mean_5"] = price_range_mean_5
    df["price_range_mean_10"] = price_range_mean_10
    df["rsi_5_mean"] = rsi_5_mean
    df["macd_hist_mean_3"] = macd_hist_mean_3
    df["obv_change_mean_5"] = obv_change_mean_5

    df["market_return_mean_1d"] = df.groupby("date")["return"].transform("mean")
    df["market_volatility_mean_20"] = df.groupby("date")["volatility_20"].transform("mean")
    df["market_up_ratio_1d"] = df.groupby("date")["return"].transform(lambda s: (s > 0).mean())

    market_momentum_mean_20 = df.groupby("date")["price_momentum_20"].transform("mean")
    df["relative_return_vs_market"] = df["return"] - df["market_return_mean_1d"]
    df["relative_momentum_vs_market"] = df["price_momentum_20"] - market_momentum_mean_20

    df["cs_rank_close_to_sma"] = df.groupby("date")["close_to_SMA"].rank(pct=True)
    df["cs_rank_rsi_14"] = df.groupby("date")["RSI_14"].rank(pct=True)
    df["cs_rank_volatility_20"] = df.groupby("date")["volatility_20"].rank(pct=True)
    df["cs_rank_momentum_20"] = df.groupby("date")["price_momentum_20"].rank(pct=True)
    df["cs_rank_volume_ratio"] = df.groupby("date")["volume_ratio"].rank(pct=True)
    return df


def write_outputs(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "stock_data_ml_features.csv"
    next_day_train = output_dir / "stock_data_next_day_features_train.csv"
    next_day_valid = output_dir / "stock_data_next_day_features_valid.csv"
    future_5d_train = output_dir / "stock_data_future_5d_features_train.csv"
    future_5d_valid = output_dir / "stock_data_future_5d_features_valid.csv"

    df.to_csv(full_path, index=False)
    df.loc[df["split_1d"] == "train"].to_csv(next_day_train, index=False)
    df.loc[df["split_1d"] == "valid"].to_csv(next_day_valid, index=False)
    df.loc[df["split_5d"] == "train"].to_csv(future_5d_train, index=False)
    df.loc[df["split_5d"] == "valid"].to_csv(future_5d_valid, index=False)

    return {
        "full_features": str(full_path),
        "next_day_train": str(next_day_train),
        "next_day_valid": str(next_day_valid),
        "future_5d_train": str(future_5d_train),
        "future_5d_valid": str(future_5d_valid),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    df = pd.read_csv(input_path, parse_dates=["date"])
    df = build_features(df)
    output_paths = write_outputs(df, output_dir)

    summary = {
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "engineered_feature_count": len(ENGINEERED_FEATURES),
        "engineered_features": ENGINEERED_FEATURES,
        "output_files": output_paths,
        "feature_na_counts": {
            feature: int(df[feature].isna().sum()) for feature in ENGINEERED_FEATURES
        },
    }

    summary_path = output_dir / "ml_features_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Feature dataset written to: {output_paths['full_features']}")
    print(f"Feature summary written to: {summary_path}")
    print("New features:")
    for feature in ENGINEERED_FEATURES:
        print(f"- {feature}")


if __name__ == "__main__":
    main()
