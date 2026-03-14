from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from build_ml_features import ENGINEERED_FEATURES


BASE_FEATURES = [
    "return",
    "log_return",
    "close_to_SMA",
    "RSI_14",
    "MACD",
    "MACD_hist",
    "BB_width",
    "volatility_20",
    "price_momentum_20",
    "volume_ratio",
    "close_position",
    "return_lag_1",
    "daily_return",
    "price_range",
]

# Market-level features to exclude (same value for all stocks on a given day,
# they dominate importance but add no stock-selection power).
MARKET_FEATURES = {
    "market_return_mean_1d",
    "market_volatility_mean_20",
    "market_up_ratio_1d",
}

DEFAULT_FEATURES = BASE_FEATURES + [
    f for f in ENGINEERED_FEATURES if f not in MARKET_FEATURES
]


SCENARIO_CONFIG = {
    "next_day": {
        "train_path": "ml_data/features/stock_data_next_day_features_train.csv",
        "valid_path": "ml_data/features/stock_data_next_day_features_valid.csv",
        "target": "target_up_1d",
        "task": "classification",
        "prediction_column": "pred_target_up_1d",
        "score_column": "pred_proba_up_1d",
    },
    "future_5d": {
        "train_path": "ml_data/features/stock_data_future_5d_features_train.csv",
        "valid_path": "ml_data/features/stock_data_future_5d_features_valid.csv",
        "target": "target_return_5d",
        "task": "regression",
        "prediction_column": "pred_target_return_5d",
        "score_column": "pred_target_return_5d",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a random forest on the engineered stock feature datasets."
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_CONFIG),
        required=True,
        help="Which prepared dataset and target definition to use.",
    )
    parser.add_argument("--train-path", help="Optional override for the training CSV path.")
    parser.add_argument("--valid-path", help="Optional override for the validation CSV path.")
    parser.add_argument("--output-dir", help="Directory for model artifacts.")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees in the forest.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum depth of each tree.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=300,
        help="Minimum number of samples per leaf.",
    )
    parser.add_argument(
        "--max-features",
        default="0.3",
        help="Feature sampling strategy: sqrt, log2, none, int, or float.",
    )
    parser.add_argument(
        "--max-samples",
        type=float,
        default=0.5,
        help="Share of rows sampled for each tree when bootstrap is enabled.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for fitting the forest.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Feature columns to train on.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print config without training.",
    )
    return parser.parse_args()


def parse_max_features(value: str) -> str | int | float | None:
    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered in {"sqrt", "log2"}:
        return lowered
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported max_features value: {value}") from exc


def resolve_paths(args: argparse.Namespace) -> tuple[dict[str, str], Path, Path, Path]:
    config = SCENARIO_CONFIG[args.scenario].copy()
    train_path = Path(args.train_path or config["train_path"]).resolve()
    valid_path = Path(args.valid_path or config["valid_path"]).resolve()
    output_dir = Path(args.output_dir or f"model_outputs/{args.scenario}_random_forest").resolve()
    return config, train_path, valid_path, output_dir


def load_frame(path: Path, feature_columns: list[str], target_column: str) -> pd.DataFrame:
    columns = list(dict.fromkeys(["date", "Name", *feature_columns, target_column]))
    return pd.read_csv(path, usecols=columns, parse_dates=["date"])


def validate_columns(frame: pd.DataFrame, required_columns: list[str], file_path: Path) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_path}: {missing}")


def build_daily_selection_table(
    frame: pd.DataFrame,
    score_column: str,
    target_column: str,
    quantile: float = 0.1,
) -> pd.DataFrame:
    rows = []
    for date, group in frame.groupby("date", sort=True):
        if group.empty:
            continue
        selection_size = max(1, int(np.ceil(len(group) * quantile)))
        top_target_mean = group.nlargest(selection_size, score_column)[target_column].mean()
        bottom_target_mean = group.nsmallest(selection_size, score_column)[target_column].mean()
        rows.append(
            {
                "date": date,
                "selection_size": selection_size,
                "top_bucket_target_mean": float(top_target_mean),
                "bottom_bucket_target_mean": float(bottom_target_mean),
                "spread": float(top_target_mean - bottom_target_mean),
            }
        )
    return pd.DataFrame(rows)


def mean_daily_rank_ic(frame: pd.DataFrame, score_column: str, target_column: str) -> float:
    rank_ics = []
    for _, group in frame.groupby("date", sort=True):
        if group[score_column].nunique() < 2 or group[target_column].nunique() < 2:
            continue
        corr = group[score_column].corr(group[target_column], method="spearman")
        if pd.notna(corr):
            rank_ics.append(float(corr))
    if not rank_ics:
        return float("nan")
    return float(np.mean(rank_ics))


def build_metrics(
    task: str,
    predictions: pd.DataFrame,
    target_column: str,
    prediction_column: str,
    score_column: str,
) -> dict[str, float]:
    y_true = predictions[target_column].to_numpy()
    y_pred = predictions[prediction_column].to_numpy()
    score = predictions[score_column].to_numpy()
    daily_selection = build_daily_selection_table(predictions, score_column, target_column)

    if task == "classification":
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "overall_up_rate": float(np.mean(y_true)),
            "top_decile_up_rate": float(daily_selection["top_bucket_target_mean"].mean()),
            "bottom_decile_up_rate": float(daily_selection["bottom_bucket_target_mean"].mean()),
            "top_bottom_up_rate_spread": float(daily_selection["spread"].mean()),
        }
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, score))
        return metrics

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float(np.mean((y_true > 0) == (y_pred > 0))),
        "mean_daily_rank_ic": mean_daily_rank_ic(predictions, score_column, target_column),
        "top_decile_mean_return": float(daily_selection["top_bucket_target_mean"].mean()),
        "bottom_decile_mean_return": float(daily_selection["bottom_bucket_target_mean"].mean()),
        "top_bottom_return_spread": float(daily_selection["spread"].mean()),
    }
    return metrics


def main() -> None:
    args = parse_args()
    config, train_path, valid_path, output_dir = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_features = parse_max_features(args.max_features)
    required_columns = args.features + [config["target"]]
    train_df = load_frame(train_path, args.features, config["target"])
    valid_df = load_frame(valid_path, args.features, config["target"])
    validate_columns(train_df, required_columns, train_path)
    validate_columns(valid_df, required_columns, valid_path)

    train_df = train_df.dropna(subset=args.features + [config["target"]]).copy()
    valid_df = valid_df.dropna(subset=args.features + [config["target"]]).copy()

    if train_df.empty or valid_df.empty:
        raise ValueError("Training or validation data is empty after dropping missing values.")

    metadata = {
        "scenario": args.scenario,
        "task": config["task"],
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "output_dir": str(output_dir),
        "features": args.features,
        "target": config["target"],
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": max_features,
        "max_samples": args.max_samples,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "dry_run": args.dry_run,
    }
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return

    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "scikit-learn is not installed. Install it first, then rerun this script."
        ) from exc

    X_train = train_df[args.features]
    X_valid = valid_df[args.features]
    y_train = train_df[config["target"]]
    y_valid = valid_df[config["target"]]

    if config["task"] == "classification":
        y_train = y_train.astype(int)
        y_valid = y_valid.astype(int)
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=max_features,
            max_samples=args.max_samples,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            class_weight="balanced_subsample",
            oob_score=True,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=max_features,
            max_samples=args.max_samples,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            oob_score=True,
        )

    model.fit(X_train, y_train)

    predictions = valid_df[["date", "Name", config["target"]]].copy()
    if config["task"] == "classification":
        predictions[config["prediction_column"]] = model.predict(X_valid)
        predictions[config["score_column"]] = model.predict_proba(X_valid)[:, 1]
    else:
        predictions[config["prediction_column"]] = model.predict(X_valid)
        predictions[config["score_column"]] = predictions[config["prediction_column"]]

    metrics = build_metrics(
        task=config["task"],
        predictions=predictions,
        target_column=config["target"],
        prediction_column=config["prediction_column"],
        score_column=config["score_column"],
    )
    metrics["oob_score"] = float(model.oob_score_)

    feature_importance = (
        pd.DataFrame({"feature": args.features, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    predictions.to_csv(output_dir / "valid_predictions.csv", index=False)

    daily_selection = build_daily_selection_table(
        predictions,
        score_column=config["score_column"],
        target_column=config["target"],
    )
    daily_selection.to_csv(output_dir / "daily_selection_metrics.csv", index=False)

    metrics_payload = {
        "config": metadata,
        "validation_metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    with (output_dir / "model.pkl").open("wb") as file_obj:
        pickle.dump(model, file_obj)

    print("Training complete.")
    print(json.dumps(metrics_payload, indent=2))
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
