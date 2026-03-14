from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
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


SCENARIO_CONFIG = {
    "next_day": {
        "train_path": "ml_data/stock_data_next_day_train.csv",
        "valid_path": "ml_data/stock_data_next_day_valid.csv",
        "target": "target_up_1d",
        "task": "classification",
        "prediction_column": "pred_target_up_1d",
        "probability_column": "pred_proba_up_1d",
    },
    "future_5d": {
        "train_path": "ml_data/stock_data_future_5d_train.csv",
        "valid_path": "ml_data/stock_data_future_5d_valid.csv",
        "target": "target_return_5d",
        "task": "regression",
        "prediction_column": "pred_target_return_5d",
        "probability_column": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple decision tree on the prepared stock datasets."
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_CONFIG),
        required=True,
        help="Which prepared dataset and target definition to use.",
    )
    parser.add_argument(
        "--train-path",
        help="Optional override for the training CSV path.",
    )
    parser.add_argument(
        "--valid-path",
        help="Optional override for the validation CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory where model artifacts will be written.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum tree depth. Keep this small for interpretability.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=500,
        help="Minimum number of samples per leaf.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
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
        help="Validate inputs and print config without importing scikit-learn.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[dict[str, str], Path, Path, Path]:
    config = SCENARIO_CONFIG[args.scenario].copy()
    train_path = Path(args.train_path or config["train_path"]).resolve()
    valid_path = Path(args.valid_path or config["valid_path"]).resolve()
    output_dir = Path(
        args.output_dir or f"model_outputs/{args.scenario}_decision_tree"
    ).resolve()
    return config, train_path, valid_path, output_dir


def load_frame(path: Path, feature_columns: list[str], target_column: str) -> pd.DataFrame:
    meta_columns = ["date", "Name"]
    usecols = meta_columns + feature_columns + [target_column]
    return pd.read_csv(path, usecols=usecols, parse_dates=["date"])


def validate_columns(frame: pd.DataFrame, required_columns: list[str], file_path: Path) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_path}: {missing}")


def build_metrics(task: str, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict[str, float]:
    if task == "classification":
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        }
        if y_score is not None and len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        return metrics

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float(np.mean((y_true > 0) == (y_pred > 0))),
    }
    return metrics


def main() -> None:
    args = parse_args()
    config, train_path, valid_path, output_dir = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": args.random_state,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "dry_run": args.dry_run,
    }

    metadata_path = output_dir / "run_config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return

    try:
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "scikit-learn is not installed. Install it first, then rerun this script."
        ) from exc

    X_train = train_df[args.features]
    X_valid = valid_df[args.features]
    y_train = train_df[config["target"]]
    y_valid = valid_df[config["target"]]

    if config["task"] == "classification":
        model = DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            class_weight="balanced",
        )
    else:
        model = DecisionTreeRegressor(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
        )

    model.fit(X_train, y_train)

    if config["task"] == "classification":
        valid_predictions = model.predict(X_valid)
        valid_scores = model.predict_proba(X_valid)[:, 1]
    else:
        valid_predictions = model.predict(X_valid)
        valid_scores = None

    metrics = build_metrics(
        task=config["task"],
        y_true=y_valid.to_numpy(),
        y_pred=np.asarray(valid_predictions),
        y_score=np.asarray(valid_scores) if valid_scores is not None else None,
    )

    feature_importance = (
        pd.DataFrame(
            {
                "feature": args.features,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    rules_text = export_text(model, feature_names=args.features)
    (output_dir / "tree_rules.txt").write_text(rules_text, encoding="utf-8")

    predictions = valid_df[["date", "Name", config["target"]]].copy()
    predictions[config["prediction_column"]] = valid_predictions
    if config["probability_column"] is not None:
        predictions[config["probability_column"]] = valid_scores
    predictions.to_csv(output_dir / "valid_predictions.csv", index=False)

    metrics_payload = {
        "config": metadata,
        "validation_metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    with (output_dir / "model.pkl").open("wb") as file_obj:
        pickle.dump(model, file_obj)

    print("Training complete.")
    print(json.dumps(metrics_payload, indent=2))
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
