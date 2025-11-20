"""Implements the Task 3 experiments using a Random Forest classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .features import FeatureSet
from .metrics import classification_metrics
from .plotting import plot_metric_curve, plot_param_bar
from .utils import ensure_dir, save_json


def _build_rf(params: Dict, random_state: int) -> RandomForestClassifier:
    rf_params = {
        "n_estimators": params.get("n_estimators", 400),
        "max_depth": params.get("max_depth"),
        "min_samples_split": params.get("min_samples_split", 2),
        "n_jobs": params.get("n_jobs", -1),
        "random_state": random_state,
    }
    return RandomForestClassifier(**rf_params)


def _cross_validate(
    builder: Callable[[int], RandomForestClassifier],
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: int,
    random_state: int,
) -> List[Dict]:
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rows: List[Dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        model = builder(fold_idx)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        rows.append(
            {
                "fold": fold_idx,
                "val_accuracy": float(np.mean(preds == y[val_idx])),
                "val_macro_f1": float(f1_score(y[val_idx], preds, average="macro")),
            }
        )
    return rows


def run_feature_dimension_experiment(
    feature_sets: Sequence[FeatureSet],
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    base_params: Dict,
    output_dir: Path,
    cv_splits: int,
    random_state: int,
) -> pd.DataFrame:
    feature_dir = ensure_dir(output_dir / "task3" / "rf_feature_sweep")
    reports_dir = ensure_dir(feature_dir / "reports")
    models_dir = ensure_dir(feature_dir / "models")
    cv_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    test_metrics: Dict[str, Dict] = {}
    plot_points = []

    for idx, feature_set in enumerate(feature_sets):
        seed_base = random_state + idx * 301
        builder = lambda offset=0, params=base_params, seed=seed_base: _build_rf(
            params, seed + offset
        )
        folds = _cross_validate(
            builder, feature_set.X_train, y_train, cv_splits, seed_base
        )
        for row in folds:
            row.update(
                {
                    "feature_set": feature_set.name,
                    "n_features": feature_set.n_features,
                }
            )
        cv_rows.extend(folds)

        final_model = _build_rf(base_params, seed_base + 999)
        final_model.fit(feature_set.X_train, y_train)
        preds = final_model.predict(feature_set.X_test)
        metrics = classification_metrics(y_test, preds, class_names)
        test_metrics[feature_set.name] = metrics
        joblib.dump(final_model, models_dir / f"rf_{feature_set.name}.joblib", compress=3)
        save_json(metrics, reports_dir / f"{feature_set.name}_test_metrics.json")
        summary_rows.append(
            {
                "feature_set": feature_set.name,
                "n_features": feature_set.n_features,
                "explained_variance": feature_set.explained_variance,
                "mean_val_accuracy": float(np.mean([f["val_accuracy"] for f in folds])),
                "mean_val_macro_f1": float(np.mean([f["val_macro_f1"] for f in folds])),
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
            }
        )
        plot_points.append(
            (feature_set.n_features, metrics["accuracy"], feature_set.name)
        )

    cv_df = pd.DataFrame(cv_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not cv_df.empty:
        cv_df.to_csv(feature_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(feature_dir / "summary.csv", index=False)
    save_json(test_metrics, feature_dir / "test_metrics.json")

    plot_metric_curve(
        xs=[p[0] for p in plot_points],
        ys=[p[1] for p in plot_points],
        labels=[p[2] for p in plot_points],
        xlabel="Feature Dimension",
        ylabel="Test Accuracy",
        title="Random Forest accuracy vs PCA dimensionality",
        output_path=output_dir / "task3" / "plots" / "rf_feature_accuracy.png",
    )

    return summary_df


def run_hparam_experiment(
    feature_set: FeatureSet,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    base_params: Dict,
    hparam_grid: Sequence[Dict],
    output_dir: Path,
    cv_splits: int,
    random_state: int,
) -> pd.DataFrame:
    sweep_dir = ensure_dir(output_dir / "task3" / "rf_hparam_sweep")
    reports_dir = ensure_dir(sweep_dir / "reports")
    models_dir = ensure_dir(sweep_dir / "models")
    cv_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    test_metrics: Dict[str, Dict] = {}
    plot_points = []

    for idx, config in enumerate(hparam_grid):
        name = config.get("name", f"rf_config_{idx}")
        params = {**base_params, **{k: v for k, v in config.items() if k != "name"}}
        seed_base = random_state + idx * 509
        builder = lambda offset=0, p=params, seed=seed_base: _build_rf(p, seed + offset)
        folds = _cross_validate(
            builder, feature_set.X_train, y_train, cv_splits, seed_base
        )
        for row in folds:
            row.update({"config_name": name})
        cv_rows.extend(folds)

        final_model = _build_rf(params, seed_base + 777)
        final_model.fit(feature_set.X_train, y_train)
        preds = final_model.predict(feature_set.X_test)
        metrics = classification_metrics(y_test, preds, class_names)
        test_metrics[name] = metrics
        joblib.dump(final_model, models_dir / f"rf_hparam_{name}.joblib", compress=3)
        save_json(metrics, reports_dir / f"{name}_test_metrics.json")
        summary_rows.append(
            {
                "config_name": name,
                "n_estimators": params.get("n_estimators"),
                "max_depth": params.get("max_depth"),
                "min_samples_split": params.get("min_samples_split"),
                "mean_val_accuracy": float(np.mean([f["val_accuracy"] for f in folds])),
                "mean_val_macro_f1": float(np.mean([f["val_macro_f1"] for f in folds])),
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
            }
        )
        plot_points.append((name, metrics["accuracy"]))

    cv_df = pd.DataFrame(cv_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not cv_df.empty:
        cv_df.to_csv(sweep_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(sweep_dir / "summary.csv", index=False)
    save_json(test_metrics, sweep_dir / "test_metrics.json")

    plot_param_bar(
        labels=[p[0] for p in plot_points],
        values=[p[1] for p in plot_points],
        title="Random Forest test accuracy vs hyper-parameters",
        ylabel="Test Accuracy",
        output_path=output_dir / "task3" / "plots" / "rf_hparam_accuracy.png",
    )

    return summary_df
