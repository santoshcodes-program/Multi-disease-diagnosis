from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC

from src.config import DEFAULT_RANDOM_STATE, MODELS_DIR, REPORTS_DIR
from src.data_integration import load_and_integrate_datasets, load_descriptions, load_precautions
from src.feature_search import SearchOptimizedFeatureSelector
from src.utils import normalize_columns, normalize_name


@dataclass
class EvalMetrics:
    model_name: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    roc_auc_ovr: float | None


def _stratify_or_none(y: np.ndarray) -> np.ndarray | None:
    _, counts = np.unique(y, return_counts=True)
    return y if counts.min() >= 2 else None


def _balance_data(X: pd.DataFrame, y: np.ndarray, random_state: int) -> tuple[pd.DataFrame, np.ndarray]:
    classes, counts = np.unique(y, return_counts=True)
    # Full SMOTE on hundreds of classes can explode dataset size.
    if len(classes) > 50:
        return X, y

    target_count = int(np.percentile(counts, 75))
    target_count = max(target_count, int(counts.min()))
    target_count = min(target_count, int(counts.max()))
    sampling_strategy = {int(cls): int(target_count) for cls, c in zip(classes, counts) if c < target_count}
    if not sampling_strategy:
        return X, y

    if counts.min() < 2:
        sampler = RandomOverSampler(random_state=random_state, sampling_strategy=sampling_strategy)
    else:
        k_neighbors = max(1, min(5, int(counts.min()) - 1))
        sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return pd.DataFrame(X_resampled, columns=X.columns), y_resampled


def _subsample_for_model(
    X: pd.DataFrame,
    y: np.ndarray,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(X) <= max_rows:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=max_rows,
        random_state=random_state,
        stratify=_stratify_or_none(y),
    )
    return X_sub, y_sub


def _evaluate_model(
    model_name: str,
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[EvalMetrics, dict[str, Any]]:
    y_pred = model.predict(X_test)
    metrics = EvalMetrics(
        model_name=model_name,
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision_macro=float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        f1_macro=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        roc_auc_ovr=None,
    )
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            metrics.roc_auc_ovr = float(roc_auc_score(y_test, y_proba, multi_class="ovr"))
        except Exception:
            metrics.roc_auc_ovr = None

    details = {"classification_report": report, "confusion_matrix": cm}
    return metrics, details


def _train_aux_binary_model(df: pd.DataFrame, model_name: str, random_state: int) -> dict[str, Any]:
    X = df.drop(columns=["target"]).apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["target"].astype(int).to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=1,
    )
    model.fit(X_scaled, y)
    return {
        "name": model_name,
        "features": X.columns.tolist(),
        "scaler": scaler,
        "model": model,
    }


def train_pipeline(
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    random_state: int = DEFAULT_RANDOM_STATE,
    dataset_path: Path | None = None,
    fast_mode: bool = False,
    fast_top_diseases: int = 200,
    fast_max_rows: int = 60000,
) -> dict[str, Any]:
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    integrated = None
    if dataset_path is not None:
        df = normalize_columns(pd.read_csv(dataset_path)).fillna(0)
        if "disease" not in df.columns:
            raise ValueError(f"Input dataset must contain a 'disease' column: {dataset_path}")
        X = df.drop(columns=["disease"]).apply(pd.to_numeric, errors="coerce").fillna(0)
        y_raw = df["disease"].map(normalize_name).astype(str)
    else:
        integrated = load_and_integrate_datasets()
        df = integrated.symptoms_df.copy().fillna(0)
        X = df.drop(columns=["disease"])
        y_raw = df["disease"].map(normalize_name).astype(str)

    if fast_mode:
        disease_counts = y_raw.value_counts()
        keep_diseases = disease_counts.head(fast_top_diseases).index
        keep_mask = y_raw.isin(keep_diseases)
        X = X.loc[keep_mask].reset_index(drop=True)
        y_raw = y_raw.loc[keep_mask].reset_index(drop=True)

        if len(X) > fast_max_rows:
            X, _, y_raw, _ = train_test_split(
                X,
                y_raw,
                train_size=fast_max_rows,
                random_state=random_state,
                stratify=_stratify_or_none(y_raw.to_numpy()),
            )
            X = X.reset_index(drop=True)
            y_raw = y_raw.reset_index(drop=True)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    stratify_target = _stratify_or_none(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=stratify_target
    )

    evaluations: list[EvalMetrics] = []
    eval_details: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, Any] = {}

    if fast_mode:
        selected_features = X_train.columns.tolist()
        feature_selector = None
        rf_model = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=1,
        )
        rf_model.fit(X_train, y_train)
        rf_metrics, rf_details = _evaluate_model("random_forest", rf_model, X_test, y_test)
        evaluations.append(rf_metrics)
        eval_details["random_forest"] = rf_details
        trained_models["random_forest"] = {"model": rf_model, "scaler": None}
    else:
        feature_selector = SearchOptimizedFeatureSelector(random_state=random_state)
        X_train_selected = feature_selector.fit_transform(X_train, y_train)
        X_test_selected = feature_selector.transform(X_test)
        selected_features = feature_selector.selected_features_

        X_train_balanced, y_train_balanced = _balance_data(X_train_selected, y_train, random_state=random_state)

        X_train_svm, y_train_svm = _subsample_for_model(
            X_train_balanced, y_train_balanced, max_rows=20000, random_state=random_state
        )
        X_train_ann, y_train_ann = _subsample_for_model(
            X_train_balanced, y_train_balanced, max_rows=50000, random_state=random_state
        )

        std_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()
        X_train_std = std_scaler.fit_transform(X_train_svm)
        X_test_std = std_scaler.transform(X_test_selected)
        X_train_minmax = minmax_scaler.fit_transform(X_train_ann)
        X_test_minmax = minmax_scaler.transform(X_test_selected)

        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                class_weight="balanced_subsample",
                n_jobs=1,
            ),
            "svm": OneVsRestClassifier(LinearSVC(C=1.0, random_state=random_state)),
            "ann": MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                alpha=1e-4,
                batch_size=64,
                learning_rate_init=1e-3,
                max_iter=400,
                random_state=random_state,
            ),
        }

        # Random Forest uses unscaled sparse-like symptom vectors.
        rf_model = models["random_forest"]
        rf_model.fit(X_train_balanced, y_train_balanced)
        rf_metrics, rf_details = _evaluate_model("random_forest", rf_model, X_test_selected, y_test)
        evaluations.append(rf_metrics)
        eval_details["random_forest"] = rf_details
        trained_models["random_forest"] = {"model": rf_model, "scaler": None}

        svm_model = models["svm"]
        svm_model.fit(X_train_std, y_train_svm)
        svm_metrics, svm_details = _evaluate_model("svm", svm_model, X_test_std, y_test)
        evaluations.append(svm_metrics)
        eval_details["svm"] = svm_details
        trained_models["svm"] = {"model": svm_model, "scaler": std_scaler}

        ann_model = models["ann"]
        ann_model.fit(X_train_minmax, y_train_ann)
        ann_metrics, ann_details = _evaluate_model("ann", ann_model, X_test_minmax, y_test)
        evaluations.append(ann_metrics)
        eval_details["ann"] = ann_details
        trained_models["ann"] = {"model": ann_model, "scaler": minmax_scaler}

    best_eval = max(evaluations, key=lambda m: m.f1_macro)
    best_model_info = trained_models[best_eval.model_name]

    descriptions = load_descriptions()
    precautions = load_precautions()

    artifact = {
        "feature_selector": feature_selector,
        "selected_features": selected_features,
        "label_encoder": label_encoder,
        "model_name": best_eval.model_name,
        "model": best_model_info["model"],
        "scaler": best_model_info["scaler"],
        "all_models": trained_models,
        "descriptions": descriptions,
        "precautions": precautions,
        "metrics": [asdict(e) for e in evaluations],
        "eval_details": eval_details,
    }
    joblib.dump(artifact, models_dir / "multi_disease_model.joblib")

    aux_models: dict[str, str] = {}
    if integrated is not None and integrated.diabetes_df is not None:
        diabetes_artifact = _train_aux_binary_model(integrated.diabetes_df, "diabetes", random_state)
        joblib.dump(diabetes_artifact, models_dir / "diabetes_model.joblib")
        aux_models["diabetes"] = "diabetes_model.joblib"
    if integrated is not None and integrated.heart_df is not None:
        heart_artifact = _train_aux_binary_model(integrated.heart_df, "heart_disease", random_state)
        joblib.dump(heart_artifact, models_dir / "heart_model.joblib")
        aux_models["heart"] = "heart_model.joblib"

    metrics_df = pd.DataFrame([asdict(e) for e in evaluations]).sort_values("f1_macro", ascending=False)
    metrics_df.to_csv(reports_dir / "model_metrics.csv", index=False)
    with (reports_dir / "evaluation_details.json").open("w", encoding="utf-8") as f:
        json.dump(eval_details, f, indent=2)

    summary = {
        "best_model": best_eval.model_name,
        "selected_feature_count": len(selected_features),
        "num_diseases": int(len(label_encoder.classes_)),
        "model_artifact": str(models_dir / "multi_disease_model.joblib"),
        "metrics_csv": str(reports_dir / "model_metrics.csv"),
        "dataset_path": str(dataset_path) if dataset_path is not None else "integrated_from_data_raw",
        "fast_mode": fast_mode,
        "aux_models": aux_models,
    }
    with (reports_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-disease diagnosis models.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional path to a pre-integrated dataset CSV containing symptom columns + disease.",
    )
    parser.add_argument("--fast-mode", action="store_true", help="Train a fast RandomForest-only model.")
    parser.add_argument(
        "--fast-top-diseases",
        type=int,
        default=200,
        help="In fast mode, keep only the top-N most frequent diseases.",
    )
    parser.add_argument(
        "--fast-max-rows",
        type=int,
        default=60000,
        help="In fast mode, cap training dataset rows for speed.",
    )
    args = parser.parse_args()

    summary = train_pipeline(
        random_state=args.random_state,
        dataset_path=Path(args.dataset_path) if args.dataset_path else None,
        fast_mode=args.fast_mode,
        fast_top_diseases=args.fast_top_diseases,
        fast_max_rows=args.fast_max_rows,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
