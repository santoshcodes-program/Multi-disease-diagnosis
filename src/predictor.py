from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.utils import normalize_name


class MultiDiseasePredictor:
    def __init__(self, artifact_path: Path | None = None, profile_dataset_path: Path | None = None):
        artifact_path = artifact_path or (MODELS_DIR / "multi_disease_model.joblib")
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
        self.artifact = joblib.load(artifact_path)
        self.features = self.artifact["selected_features"]
        self.model = self.artifact["model"]
        self.scaler = self.artifact["scaler"]
        self.label_encoder = self.artifact["label_encoder"]
        self.descriptions = self.artifact.get("descriptions", {})
        self.precautions = self.artifact.get("precautions", {})
        self.disease_profiles: pd.DataFrame | None = None

        if profile_dataset_path is not None and Path(profile_dataset_path).exists():
            self.disease_profiles = self._load_disease_profiles(Path(profile_dataset_path))

    def _load_disease_profiles(self, dataset_path: Path) -> pd.DataFrame | None:
        df = pd.read_csv(dataset_path)
        if "disease" not in df.columns:
            return None

        available_features = [feature for feature in self.features if feature in df.columns]
        if not available_features:
            return None

        work = df[available_features + ["disease"]].copy()
        work[available_features] = work[available_features].apply(pd.to_numeric, errors="coerce").fillna(0)
        profiles = work.groupby("disease")[available_features].mean()
        return profiles

    def available_symptoms(self) -> list[str]:
        clinical_tokens = {
            "age",
            "gender",
            "sex",
            "blood_pressure",
            "bp",
            "glucose",
            "cholesterol",
            "heart_rate",
            "insulin",
            "bmi",
        }
        return sorted([f for f in self.features if f not in clinical_tokens])

    def _build_input_row(
        self,
        symptoms: list[str],
        clinical_features: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        row = {feature: 0 for feature in self.features}
        for symptom in symptoms:
            key = normalize_name(symptom)
            if key in row:
                row[key] = 1

        if clinical_features:
            for key, value in clinical_features.items():
                normalized = normalize_name(key)
                if normalized in row:
                    try:
                        row[normalized] = float(value)
                    except (TypeError, ValueError):
                        row[normalized] = 0
        return pd.DataFrame([row], columns=self.features)

    def predict_top_k(
        self,
        symptoms: list[str],
        clinical_features: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        X = self._build_input_row(symptoms, clinical_features)
        X_values: Any = X
        if self.scaler is not None:
            X_values = self.scaler.transform(X.values)

        class_ids: np.ndarray
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_values)[0]
            class_ids = np.array(getattr(self.model, "classes_", np.arange(len(proba))))
        elif hasattr(self.model, "decision_function"):
            scores = np.asarray(self.model.decision_function(X_values))
            if scores.ndim == 1:
                scores = np.vstack([-scores, scores]).T
            scores_row = scores[0]
            scores_shifted = scores_row - np.max(scores_row)
            exp_scores = np.exp(scores_shifted)
            proba = exp_scores / np.sum(exp_scores)
            class_ids = np.array(getattr(self.model, "classes_", np.arange(len(proba))))
        else:
            predicted = int(self.model.predict(X_values)[0])
            class_ids = np.array(getattr(self.model, "classes_", [predicted]))
            proba = np.zeros(len(class_ids), dtype=float)
            match_indices = np.where(class_ids == predicted)[0]
            if len(match_indices) == 0:
                proba = np.array([1.0], dtype=float)
                class_ids = np.array([predicted])
            else:
                proba[match_indices[0]] = 1.0

        # Re-rank with symptom overlap profile so predictions reflect typed symptoms better.
        if self.disease_profiles is not None and symptoms:
            symptom_keys = [normalize_name(symptom) for symptom in symptoms]
            symptom_keys = [key for key in symptom_keys if key in self.disease_profiles.columns]
            if symptom_keys:
                overlap_scores = np.zeros(len(class_ids), dtype=float)
                for i, class_id in enumerate(class_ids):
                    disease = self.label_encoder.inverse_transform([int(class_id)])[0]
                    if disease in self.disease_profiles.index:
                        overlap_scores[i] = float(self.disease_profiles.loc[disease, symptom_keys].mean())

                if overlap_scores.max() > overlap_scores.min():
                    overlap_scores = (overlap_scores - overlap_scores.min()) / (
                        overlap_scores.max() - overlap_scores.min()
                    )
                    proba = (0.75 * proba) + (0.25 * overlap_scores)
                    proba = np.maximum(proba, 0)
                    total = float(proba.sum())
                    if total > 0:
                        proba = proba / total

        top_indices = np.argsort(proba)[::-1][:top_k]
        predictions = []
        for index in top_indices:
            encoded_label = int(class_ids[index]) if len(class_ids) > index else int(index)
            disease = self.label_encoder.inverse_transform([encoded_label])[0]
            predictions.append(
                {
                    "disease": disease,
                    "probability": float(proba[index]),
                    "description": self.descriptions.get(disease, ""),
                    "precautions": self.precautions.get(disease, []),
                }
            )
        return predictions


class AuxiliaryRiskPredictor:
    def __init__(self, artifact_path: Path):
        if not artifact_path.exists():
            raise FileNotFoundError(artifact_path)
        self.artifact = joblib.load(artifact_path)
        self.model = self.artifact["model"]
        self.scaler = self.artifact["scaler"]
        self.features = self.artifact["features"]
        self.name = self.artifact["name"]

    def predict_risk(self, input_features: dict[str, Any]) -> dict[str, Any]:
        row = []
        for feature in self.features:
            value = input_features.get(feature, input_features.get(normalize_name(feature), 0))
            try:
                row.append(float(value))
            except (TypeError, ValueError):
                row.append(0.0)
        X = np.array([row], dtype=float)
        X_scaled = self.scaler.transform(X)
        probability = float(self.model.predict_proba(X_scaled)[0][1])
        return {"condition": self.name, "risk_probability": probability}


def load_optional_aux_predictors(models_dir: Path = MODELS_DIR) -> dict[str, AuxiliaryRiskPredictor]:
    out: dict[str, AuxiliaryRiskPredictor] = {}
    diabetes_path = models_dir / "diabetes_model.joblib"
    heart_path = models_dir / "heart_model.joblib"
    if diabetes_path.exists():
        out["diabetes"] = AuxiliaryRiskPredictor(diabetes_path)
    if heart_path.exists():
        out["heart"] = AuxiliaryRiskPredictor(heart_path)
    return out
