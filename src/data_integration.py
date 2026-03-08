from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR
from src.utils import find_label_column, is_binary_series, normalize_columns, normalize_name, yes_no_to_binary


METADATA_FILES = {
    "symptom_severity.csv",
    "description.csv",
    "precautions.csv",
}

DIABETES_FEATURE_HINTS = {
    "pregnancies",
    "glucose",
    "bloodpressure",
    "skinthickness",
    "insulin",
    "bmi",
    "diabetespedigreefunction",
    "age",
}

HEART_FEATURE_HINTS = {
    "cp",
    "thalach",
    "oldpeak",
    "ca",
    "thal",
    "trestbps",
    "chol",
    "fbs",
    "exang",
}


@dataclass
class IntegratedData:
    symptoms_df: pd.DataFrame
    diabetes_df: pd.DataFrame | None
    heart_df: pd.DataFrame | None


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_columns(df)


def _classify_dataset(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    raw_cols = {c.replace("_", "") for c in cols}

    if len(raw_cols.intersection(DIABETES_FEATURE_HINTS)) >= 6 and ("outcome" in cols or "target" in cols):
        return "diabetes"
    if len(raw_cols.intersection(HEART_FEATURE_HINTS)) >= 5 and ("target" in cols or "output" in cols):
        return "heart"

    label_col = find_label_column(list(cols))
    if label_col is not None:
        return "symptom"
    return "unknown"


def _symptom_df_from_binary(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    symptom_cols = [c for c in df.columns if c != label_col]
    usable_cols = []
    for col in symptom_cols:
        if is_binary_series(df[col]):
            usable_cols.append(col)
    if not usable_cols:
        return pd.DataFrame()

    out = df[usable_cols].copy()
    for col in usable_cols:
        out[col] = yes_no_to_binary(out[col])
    out["disease"] = df[label_col].astype(str).map(normalize_name)
    rows.append(out)
    return pd.concat(rows, ignore_index=True)


def _symptom_df_from_slots(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    symptom_cols = [c for c in df.columns if c != label_col]
    if not symptom_cols:
        return pd.DataFrame()

    records = []
    for _, row in df.iterrows():
        disease = normalize_name(row[label_col])
        active = {}
        for col in symptom_cols:
            value = row[col]
            if pd.isna(value):
                continue
            token_text = str(value)
            if not token_text.strip():
                continue
            for token in token_text.split(","):
                symptom = normalize_name(token)
                if symptom and symptom not in {"none", "nan", "null"}:
                    active[symptom] = 1
        if active:
            active["disease"] = disease
            records.append(active)

    if not records:
        return pd.DataFrame()
    out = pd.DataFrame(records).fillna(0).astype({k: int for k in pd.DataFrame(records).columns if k != "disease"})
    return out


def _build_symptom_dataset(df: pd.DataFrame) -> pd.DataFrame:
    label_col = find_label_column(df.columns.tolist())
    if label_col is None:
        return pd.DataFrame()

    binary_ratio = 0.0
    features = [c for c in df.columns if c != label_col]
    if features:
        binary_ratio = sum(1 for c in features if is_binary_series(df[c])) / len(features)

    if binary_ratio >= 0.4:
        out = _symptom_df_from_binary(df, label_col)
        if not out.empty:
            return out
    return _symptom_df_from_slots(df, label_col)


def _prepare_diabetes(df: pd.DataFrame) -> pd.DataFrame:
    label_col = "outcome" if "outcome" in df.columns else "target"
    features = [c for c in df.columns if c != label_col]
    out = df[features + [label_col]].copy().fillna(0)
    out = out.rename(columns={label_col: "target"})
    return out


def _prepare_heart(df: pd.DataFrame) -> pd.DataFrame:
    label_col = "target" if "target" in df.columns else "output"
    features = [c for c in df.columns if c != label_col]
    out = df[features + [label_col]].copy().fillna(0)
    out = out.rename(columns={label_col: "target"})
    return out


def load_and_integrate_datasets(raw_data_dir: Path = RAW_DATA_DIR) -> IntegratedData:
    csv_files = sorted(raw_data_dir.glob("*.csv"))
    symptom_frames: list[pd.DataFrame] = []
    diabetes_df: pd.DataFrame | None = None
    heart_df: pd.DataFrame | None = None

    for file in csv_files:
        if file.name.lower() in METADATA_FILES:
            continue
        df = _read_csv(file)
        dataset_kind = _classify_dataset(df)
        if dataset_kind == "symptom":
            converted = _build_symptom_dataset(df)
            if not converted.empty:
                symptom_frames.append(converted)
        elif dataset_kind == "diabetes":
            diabetes_df = _prepare_diabetes(df)
        elif dataset_kind == "heart":
            heart_df = _prepare_heart(df)

    if not symptom_frames:
        raise FileNotFoundError(
            "No usable symptom datasets were found in data/raw. "
            "Add CSV files containing symptom->disease mappings."
        )

    unified = pd.concat(symptom_frames, ignore_index=True).fillna(0)
    unified = unified.loc[:, ~unified.columns.duplicated()]
    feature_cols = [c for c in unified.columns if c != "disease"]
    unified[feature_cols] = unified[feature_cols].astype(int)
    unified = unified[feature_cols + ["disease"]]
    unified = unified.drop_duplicates().reset_index(drop=True)

    return IntegratedData(symptoms_df=unified, diabetes_df=diabetes_df, heart_df=heart_df)


def load_descriptions(raw_data_dir: Path = RAW_DATA_DIR) -> dict[str, str]:
    path = raw_data_dir / "description.csv"
    if not path.exists():
        return {}

    df = normalize_columns(pd.read_csv(path))
    disease_col = find_label_column(df.columns.tolist()) or df.columns[0]
    description_col = "description" if "description" in df.columns else df.columns[-1]
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        out[normalize_name(row[disease_col])] = str(row[description_col])
    return out


def load_precautions(raw_data_dir: Path = RAW_DATA_DIR) -> dict[str, list[str]]:
    path = raw_data_dir / "precautions.csv"
    if not path.exists():
        return {}

    df = normalize_columns(pd.read_csv(path))
    disease_col = find_label_column(df.columns.tolist()) or df.columns[0]
    precaution_cols = [c for c in df.columns if c != disease_col]
    out: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        precautions = [str(row[c]).strip() for c in precaution_cols if pd.notna(row[c]) and str(row[c]).strip()]
        out[normalize_name(row[disease_col])] = precautions
    return out
