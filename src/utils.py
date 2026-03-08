import re
from typing import Any

import numpy as np
import pandas as pd


def normalize_name(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: normalize_name(col) for col in df.columns}
    return df.rename(columns=renamed)


def is_binary_series(series: pd.Series) -> bool:
    values = series.dropna().unique()
    if len(values) == 0:
        return False
    normalized = set()
    for value in values:
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"yes", "y", "true"}:
                normalized.add(1)
            elif token in {"no", "n", "false"}:
                normalized.add(0)
            else:
                return False
        elif np.issubdtype(type(value), np.number):
            if value in {0, 1}:
                normalized.add(int(value))
            else:
                return False
        else:
            return False
    return normalized.issubset({0, 1})


def yes_no_to_binary(series: pd.Series) -> pd.Series:
    def mapper(value: Any) -> int:
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"yes", "y", "true"}:
                return 1
            if token in {"no", "n", "false"}:
                return 0
        if pd.isna(value):
            return 0
        try:
            return int(float(value) > 0)
        except (TypeError, ValueError):
            return 0

    return series.map(mapper).astype(int)


def find_label_column(columns: list[str]) -> str | None:
    candidates = [
        "prognosis",
        "disease",
        "diseases",
        "diagnosis",
        "label",
        "target",
        "outcome",
    ]
    for column in columns:
        if column in candidates:
            return column
    for column in columns:
        if any(token in column for token in {"disease", "prognosis", "diagnosis"}):
            return column
    return None

