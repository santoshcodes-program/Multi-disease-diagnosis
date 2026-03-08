from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2

from src.config import DEFAULT_RANDOM_STATE


class SearchOptimizedFeatureSelector:
    def __init__(self, n_features: int | None = None, random_state: int = DEFAULT_RANDOM_STATE):
        self.n_features = n_features
        self.random_state = random_state
        self.selected_features_: list[str] = []
        self.rf_scores_: dict[str, float] = {}
        self.rfe_selected_: set[str] = set()
        self.chi2_scores_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SearchOptimizedFeatureSelector":
        n_total = X.shape[1]
        target_k = self.n_features or max(20, int(n_total * 0.5))
        target_k = max(5, min(target_k, n_total))

        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=self.random_state,
            class_weight="balanced_subsample",
            n_jobs=1,
        )
        rf.fit(X, y)
        importances = rf.feature_importances_
        self.rf_scores_ = {col: float(score) for col, score in zip(X.columns, importances)}
        rf_ranked = [c for c, _ in sorted(self.rf_scores_.items(), key=lambda item: item[1], reverse=True)]
        rf_top = set(rf_ranked[:target_k])

        rfe_estimator = RandomForestClassifier(
            n_estimators=120,
            random_state=self.random_state,
            class_weight="balanced_subsample",
            n_jobs=1,
        )
        rfe = RFE(estimator=rfe_estimator, n_features_to_select=target_k, step=0.1)
        rfe.fit(X, y)
        self.rfe_selected_ = {col for col, keep in zip(X.columns, rfe.support_) if keep}

        chi2_k = min(target_k, X.shape[1])
        chi2_selector = SelectKBest(score_func=chi2, k=chi2_k)
        chi2_selector.fit(X.clip(lower=0), y)
        scores = chi2_selector.scores_
        self.chi2_scores_ = {
            col: float(score) if score is not None and not np.isnan(score) else 0.0
            for col, score in zip(X.columns, scores)
        }
        chi2_ranked = [c for c, _ in sorted(self.chi2_scores_.items(), key=lambda item: item[1], reverse=True)]
        chi2_top = set(chi2_ranked[:chi2_k])

        union_features = rf_top.union(self.rfe_selected_).union(chi2_top)
        if len(union_features) > max(target_k, 10):
            # Keep strongest features by combined normalized scores.
            combined = {}
            rf_max = max(self.rf_scores_.values()) if self.rf_scores_ else 1.0
            chi2_max = max(self.chi2_scores_.values()) if self.chi2_scores_ else 1.0
            for col in union_features:
                score = (self.rf_scores_.get(col, 0.0) / (rf_max or 1.0)) + (
                    self.chi2_scores_.get(col, 0.0) / (chi2_max or 1.0)
                )
                if col in self.rfe_selected_:
                    score += 1.0
                combined[col] = score
            sorted_union = sorted(combined.items(), key=lambda item: item[1], reverse=True)
            union_features = {col for col, _ in sorted_union[: max(target_k, 10)]}

        self.selected_features_ = sorted(union_features)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features_:
            raise ValueError("Feature selector is not fitted.")
        missing = [f for f in self.selected_features_ if f not in X.columns]
        X_work = X.copy()
        for feature in missing:
            X_work[feature] = 0
        return X_work[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)
