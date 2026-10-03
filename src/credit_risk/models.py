"""Logistic regression baseline and LightGBM challenger, with the usual credit metrics."""
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import APPLICATION_CATEGORICAL, ID, RANDOM_STATE, TARGET


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cats = [c for c in APPLICATION_CATEGORICAL if c in df]
    nums = [c for c in df.columns if c not in (ID, TARGET, *cats)]
    return nums, cats


def logistic_baseline(nums: list[str], cats: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), nums),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=0.5))])


def lightgbm_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
        random_state=RANDOM_STATE, verbose=-1,
    )


def to_lgb_frame(X: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in cats:
        X[c] = X[c].astype("category")
    return X


@dataclass
class Metrics:
    auc: float
    gini: float
    ks: float

    def as_dict(self) -> dict:
        return {"AUC": round(self.auc, 4), "Gini": round(self.gini, 4), "KS": round(self.ks, 4)}


def credit_metrics(y_true, pd_scores) -> Metrics:
    auc = roc_auc_score(y_true, pd_scores)
    fpr, tpr, _ = roc_curve(y_true, pd_scores)
    return Metrics(auc=auc, gini=2 * auc - 1, ks=float(np.max(tpr - fpr)))


class PlattCalibrator:
    """Maps raw scores to calibrated PDs with a one-feature logistic regression on the score's log-odds.

    Fitted on the validation set so tier PDs, prices and expected losses match observed default rates.
    """

    def fit(self, raw_pd, y):
        self._lr = LogisticRegression(C=1e6).fit(self._logit(raw_pd), y)
        return self

    def transform(self, raw_pd):
        return self._lr.predict_proba(self._logit(raw_pd))[:, 1]

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)
