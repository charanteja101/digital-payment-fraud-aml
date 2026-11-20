import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from src.config import SEED

SEARCH_SPACE = {
    "n_estimators": [200, 400, 600],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 5, 10],
    "gamma": [0, 1],
    "reg_lambda": [1, 5, 10],
}


def signed_log1p(x):
    x = np.asarray(x, dtype="float64")
    return np.sign(x) * np.log1p(np.abs(x))


def make_xgb(**params):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
        **params,
    )


def make_logreg():
    return Pipeline(
        [
            ("log", FunctionTransformer(signed_log1p)),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def sample_params(n_iter):
    return list(ParameterSampler(SEARCH_SPACE, n_iter=n_iter, random_state=SEED))


def downsample_negatives(X, y, frac, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    keep = (y == 1) | (rng.random(len(y)) < frac)
    return X[keep], y[keep]


def prior_correct(p, neg_frac):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-9, 1 - 1e-9)
    return p * neg_frac / (p * neg_frac + (1 - p))
