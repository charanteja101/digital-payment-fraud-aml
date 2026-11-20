import joblib
import numpy as np

from src.config import MODEL_PATH
from src.features import build_features
from src.models import prior_correct

_cache = {}


def _load():
    if "bundle" not in _cache:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found in models/. Train first: python -m src.train --data data/paysim.csv"
            )
        _cache["bundle"] = joblib.load(MODEL_PATH)
    return _cache["bundle"]


def score(df):
    bundle = _load()
    d = df.copy()
    for c in ("newbalanceOrig", "newbalanceDest"):
        if c not in d.columns:
            d[c] = np.nan
    X = build_features(d)
    p = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]
    p = prior_correct(p, bundle["neg_frac"])
    out = d.loc[X.index].copy()
    out["fraud_probability"] = p
    out["flagged"] = (p >= bundle["threshold"]).astype(int)
    return out.sort_index()
