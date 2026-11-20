import numpy as np
import pandas as pd

from src.config import TYPES

POST_TXN = ["newbalanceOrig", "newbalanceDest", "error_balance_orig", "error_balance_dest", "orig_emptied"]
BALANCE = ["oldbalanceOrg", "oldbalanceDest", "amount_to_balance"] + POST_TXN
FEATURE_SETS = ("full", "pretxn", "no_balance")


def _is_merchant(s):
    if str(s.dtype) == "category":
        flags = np.asarray(s.cat.categories.astype(str).str.startswith("M"))
        return flags[s.cat.codes.values].astype("int8")
    return s.astype(str).str.startswith("M").astype("int8")


def build_features(df):
    d = df.sort_values("step", kind="stable")
    X = pd.DataFrame(index=d.index)
    for t in TYPES:
        X[f"type_{t}"] = (d["type"].astype(str) == t).astype("int8")

    X["amount"] = d["amount"].astype("float64")
    X["log_amount"] = np.log1p(X["amount"])
    X["oldbalanceOrg"] = d["oldbalanceOrg"].astype("float64")
    X["oldbalanceDest"] = d["oldbalanceDest"].astype("float64")
    X["newbalanceOrig"] = d["newbalanceOrig"].astype("float64")
    X["newbalanceDest"] = d["newbalanceDest"].astype("float64")
    X["amount_to_balance"] = (X["amount"] / (X["oldbalanceOrg"] + 1)).clip(upper=1000)
    X["error_balance_orig"] = X["newbalanceOrig"] + X["amount"] - X["oldbalanceOrg"]
    X["error_balance_dest"] = X["oldbalanceDest"] + X["amount"] - X["newbalanceDest"]
    X["orig_emptied"] = ((X["newbalanceOrig"] == 0) & (X["oldbalanceOrg"] > 0)).astype("int8")
    X["dest_is_merchant"] = _is_merchant(d["nameDest"])
    X["hour"] = ((d["step"].astype("int32") - 1) % 24).astype("int8")
    X["day"] = ((d["step"].astype("int32") - 1) // 24).astype("int8")

    if "dest_prior_txn_count" in d.columns:
        X["dest_prior_txn_count"] = d["dest_prior_txn_count"].astype("float64")
        X["dest_prior_inflow"] = d["dest_prior_inflow"].astype("float64")
    else:
        g = d.groupby("nameDest", sort=False, observed=True)
        X["dest_prior_txn_count"] = g.cumcount().astype("float64")
        X["dest_prior_inflow"] = g["amount"].cumsum().astype("float64") - X["amount"]
    return X


def feature_columns(columns, feature_set):
    columns = list(columns)
    if feature_set == "full":
        return columns
    if feature_set == "pretxn":
        return [c for c in columns if c not in POST_TXN]
    if feature_set == "no_balance":
        return [c for c in columns if c not in BALANCE]
    raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
