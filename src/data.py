import numpy as np
import pandas as pd

from src.config import DATA_PATH, SEED, TARGET, TYPES

USECOLS = [
    "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]
DTYPES = {
    "step": "int16",
    "type": "category",
    "nameDest": "category",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


def make_synthetic(n=150000, seed=SEED):
    rng = np.random.default_rng(seed)
    step = np.sort(rng.integers(1, 744, n))
    typ = rng.choice(TYPES, size=n, p=[0.22, 0.35, 0.0065, 0.34, 0.0835])
    amount = rng.lognormal(9.3, 1.5, n)
    old_org = rng.lognormal(10.2, 2.0, n) * (rng.random(n) > 0.3)

    outgoing = np.isin(typ, ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"])
    p_fraud = np.where(typ == "TRANSFER", 0.02, np.where(typ == "CASH_OUT", 0.005, 0.0))
    fraud = rng.random(n) < p_fraud

    old_org = np.where(fraud, amount, old_org)
    new_org = np.where(outgoing, np.maximum(old_org - amount, 0), old_org + amount)
    new_org = np.where(fraud, 0.0, new_org)

    is_merchant = np.isin(typ, ["PAYMENT", "DEBIT"])
    dest_id = rng.integers(0, max(n // 3, 10), n)
    name_dest = np.where(is_merchant, "M" + dest_id.astype(str), "C" + dest_id.astype(str))
    old_dest = np.where(is_merchant, 0.0, rng.lognormal(10, 2, n) * (rng.random(n) > 0.4))
    new_dest = np.where(is_merchant, 0.0, old_dest + amount)
    new_dest = np.where(fraud & (typ == "TRANSFER"), 0.0, new_dest)
    old_dest = np.where(fraud & (typ == "TRANSFER"), 0.0, old_dest)

    return pd.DataFrame(
        {
            "step": step.astype("int16"),
            "type": pd.Categorical(typ, categories=TYPES),
            "amount": amount,
            "oldbalanceOrg": old_org,
            "newbalanceOrig": new_org,
            "nameDest": pd.Categorical(name_dest),
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "isFraud": fraud.astype("int8"),
            "isFlaggedFraud": ((typ == "TRANSFER") & (amount > 200_000) & fraud).astype("int8"),
        }
    )


def load_data(path=None, nrows=None, synthetic=False):
    if synthetic:
        print("[data] Using SYNTHETIC data (smoke test only, results are meaningless)")
        df = make_synthetic()
    else:
        path = path or DATA_PATH
        df = pd.read_csv(path, usecols=USECOLS, dtype=DTYPES, nrows=nrows)
        print(f"[data] Loaded {path}: {len(df):,} rows")
    n_fraud = int(df[TARGET].sum())
    print(f"[data] Fraud cases: {n_fraud:,} ({df[TARGET].mean():.3%})")
    return df


def time_split(step, train_frac=0.64, val_frac=0.16):
    step = np.asarray(step)
    q_train = np.quantile(step, train_frac)
    q_val = np.quantile(step, train_frac + val_frac)
    train = step <= q_train
    val = (step > q_train) & (step <= q_val)
    test = step > q_val
    return train, val, test
