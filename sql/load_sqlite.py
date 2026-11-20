import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

COLUMNS = {
    "step": "step",
    "type": "type",
    "amount": "amount",
    "nameOrig": "name_orig",
    "oldbalanceOrg": "old_balance_orig",
    "newbalanceOrig": "new_balance_orig",
    "nameDest": "name_dest",
    "oldbalanceDest": "old_balance_dest",
    "newbalanceDest": "new_balance_dest",
    "isFraud": "is_fraud",
    "isFlaggedFraud": "is_flagged_fraud",
}


def main():
    ap = argparse.ArgumentParser(description="Load the PaySim CSV (and optional model scores) into SQLite")
    ap.add_argument("--data", default=str(ROOT / "data" / "paysim.csv"))
    ap.add_argument("--scores", default=str(ROOT / "dashboard" / "powerbi_data" / "scored_test.csv"))
    ap.add_argument("--db", default=str(ROOT / "sql" / "paysim.db"))
    ap.add_argument("--nrows", type=int, default=None, help="load only the first N rows")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.executescript((ROOT / "sql" / "00_schema.sql").read_text())

    total, next_id = 0, 1
    for chunk in pd.read_csv(args.data, chunksize=500_000, nrows=args.nrows):
        chunk = chunk.rename(columns=COLUMNS)[list(COLUMNS.values())]
        chunk.insert(0, "transaction_id", range(next_id, next_id + len(chunk)))
        next_id += len(chunk)
        chunk.to_sql("transactions", con, if_exists="append", index=False)
        total += len(chunk)
        print(f"  loaded {total:,} rows", end="\r")
    print(f"transactions: {total:,} rows loaded")
    for col in ("type", "step", "is_fraud", "name_dest"):
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_transactions_{col} ON transactions({col})")

    if Path(args.scores).exists():
        s = pd.read_csv(args.scores)
        if "outcome" not in s.columns:
            s["outcome"] = np.select(
                [
                    (s["flagged"] == 1) & (s["actual_fraud"] == 1),
                    (s["flagged"] == 1) & (s["actual_fraud"] == 0),
                    (s["flagged"] == 0) & (s["actual_fraud"] == 1),
                ],
                ["True positive", "False positive", "False negative"],
                default="True negative",
            )
        cols = ["row_id", "step", "hour_of_day", "type", "amount", "actual_fraud",
                "fraud_probability", "flagged", "existing_flag", "outcome"]
        s[[c for c in cols if c in s.columns]].to_sql("scored_transactions", con, if_exists="append", index=False)
        print(f"scored_transactions: {len(s):,} rows loaded")
    else:
        print("scored_transactions: skipped (run dashboard/export_scored_test.py first)")

    con.commit()
    con.close()
    print(f"Database written to {args.db}")
    print("Run a query, e.g.:  sqlite3 sql/paysim.db < sql/02_fraud_by_type.sql")


if __name__ == "__main__":
    main()
