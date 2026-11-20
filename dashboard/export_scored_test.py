import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TARGET  # noqa: E402
from src.data import load_data, time_split  # noqa: E402
from src.predict import score  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Export the scored test period for Power BI and SQL")
    ap.add_argument("--data", default=None)
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "powerbi_data"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data, args.nrows, args.synthetic)
    scored_all = score(df)
    _, _, test = time_split(scored_all["step"].values)
    t = scored_all[test]

    export = pd.DataFrame(
        {
            "row_id": t.index,
            "step": t["step"].astype(int).values,
            "hour_of_day": ((t["step"].astype(int) - 1) % 24).values,
            "type": t["type"].astype(str).values,
            "amount": t["amount"].values,
            "actual_fraud": t[TARGET].astype(int).values,
            "fraud_probability": t["fraud_probability"].values,
            "flagged": t["flagged"].astype(int).values,
            "existing_flag": t["isFlaggedFraud"].astype(int).values,
        }
    )
    export["outcome"] = np.select(
        [
            (export["flagged"] == 1) & (export["actual_fraud"] == 1),
            (export["flagged"] == 1) & (export["actual_fraud"] == 0),
            (export["flagged"] == 0) & (export["actual_fraud"] == 1),
        ],
        ["True positive", "False positive", "False negative"],
        default="True negative",
    )
    export.to_csv(out / "scored_test.csv", index=False)
    print(f"Exported {len(export):,} scored test-period transactions to {out / 'scored_test.csv'}")


if __name__ == "__main__":
    main()
