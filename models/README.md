# models/

Everything here is written by the training script. The outputs from the full-data run (model, metrics and figures) are committed so the results can be viewed without retraining.

## Reproduce

From the repository root:

```bash
python -m src.train --data data/paysim.csv
```

Useful options: `--nrows 1000000` (quick test on the first million rows), `--n-iter 8` (random-search iterations), `--neg-frac 0.2` (share of legitimate training rows kept), `--max-fpr 0.0005` (false-alert budget for the headline result), `--synthetic` (smoke test with fake data).

The random seed is fixed (42). Results can still differ slightly across library versions.

## Outputs

| File | Content |
|---|---|
| `paysim_fraud_model.joblib` | Dict with the final XGBoost model, its feature list, the decision threshold and the downsampling rate (used by `src/predict.py`) |
| `metrics.json` | Model comparison, rule baselines, operating points with confidence intervals, split sizes, best parameters and feature importances (shown in the app) |
| `figures/` | EDA, precision-recall, recall-vs-FPR, confusion matrix, feature importance and money-view plots (shown in the app) |

## Score new transactions

Run from the repository root so `import src...` works:

```python
from src.predict import score
scored = score(transactions_df)   # adds fraud_probability and flagged
```

`score` expects PaySim-format columns (`step`, `type`, `amount`, `oldbalanceOrg`, `nameDest`, `oldbalanceDest`). Pass the whole period you want scored: destination-account history is computed from the earlier rows.
