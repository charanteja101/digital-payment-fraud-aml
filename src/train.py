import argparse
import json
import time
import warnings

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from src.config import (
    FIG_DIR, FPR_BUDGETS, LARGE_TRANSFER, METRICS_PATH, MODEL_PATH, MODELS_DIR, TARGET,
)
from src.data import load_data, time_split
from src.evaluate import (
    operating_point, plot_cm, plot_curves, plot_eda, plot_importance, plot_value,
    score_summary, threshold_for_fpr,
)
from src.features import build_features, feature_columns
from src.models import downsample_negatives, make_logreg, make_xgb, prior_correct, sample_params

warnings.filterwarnings("ignore")


def rule_row(name, flagged, y, amount):
    op = operating_point(y, flagged, amount)
    print(f"{name:42s} recall {op['recall']:7.1%} | precision {op['precision']:7.1%} | FPR {op['fpr']:.4%} | alerts {op['alerts']:,}")
    return op


def main():
    ap = argparse.ArgumentParser(description="Train the PaySim fraud / AML models")
    ap.add_argument("--data", default=None, help="path to the PaySim CSV (default: data/paysim.csv)")
    ap.add_argument("--nrows", type=int, default=None, help="read only the first N rows (quick tests)")
    ap.add_argument("--n-iter", type=int, default=8, help="random-search iterations")
    ap.add_argument("--neg-frac", type=float, default=0.2, help="share of legitimate training rows kept")
    ap.add_argument("--max-fpr", type=float, default=0.0005, help="false-positive-rate budget for the headline result")
    ap.add_argument("--synthetic", action="store_true", help="synthetic data (smoke test only)")
    args = ap.parse_args()

    t0 = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data, args.nrows, args.synthetic)
    plot_eda(df, FIG_DIR)
    X = build_features(df)
    meta = df.loc[X.index, ["step", "type", "amount", "oldbalanceOrg", TARGET, "isFlaggedFraud"]]
    y = meta[TARGET].values
    amount = meta["amount"].values

    tr, va, te = time_split(meta["step"].values)
    for name, m in [("train", tr), ("validation", va), ("test", te)]:
        print(f"[split] {name:10s}: {int(m.sum()):>9,} rows | {int(y[m].sum()):>6,} fraud | steps {int(meta['step'].values[m].min())}-{int(meta['step'].values[m].max())}")

    cols = {s: feature_columns(X.columns, s) for s in ("full", "pretxn", "no_balance")}
    X_tr, y_tr = downsample_negatives(X[tr], y[tr], args.neg_frac)
    print(f"[train] after negative downsampling ({args.neg_frac:.0%} kept): {len(X_tr):,} rows, {int(y_tr.sum()):,} fraud")
    X_va, y_va = X[va], y[va]
    X_te, y_te, amt_te = X[te], y[te], amount[te]

    print(f"\n[tuning] {args.n_iter} random configurations, scored on validation PR-AUC (natural class balance) ...")
    best, best_ap = None, -1.0
    for i, params in enumerate(sample_params(args.n_iter), 1):
        m = make_xgb(**params).fit(X_tr[cols["pretxn"]], y_tr)
        ap_val = average_precision_score(y_va, prior_correct(m.predict_proba(X_va[cols["pretxn"]])[:, 1], args.neg_frac))
        print(f"  {i}/{args.n_iter}: val PR-AUC {ap_val:.4f}")
        if ap_val > best_ap:
            best, best_ap = params, ap_val
    print(f"  best val PR-AUC {best_ap:.4f} with {best}")

    print("\n[models] Fitting final and comparison models ...")
    models = {
        "Logistic Regression (pre-transaction)": ("pretxn", make_logreg()),
        "XGBoost (all features, leakage-prone)": ("full", make_xgb(**best)),
        "XGBoost (pre-transaction) FINAL": ("pretxn", make_xgb(**best)),
        "XGBoost (no balance features)": ("no_balance", make_xgb(**best)),
    }
    final_name = "XGBoost (pre-transaction) FINAL"
    val_scores, test_scores = {}, {}
    for name, (fs, model) in models.items():
        model.fit(X_tr[cols[fs]], y_tr)
        val_scores[name] = prior_correct(model.predict_proba(X_va[cols[fs]])[:, 1], args.neg_frac)
        test_scores[name] = prior_correct(model.predict_proba(X_te[cols[fs]])[:, 1], args.neg_frac)
    final_model = models[final_name][1]

    comparison = {}
    print("\n" + "=" * 96)
    print("MODEL COMPARISON on the time-based test set (thresholds fixed on the validation period)")
    print("=" * 96)
    print(f"{'Model':40s} {'ROC-AUC':>8s} {'PR-AUC':>8s} " + " ".join(f"{'R@' + format(b, '.2%'):>9s}" for b in FPR_BUDGETS))
    for name in models:
        s = score_summary(y_te, test_scores[name], y_va, val_scores[name], amt_te)
        comparison[name] = s
        print(f"{name:40s} {s['roc_auc']:8.4f} {s['pr_auc']:8.4f} " + " ".join(f"{s[f'recall_at_fpr_{b}']:9.3f}" for b in FPR_BUDGETS))

    print("\n" + "=" * 96)
    print("RULE-BASED AML BASELINES on the test period (no learning)")
    print("=" * 96)
    t = meta["type"].astype(str).values[te]
    old_org = meta["oldbalanceOrg"].values[te]
    rules = {
        "Existing isFlaggedFraud (>200,000 transfers)": meta["isFlaggedFraud"].values[te] == 1,
        "Rule: transfer over 200,000": (t == "TRANSFER") & (amt_te > LARGE_TRANSFER),
        "Rule: account drain (amount >= old balance)": np.isin(t, ["TRANSFER", "CASH_OUT"]) & (old_org > 0) & (amt_te >= old_org),
    }
    rule_results = {name: rule_row(name, flag, y_te, amt_te) for name, flag in rules.items()}

    main_budget = args.max_fpr
    thr = threshold_for_fpr(y_va, val_scores[final_name], main_budget)
    main_op = operating_point(y_te, test_scores[final_name] >= thr, amt_te)
    print("\n" + "=" * 96)
    print(f"FINAL MODEL operating points (validation-chosen thresholds, applied once to the test period)")
    print("=" * 96)
    for b in sorted(set(FPR_BUDGETS + [main_budget])):
        op = comparison[final_name]["operating_points"].get(str(b)) or main_op
        print(f"FPR budget {b:.3%}: recall {op['recall']:.1%} (95% CI {op['recall_95ci'][0]:.1%}-{op['recall_95ci'][1]:.1%}) | "
              f"actual FPR {op['fpr']:.4%} | precision {op['precision']:.1%} | alerts {op['alerts']:,} | "
              f"fraud amount caught {op['fraud_amount_caught_pct']:.1%}")

    imp = np.asarray(final_model.feature_importances_)
    names_final = cols["pretxn"]
    top = [names_final[i] for i in np.argsort(imp)[::-1][:5]]
    print(f"\nTop features (final model): {', '.join(top)}")

    plot_curves(y_te, test_scores, FIG_DIR)
    plot_cm(main_op, main_budget, FIG_DIR)
    plot_importance(names_final, imp, FIG_DIR)
    plot_value(main_op, FIG_DIR)

    joblib.dump({"model": final_model, "features": names_final, "threshold": thr,
                 "neg_frac": args.neg_frac, "fpr_budget": main_budget}, MODEL_PATH)
    metrics = {
        "n_transactions": int(len(df)), "n_fraud": int(y.sum()), "fraud_rate": float(y.mean()),
        "splits": {n: {"rows": int(m.sum()), "fraud": int(y[m].sum())} for n, m in [("train", tr), ("validation", va), ("test", te)]},
        "neg_frac": args.neg_frac, "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()},
        "val_pr_auc": best_ap, "comparison": comparison, "rule_baselines": rule_results,
        "main_fpr_budget": main_budget, "main_threshold": thr, "main_operating_point": main_op,
        "top_features": top, "feature_importance": {c: float(v) for c, v in zip(names_final, imp)},
        "final_model": final_name,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
