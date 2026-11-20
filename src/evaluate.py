import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import FPR_BUDGETS


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def threshold_for_fpr(y_ref, p_ref, max_fpr):
    return float(np.quantile(np.asarray(p_ref)[np.asarray(y_ref) == 0], 1 - max_fpr))


def operating_point(y, flagged, amount):
    y = np.asarray(y)
    flagged = np.asarray(flagged).astype(bool)
    amount = np.asarray(amount, dtype="float64")
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())
    fraud_amt = float(amount[y == 1].sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / max(tp + fn, 1), "recall_95ci": wilson(tp, tp + fn),
        "fpr": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
        "alerts": tp + fp,
        "fraud_amount_caught": float(amount[flagged & (y == 1)].sum()),
        "fraud_amount_total": fraud_amt,
        "fraud_amount_caught_pct": float(amount[flagged & (y == 1)].sum() / fraud_amt) if fraud_amt else float("nan"),
        "legit_amount_blocked": float(amount[flagged & (y == 0)].sum()),
    }


def score_summary(y, p, val_y, val_p, amount):
    out = {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}
    points = {}
    for b in FPR_BUDGETS:
        thr = threshold_for_fpr(val_y, val_p, b)
        op = operating_point(y, np.asarray(p) >= thr, amount)
        op["threshold"] = thr
        points[str(b)] = op
        out[f"recall_at_fpr_{b}"] = op["recall"]
    out["operating_points"] = points
    return out


def plot_eda(df, out):
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    by_type = df.groupby("type", observed=True)["isFraud"].agg(["count", "sum"])
    ax[0].bar(by_type.index.astype(str), by_type["count"], color="#4C78A8", label="All")
    ax[0].bar(by_type.index.astype(str), by_type["sum"], color="#E45756", label="Fraud")
    ax[0].set_yscale("log")
    ax[0].set_title("Transactions by type (log scale)")
    ax[0].legend()
    ax[0].tick_params(axis="x", rotation=30)
    hour = (df["step"].astype("int32") - 1) % 24
    ax[1].bar(range(24), df.groupby(hour)["isFraud"].mean() * 100, color="#E45756")
    ax[1].set_title("Fraud rate (%) by hour of day")
    ax[1].set_xlabel("hour")
    day = (df["step"].astype("int32") - 1) // 24
    ax[2].bar(sorted(day.unique()), df.groupby(day)["isFraud"].sum(), color="#F58518")
    ax[2].set_title("Fraud cases per day")
    ax[2].set_xlabel("day")
    plt.tight_layout()
    plt.savefig(out / "eda_overview.png", dpi=150)
    plt.close()


def plot_curves(y, scores, out):
    plt.figure(figsize=(6.5, 5))
    for name, p in scores.items():
        pr, rc, _ = precision_recall_curve(y, p)
        plt.plot(rc, pr, label=f"{name} (AP={average_precision_score(y, p):.3f})")
    plt.axhline(np.mean(y), color="k", ls="--", alpha=0.4)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (time-based test set)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "pr_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6.5, 5))
    for name, p in scores.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    for b in FPR_BUDGETS:
        plt.axvline(b, color="grey", ls=":", alpha=0.5)
    plt.xscale("log")
    plt.xlim(1e-5, 1)
    plt.xlabel("False positive rate (log scale)")
    plt.ylabel("Recall (TPR)")
    plt.title("Recall vs false positive rate")
    plt.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(out / "roc_curve.png", dpi=150)
    plt.close()


def plot_cm(op, budget, out):
    cm = np.array([[op["tn"], op["fp"]], [op["fn"], op["tp"]]])
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues", norm=matplotlib.colors.LogNorm(vmin=max(cm.min(), 1), vmax=cm.max()))
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                     color="white" if cm[i, j] > np.sqrt(cm.max()) else "black")
    plt.xticks([0, 1], ["Pred: Legit", "Pred: Fraud"])
    plt.yticks([0, 1], ["Actual: Legit", "Actual: Fraud"])
    plt.title(f"Final model at {budget:.2%} FPR budget")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()


def plot_importance(names, imp, out):
    order = np.argsort(imp)[-15:]
    plt.figure(figsize=(7, 5.5))
    plt.barh(np.array(names)[order], np.array(imp)[order], color="#4C78A8")
    plt.title("Final model feature importance")
    plt.tight_layout()
    plt.savefig(out / "feature_importance.png", dpi=150)
    plt.close()


def plot_value(op, out):
    plt.figure(figsize=(5.5, 4.5))
    caught, missed = op["fraud_amount_caught"], op["fraud_amount_total"] - op["fraud_amount_caught"]
    plt.bar(["Fraud caught", "Fraud missed", "Legit blocked"], [caught, missed, op["legit_amount_blocked"]],
            color=["#54A24B", "#E45756", "#F58518"])
    plt.title("Money view at the main operating point")
    plt.ylabel("Amount")
    plt.tight_layout()
    plt.savefig(out / "value_caught.png", dpi=150)
    plt.close()
