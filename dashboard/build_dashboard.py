"""Build an interactive fraud & AML dashboard (one self-contained HTML file) with Python and Plotly.

Run from the repository root after training:

    python dashboard/build_dashboard.py --data data/paysim.csv

Output: dashboard/fraud_aml_dashboard.html (open it in any browser).

Sections
    1. Fraud overview   - where and when fraud happens (whole dataset)
    2. Alert performance - how the final model did on the untouched, later test period
    3. Model vs rules   - comparison table, operating points, top alerts
"""
import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import METRICS_PATH, TARGET  # noqa: E402

AMOUNT_BINS = [0, 1_000, 10_000, 100_000, 200_000, 1_000_000, np.inf]
AMOUNT_LABELS = ["< 1k", "1k-10k", "10k-100k", "100k-200k", "200k-1M", "1M+"]

BLUE, RED, GREY, GREEN, AMBER = "#2563eb", "#dc2626", "#9ca3af", "#16a34a", "#d97706"


# --------------------------------------------------------------------------- data


def _rate_table(df, key):
    g = df.groupby(key, observed=True)[TARGET].agg(transactions="size", fraud="sum").reset_index()
    g["fraud_rate"] = g["fraud"] / g["transactions"]
    return g


def compute(df, test, metrics):
    """Aggregate everything the dashboard shows. `df` is the full data, `test` the scored test period.

    `test` must have the columns: step, type, amount, isFraud, fraud_probability, flagged.
    """
    d = df[["step", "type", "amount", TARGET]].copy()
    d["hour_of_day"] = (d["step"].astype(int) - 1) % 24
    d["day"] = (d["step"].astype(int) - 1) // 24 + 1
    d["amount_band"] = pd.cut(d["amount"], AMOUNT_BINS, labels=AMOUNT_LABELS, right=False)

    out = {
        "by_type": _rate_table(d, "type"),
        "by_hour": _rate_table(d, "hour_of_day"),
        "by_day": _rate_table(d, "day"),
        "by_band": _rate_table(d, "amount_band"),
        "n": len(d),
        "n_fraud": int(d[TARGET].sum()),
    }

    y = test[TARGET].astype(int).to_numpy()
    flag = test["flagged"].astype(int).to_numpy()
    amt = test["amount"].to_numpy()
    tp = int(((flag == 1) & (y == 1)).sum())
    fp = int(((flag == 1) & (y == 0)).sum())
    fn = int(((flag == 0) & (y == 1)).sum())
    tn = int(((flag == 0) & (y == 0)).sum())
    out["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    out["test"] = {
        "rows": len(test),
        "fraud": int(y.sum()),
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "alerts": tp + fp,
        "fraud_amount_total": float(amt[y == 1].sum()),
        "fraud_amount_caught": float(amt[(flag == 1) & (y == 1)].sum()),
        "legit_amount_flagged": float(amt[(flag == 1) & (y == 0)].sum()),
    }

    # score distribution by true class, on a log y axis so the tiny fraud class stays visible
    edges = np.linspace(0, 1, 21)
    p = test["fraud_probability"].to_numpy()
    out["hist_edges"] = edges
    out["hist_legit"] = np.histogram(p[y == 0], bins=edges)[0]
    out["hist_fraud"] = np.histogram(p[y == 1], bins=edges)[0]

    top = test.sort_values("fraud_probability", ascending=False).head(15)
    out["top_alerts"] = pd.DataFrame(
        {
            "Step": top["step"].astype(int).to_numpy(),
            "Type": top["type"].astype(str).to_numpy(),
            "Amount": top["amount"].to_numpy(),
            "Fraud probability": top["fraud_probability"].to_numpy(),
            "Actual": np.where(top[TARGET].to_numpy() == 1, "Fraud", "Legit"),
        }
    )

    # model vs rules (from the training run, same test period)
    final_name = metrics.get("final_model", "XGBoost (pre-transaction) FINAL")
    op = metrics["main_operating_point"]
    rows = [("Final model", op["recall"], op["precision"], op["alerts"])]
    for name, v in metrics["rule_baselines"].items():
        short = "Built-in flag" if name.startswith("Existing") else name.replace("Rule: ", "").split(" (")[0].capitalize()
        rows.append((short, v["recall"], v["precision"], v["alerts"]))
    out["vs_rules"] = pd.DataFrame(rows, columns=["Approach", "Recall", "Precision", "Alerts"])

    out["comparison"] = pd.DataFrame(
        [
            {
                "Model": name,
                "ROC-AUC": v["roc_auc"],
                "PR-AUC": v["pr_auc"],
                "Recall @0.01% FPR": v["recall_at_fpr_0.0001"],
                "Recall @0.05% FPR": v["recall_at_fpr_0.0005"],
                "Recall @0.10% FPR": v["recall_at_fpr_0.001"],
            }
            for name, v in metrics["comparison"].items()
        ]
    )
    ops = metrics["comparison"][final_name]["operating_points"]
    out["operating_points"] = pd.DataFrame(
        [
            {
                "FPR budget": f"{float(k):.2%}",
                "Recall": o["recall"],
                "Precision": o["precision"],
                "Alerts": o["alerts"],
                "Fraud amount caught": o["fraud_amount_caught_pct"],
            }
            for k, o in ops.items()
        ]
    )
    out["feature_importance"] = pd.Series(metrics["feature_importance"]).sort_values()
    return out


# ------------------------------------------------------------------------ figures


def _layout(fig, title, height=340):
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=15)),
        height=height,
        margin=dict(l=50, r=20, t=50, b=45),
        template="plotly_white",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def build_figures(c):
    import plotly.graph_objects as go

    figs = {}

    t = c["by_type"]
    f = go.Figure(go.Bar(x=t["type"].astype(str), y=t["fraud"], marker_color=RED,
                         customdata=t["fraud_rate"],
                         hovertemplate="%{x}<br>%{y:,} fraud cases<br>rate %{customdata:.3%}<extra></extra>"))
    figs["type"] = _layout(f, "Fraud cases by transaction type")
    figs["type"].update_yaxes(title="Fraud cases")

    h = c["by_hour"]
    f = go.Figure(go.Scatter(x=h["hour_of_day"], y=h["fraud_rate"], mode="lines+markers",
                             line=dict(color=BLUE, width=2.5),
                             hovertemplate="Hour %{x}:00<br>fraud rate %{y:.3%}<extra></extra>"))
    figs["hour"] = _layout(f, "Fraud rate by hour of day")
    f.update_yaxes(title="Fraud rate", tickformat=".2%")
    f.update_xaxes(title="Hour of day", dtick=2)

    dd = c["by_day"]
    f = go.Figure(go.Bar(x=dd["day"], y=dd["fraud"], marker_color=BLUE,
                         hovertemplate="Day %{x}<br>%{y:,} fraud cases<extra></extra>"))
    figs["day"] = _layout(f, "Fraud cases per day")
    f.update_xaxes(title="Day of the simulation")
    f.update_yaxes(title="Fraud cases")

    b = c["by_band"]
    f = go.Figure(go.Bar(x=b["amount_band"].astype(str), y=b["fraud_rate"], marker_color=AMBER,
                         customdata=b["fraud"],
                         hovertemplate="%{x}<br>fraud rate %{y:.3%}<br>%{customdata:,} cases<extra></extra>"))
    figs["band"] = _layout(f, "Fraud rate by amount band")
    f.update_yaxes(title="Fraud rate", tickformat=".2%")

    cm = c["confusion"]
    labels = ["True positive", "False positive", "False negative"]
    vals = [cm["tp"], cm["fp"], cm["fn"]]
    f = go.Figure(go.Bar(x=labels, y=vals, marker_color=[GREEN, AMBER, RED], text=[f"{v:,}" for v in vals],
                         textposition="outside", hovertemplate="%{x}: %{y:,}<extra></extra>"))
    figs["outcome"] = _layout(f, f"Alert outcomes ({cm['tn']:,} true negatives not shown)")
    f.update_yaxes(title="Transactions", range=[0, max(vals) * 1.2 if max(vals) else 1])

    mids = (c["hist_edges"][:-1] + c["hist_edges"][1:]) / 2
    f = go.Figure()
    f.add_bar(x=mids, y=c["hist_legit"], name="Legitimate", marker_color=GREY)
    f.add_bar(x=mids, y=c["hist_fraud"], name="Fraud", marker_color=RED)
    f.update_layout(barmode="group")
    figs["scores"] = _layout(f, "Model score by true class (log scale)")
    f.update_yaxes(type="log", title="Transactions")
    f.update_xaxes(title="Fraud probability", tickformat=".0%")

    ts = c["test"]
    caught = ts["fraud_amount_caught"]
    missed = ts["fraud_amount_total"] - caught
    f = go.Figure(go.Bar(x=["Fraud caught", "Fraud missed", "Legit flagged"],
                         y=[caught, missed, ts["legit_amount_flagged"]],
                         marker_color=[GREEN, RED, AMBER],
                         hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
    figs["money"] = _layout(f, "Money view (test period, amount)")
    f.update_yaxes(title="Amount", tickformat=".2s")

    v = c["vs_rules"]
    f = go.Figure()
    f.add_bar(x=v["Approach"], y=v["Recall"], name="Recall", marker_color=BLUE)
    f.add_bar(x=v["Approach"], y=v["Precision"], name="Precision", marker_color=AMBER)
    f.update_layout(barmode="group")
    figs["rules"] = _layout(f, "Model vs rules: recall and precision")
    f.update_yaxes(tickformat=".0%", range=[0, 1.05])

    f = go.Figure(go.Bar(x=v["Alerts"], y=v["Approach"], orientation="h", marker_color=BLUE,
                         text=[f"{a:,}" for a in v["Alerts"]], textposition="outside", cliponaxis=False,
                         hovertemplate="%{y}: %{x:,} alerts<extra></extra>"))
    figs["alerts"] = _layout(f, "Alerts raised on the test period (fewer is cheaper to review)")
    f.update_xaxes(type="log", title="Alerts (log scale)", range=[0.5, 6.2])

    fi = c["feature_importance"]
    f = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h", marker_color=BLUE,
                         hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    figs["features"] = _layout(f, "Feature importance (final model)", height=420)
    f.update_xaxes(tickformat=".0%")
    return figs


def _table_html(df, fmt):
    head = "".join(f"<th>{html.escape(str(col))}</th>" for col in df.columns)
    body = ""
    for _, row in df.iterrows():
        cells = "".join(
            f"<td>{html.escape(fmt.get(col, str)(row[col]))}</td>" for col in df.columns
        )
        body += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


# --------------------------------------------------------------------------- page

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--text:#111827;--muted:#6b7280;--line:#e5e7eb}
@media (prefers-color-scheme: dark){:root{--bg:#0f172a;--card:#1e293b;--text:#f1f5f9;--muted:#94a3b8;--line:#334155}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,'Segoe UI',Arial,sans-serif}
header{padding:28px 32px 8px}h1{margin:0;font-size:26px}h2{margin:34px 32px 12px;font-size:19px}
.sub{color:var(--muted);margin:6px 0 0;font-size:14px;max-width:900px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;padding:18px 32px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.kpi .v{font-size:26px;font-weight:700}.kpi .l{color:var(--muted);font-size:13px;margin-top:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;padding:0 32px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:8px 10px;overflow:auto}
.card h3{margin:10px 8px 8px;font-size:15px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}
footer{color:var(--muted);font-size:12px;padding:28px 32px 40px}
"""


def kpi(value, label):
    return f'<div class="kpi"><div class="v">{html.escape(value)}</div><div class="l">{html.escape(label)}</div></div>'


def render(c, figs, metrics, plotlyjs):
    def fig(key):
        return f'<div class="card">{figs[key].to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})}</div>'

    ts, cm = c["test"], c["confusion"]
    op = metrics["main_operating_point"]
    budget = float(metrics.get("main_fpr_budget", 0.0005))
    top_kpis = "".join(
        [
            kpi(f"{c['n']:,}", "Transactions in the data"),
            kpi(f"{c['n_fraud']:,}", "Fraud cases"),
            kpi(f"{c['n_fraud'] / c['n']:.3%}", "Fraud rate"),
            kpi(f"{ts['recall']:.1%}", "Fraud caught on test period (recall)"),
            kpi(f"{ts['precision']:.1%}", "Alert precision"),
            kpi(f"{ts['alerts']:,}", "Alerts raised on test period"),
            kpi(f"{ts['fraud_amount_caught'] / ts['fraud_amount_total']:.2%}" if ts["fraud_amount_total"] else "n/a",
                "Fraud amount caught"),
        ]
    )

    comp = _table_html(
        c["comparison"],
        {k: (lambda x: f"{x:.3f}") for k in c["comparison"].columns if k != "Model"},
    )
    ops = _table_html(
        c["operating_points"],
        {"Recall": lambda x: f"{x:.1%}", "Precision": lambda x: f"{x:.1%}",
         "Alerts": lambda x: f"{int(x):,}", "Fraud amount caught": lambda x: f"{x:.1%}"},
    )
    top = _table_html(
        c["top_alerts"],
        {"Amount": lambda x: f"{x:,.0f}", "Fraud probability": lambda x: f"{x:.1%}", "Step": lambda x: str(int(x))},
    )
    script = (
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>' if plotlyjs == "cdn"
        else f"<script>{plotlyjs}</script>"
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Digital Payment Fraud &amp; AML Dashboard</title><style>{CSS}</style>{script}</head><body>
<header><h1>Digital Payment Fraud &amp; AML Dashboard</h1>
<p class="sub">PaySim (synthetic) mobile-money transactions. Overview uses the whole dataset; alert performance uses the later,
untouched test period with thresholds chosen on validation data (false-alert budget {budget:.2%}). Portfolio project, not a production system.</p></header>
<div class="kpis">{top_kpis}</div>

<h2>1. Fraud overview</h2>
<div class="grid">{fig("type")}{fig("hour")}{fig("day")}{fig("band")}</div>

<h2>2. Alert performance (test period)</h2>
<div class="kpis" style="padding-top:0;padding-bottom:16px">{"".join([
    kpi(f"{cm['tp']:,}", "Frauds caught"), kpi(f"{cm['fn']:,}", "Frauds missed"),
    kpi(f"{cm['fp']:,}", "False alerts"), kpi(f"{ts['legit_amount_flagged']:,.0f}", "Legit amount flagged")])}</div>
<div class="grid">{fig("outcome")}{fig("scores")}{fig("money")}{fig("features")}</div>

<h2>3. Model vs rules</h2>
<div class="grid">{fig("rules")}{fig("alerts")}</div>
<h2>Details</h2>
<div class="grid">
<div class="card"><h3>Model comparison (test period)</h3>{comp}</div>
<div class="card"><h3>Final model operating points</h3>{ops}</div>
<div class="card"><h3>Highest-scored transactions in the test period</h3>{top}</div></div>
<footer>Generated by dashboard/build_dashboard.py from models/metrics.json and the scored data.
Final model recall at the headline budget: {op['recall']:.1%} (95% CI {op['recall_95ci'][0]:.1%} to {op['recall_95ci'][1]:.1%}).</footer>
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Build the interactive fraud & AML dashboard (HTML)")
    ap.add_argument("--data", default=None, help="path to paysim.csv")
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true", help="smoke test with fake data (numbers are meaningless)")
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "fraud_aml_dashboard.html"))
    ap.add_argument("--offline", action="store_true", help="embed plotly.js so the file works without internet (+4 MB)")
    args = ap.parse_args()

    try:
        import plotly  # noqa: F401
    except ImportError:
        sys.exit("plotly is not installed. Run: python -m pip install plotly")

    from src.data import load_data, time_split
    from src.predict import score

    if not METRICS_PATH.exists():
        sys.exit("models/metrics.json not found. Train first: python -m src.train --data data/paysim.csv")
    metrics = json.loads(METRICS_PATH.read_text())

    df = load_data(args.data, args.nrows, args.synthetic)
    print("[dashboard] scoring all transactions ...")
    scored = score(df)
    _, _, test_mask = time_split(scored["step"].to_numpy())
    test = scored[test_mask]

    c = compute(df, test, metrics)
    figs = build_figures(c)
    plotlyjs = "cdn"
    if args.offline:
        from plotly.offline import get_plotlyjs

        plotlyjs = get_plotlyjs()
    out = Path(args.out)
    out.write_text(render(c, figs, metrics, plotlyjs), encoding="utf-8")
    print(f"[dashboard] wrote {out}")


if __name__ == "__main__":
    main()
