import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "models" / "metrics.json"
FIG_DIR = ROOT / "models" / "figures"

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
REQUIRED = ["step", "type", "amount", "oldbalanceOrg", "nameDest", "oldbalanceDest"]
OPTIONAL = ["nameOrig", "newbalanceOrig", "newbalanceDest", "isFraud", "isFlaggedFraud"]

st.set_page_config(page_title="Digital Payment Fraud & AML", page_icon="🛡️", layout="wide")
st.title("Digital Payment Fraud & AML Monitor")
st.caption(
    "Scores mobile-money transactions with a model trained on the synthetic PaySim dataset, using only information "
    "known before a transaction completes. Portfolio project, not a production system."
)

try:
    from src.predict import score
except ImportError as exc:
    st.error(f"Could not import `src.predict`: {exc}")
    st.stop()

tab_csv, tab_single, tab_perf = st.tabs(["Score a CSV", "Single transaction", "Model performance"])

with tab_csv:
    st.write("Upload a CSV in PaySim format. Required columns:")
    st.code(", ".join(REQUIRED))
    st.caption(
        "Optional columns: " + ", ".join(OPTIONAL) + ". Include the whole period you want scored: the model uses "
        "destination-account history computed from earlier rows in the file. If `isFraud` is present, the app also shows how well the model did."
    )
    upload = st.file_uploader("CSV file", type=["csv"])
    if upload is not None:
        data = pd.read_csv(upload)
        missing = [c for c in REQUIRED if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            result = score(data)
            alerts = result[result["flagged"] == 1].sort_values("fraud_probability", ascending=False)
            c1, c2, c3 = st.columns(3)
            c1.metric("Transactions scored", f"{len(result):,}")
            c2.metric("Alerts raised", f"{len(alerts):,}")
            c3.metric("Alert rate", f"{len(alerts) / len(result):.4%}")
            if "isFraud" in result.columns:
                actual = int((result["isFraud"] == 1).sum())
                caught = int(((result["flagged"] == 1) & (result["isFraud"] == 1)).sum())
                false_pos = int(((result["flagged"] == 1) & (result["isFraud"] == 0)).sum())
                d1, d2, d3 = st.columns(3)
                d1.metric("Actual fraud cases", f"{actual:,}")
                d2.metric("Recall", f"{caught / actual:.1%}" if actual else "n/a")
                d3.metric("False alerts", f"{false_pos:,}")
                st.caption("These numbers are only meaningful on a period the model was not trained on.")
            st.dataframe(alerts.head(500))
            st.download_button(
                "Download alerts",
                data=alerts.to_csv(index=False).encode("utf-8"),
                file_name="fraud_alerts.csv",
                mime="text/csv",
            )

with tab_single:
    st.write("Enter what is known before the transaction completes.")
    a, b = st.columns(2)
    with a:
        tx_type = st.selectbox("Transaction type", TYPES, index=4)
        amount = st.number_input("Amount", min_value=0.0, max_value=100000000.0, value=181.0, step=100.0)
        step = st.number_input("Hour of the simulation (step)", min_value=1, max_value=744, value=100, step=1)
    with b:
        old_orig = st.number_input("Sender balance before", min_value=0.0, max_value=100000000.0, value=181.0, step=100.0)
        old_dest = st.number_input("Receiver balance before", min_value=0.0, max_value=100000000.0, value=0.0, step=100.0)
        dest_name = st.text_input("Receiver account (starts with M for a merchant)", value="C553264065")
    with st.expander("Receiver history (optional)"):
        prior_count = st.number_input("Earlier transactions received by this account", min_value=0, max_value=100000, value=0, step=1)
        prior_inflow = st.number_input("Total amount received earlier", min_value=0.0, max_value=1000000000.0, value=0.0, step=1000.0)
    row = pd.DataFrame(
        [
            {
                "step": int(step),
                "type": tx_type,
                "amount": amount,
                "oldbalanceOrg": old_orig,
                "nameDest": dest_name,
                "oldbalanceDest": old_dest,
                "dest_prior_txn_count": prior_count,
                "dest_prior_inflow": prior_inflow,
            }
        ]
    )
    scored = score(row)
    prob = float(scored["fraud_probability"].iloc[0])
    st.divider()
    st.metric("Fraud probability", f"{prob:.2%}")
    if int(scored["flagged"].iloc[0]) == 1:
        st.warning("Alert: this transaction would be sent for review.")
    else:
        st.success("No alert.")
    st.caption("The probability is corrected for the negative downsampling used in training, but treat it as a risk score rather than an exact probability.")

with tab_perf:
    if not METRICS_PATH.exists():
        st.info("Train the model first: `python -m src.train --data data/paysim.csv`")
    else:
        m = json.loads(METRICS_PATH.read_text())
        final_name = m.get("final_model", "XGBoost (pre-transaction) FINAL")
        final = m["comparison"][final_name]
        budget = str(m.get("main_fpr_budget", 0.0005))
        op = m.get("main_operating_point") or final["operating_points"][budget]
        sp = m["splits"]

        st.subheader("Headline result")
        st.caption(
            f"Final model (pre-transaction features) on the time-based test period, "
            f"false-alert budget {float(budget):.2%}. Thresholds were fixed on the validation period."
        )
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Fraud caught (recall)", f"{op['recall']:.1%}",
                  help=f"95% CI {op['recall_95ci'][0]:.1%} to {op['recall_95ci'][1]:.1%}")
        k2.metric("Precision", f"{op['precision']:.1%}")
        k3.metric("Alerts raised", f"{op['alerts']:,}")
        k4.metric("Fraud amount caught", f"{op['fraud_amount_caught_pct']:.2%}")
        k5, k6, k7, k8 = st.columns(4)
        k5.metric("PR-AUC", f"{final['pr_auc']:.4f}")
        k6.metric("ROC-AUC", f"{final['roc_auc']:.4f}")
        k7.metric("Frauds missed", f"{op['fn']:,}")
        k8.metric("False alerts", f"{op['fp']:,}")

        st.divider()
        st.subheader("Model comparison")
        comp = pd.DataFrame(
            [
                {
                    "Model": name,
                    "ROC-AUC": v["roc_auc"],
                    "PR-AUC": v["pr_auc"],
                    "Recall @ 0.01% FPR": v["recall_at_fpr_0.0001"],
                    "Recall @ 0.05% FPR": v["recall_at_fpr_0.0005"],
                    "Recall @ 0.10% FPR": v["recall_at_fpr_0.001"],
                }
                for name, v in m["comparison"].items()
            ]
        ).set_index("Model")
        st.dataframe(
            comp.style.format("{:.3f}").highlight_max(axis=0, color="#d7efe0"),
            width="stretch",
        )
        st.caption(
            "The all-features model uses the balance-after columns, which can leak the label in PaySim. "
            "The final model drops them and loses almost nothing."
        )

        left, right = st.columns(2)
        with left:
            st.subheader("Model vs simple rules")
            rows = [
                {"Approach": "Final model", "Recall": op["recall"], "Precision": op["precision"], "Alerts": op["alerts"]}
            ]
            for name, v in m["rule_baselines"].items():
                rows.append({"Approach": name, "Recall": v["recall"], "Precision": v["precision"], "Alerts": v["alerts"]})
            rules = pd.DataFrame(rows).set_index("Approach")
            st.dataframe(
                rules.style.format({"Recall": "{:.1%}", "Precision": "{:.1%}", "Alerts": "{:,}"}),
                width="stretch",
            )
            st.caption("Same test period. The model finds nearly all fraud with a small fraction of the alerts.")
        with right:
            st.subheader("Operating points")
            ops = pd.DataFrame(
                [
                    {
                        "FPR budget": f"{float(k):.2%}",
                        "Recall": o["recall"],
                        "95% CI": f"{o['recall_95ci'][0]:.1%} - {o['recall_95ci'][1]:.1%}",
                        "Precision": o["precision"],
                        "Alerts": o["alerts"],
                        "Amount caught": o["fraud_amount_caught_pct"],
                    }
                    for k, o in final["operating_points"].items()
                ]
            ).set_index("FPR budget")
            st.dataframe(
                ops.style.format(
                    {"Recall": "{:.1%}", "Precision": "{:.1%}", "Alerts": "{:,}", "Amount caught": "{:.1%}"}
                ),
                width="stretch",
            )
            st.caption("Choose the budget by how many alerts the review team can handle.")

        st.divider()
        st.subheader("Data and splits (by time)")
        s1, s2, s3 = st.columns(3)
        for col, key, label in ((s1, "train", "Train"), (s2, "validation", "Validation"), (s3, "test", "Test")):
            col.metric(label, f"{sp[key]['rows']:,} rows", f"{sp[key]['fraud']:,} fraud", delta_color="off")

        st.subheader("Feature importance")
        fi = pd.Series(m["feature_importance"]).sort_values(ascending=True)
        st.bar_chart(fi, horizontal=True)

        with st.expander("Best hyperparameters and raw metrics"):
            st.json(m["best_params"])
            st.json(m, expanded=False)

    figures = {p.stem: p for p in sorted(FIG_DIR.glob("*.png"))} if FIG_DIR.exists() else {}
    if figures:
        st.divider()
        st.subheader("Figures")
        items = list(figures.items())
        for i in range(0, len(items), 2):
            cols = st.columns(2)
            for col, (stem, path) in zip(cols, items[i : i + 2]):
                col.image(str(path), caption=stem.replace("_", " ").title(), width="stretch")
