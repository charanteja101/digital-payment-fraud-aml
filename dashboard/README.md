# Dashboards

Two options live in this folder.

## A. Python dashboard (Plotly, no extra software)

An interactive, single-file HTML dashboard with KPI cards, fraud overview charts, alert performance, a money view, a model-vs-rules comparison and tables. It follows your light or dark browser theme.

```bash
python -m pip install plotly
python dashboard/build_dashboard.py --data data/paysim.csv
```

This writes `dashboard/fraud_aml_dashboard.html`; open it in a browser (double-click). Scoring all 6.4M rows takes a few minutes. Options: `--nrows 1000000` for a quick run, `--offline` to embed plotly.js (the file then works without internet but is about 4 MB larger).

Take screenshots of it and save them in `dashboard/screenshots/` for the README.

## B. Power BI

A `.pbix` file cannot be generated from code, so this folder gives you the export script, the DAX and a layout guide. Build the report in about 30 minutes, then save screenshots here.

## 1. Export the scored test period

After training the model (`python -m src.train --data data/paysim.csv`):

```bash
python dashboard/export_scored_test.py --data data/paysim.csv
```

This writes `dashboard/powerbi_data/scored_test.csv` (git-ignored) with one row per test-period transaction:

| Column | Meaning |
|---|---|
| `row_id` | Row number in the original file |
| `step`, `hour_of_day` | Simulation hour (1-744) and hour of day (0-23) |
| `type` | Transaction type |
| `amount` | Transaction amount |
| `actual_fraud` | 1 if the transaction was fraud |
| `fraud_probability` | The model's risk score |
| `flagged` | 1 if the score is at or above the model's decision threshold |
| `existing_flag` | The simulator's own `isFlaggedFraud` rule |
| `outcome` | True positive / False positive / False negative / True negative |

## 2. Build the report in Power BI Desktop

1. Get Data -> Text/CSV -> load `scored_test.csv` and name the table `scored_test`.
2. Add the calculated columns and measures from `measures.dax`.
3. Suggested layout:
   - **Page 1, Fraud overview:** cards for Total Transactions, Actual Fraud Cases and Fraud Rate; a column chart of Actual Fraud Cases by `type`; a column chart of Actual Fraud Cases by `hour_of_day`; a column chart of Fraud Rate by `amount_band`.
   - **Page 2, Alert performance:** cards for Recall, Precision, False Positive Rate and Alerts Raised; a bar chart of `outcome`; cards for Fraud Amount Caught, Fraud Amount Missed and Legit Amount Flagged; a card comparing Recall with Existing Rule Recall.
   - Slicers: `type`, `amount_band`, `score_band`.
4. Save as `dashboard/fraud_aml_dashboard.pbix`.

The DAX has not been tested in Power BI Desktop, so check the column and table names match your CSV.

## 3. Screenshots

Export each page as an image and save them in `dashboard/screenshots/` so reviewers can see the dashboard without Power BI. Reference them in the main README.
