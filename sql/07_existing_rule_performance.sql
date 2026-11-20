-- How well does the built-in is_flagged_fraud rule (transfer over 200,000) catch real fraud?
SELECT
    SUM(CASE WHEN is_flagged_fraud = 1 AND is_fraud = 1 THEN 1 ELSE 0 END)   AS flagged_and_fraud,
    SUM(CASE WHEN is_flagged_fraud = 1 AND is_fraud = 0 THEN 1 ELSE 0 END)   AS flagged_not_fraud,
    SUM(CASE WHEN is_flagged_fraud = 0 AND is_fraud = 1 THEN 1 ELSE 0 END)   AS fraud_missed,
    ROUND(100.0 * SUM(CASE WHEN is_flagged_fraud = 1 AND is_fraud = 1 THEN 1 ELSE 0 END)
          / NULLIF(SUM(is_fraud), 0), 3)                                     AS rule_recall_pct
FROM transactions;
