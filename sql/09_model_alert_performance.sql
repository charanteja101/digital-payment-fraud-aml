-- Model alerts vs the existing rule on the test period (needs the scored_transactions table).
SELECT
    SUM(CASE WHEN outcome = 'True positive'  THEN 1 ELSE 0 END)   AS true_positives,
    SUM(CASE WHEN outcome = 'False positive' THEN 1 ELSE 0 END)   AS false_positives,
    SUM(CASE WHEN outcome = 'False negative' THEN 1 ELSE 0 END)   AS false_negatives,
    ROUND(100.0 * SUM(CASE WHEN outcome = 'True positive' THEN 1 ELSE 0 END)
          / NULLIF(SUM(actual_fraud), 0), 2)                      AS model_recall_pct,
    ROUND(100.0 * SUM(CASE WHEN outcome = 'True positive' THEN 1 ELSE 0 END)
          / NULLIF(SUM(flagged), 0), 2)                           AS model_precision_pct,
    ROUND(100.0 * SUM(CASE WHEN existing_flag = 1 AND actual_fraud = 1 THEN 1 ELSE 0 END)
          / NULLIF(SUM(actual_fraud), 0), 2)                      AS existing_rule_recall_pct
FROM scored_transactions;
