-- Fraud by transaction type (fraud is expected only in TRANSFER and CASH_OUT).
SELECT
    type,
    COUNT(*)                                        AS transactions,
    SUM(is_fraud)                                   AS fraud_cases,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4)      AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 0) AS fraud_amount
FROM transactions
GROUP BY type
ORDER BY fraud_cases DESC, transactions DESC;
