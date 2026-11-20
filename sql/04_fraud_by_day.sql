-- Fraud volume per day of the 30-day window (useful for spotting drift before a time-based split).
SELECT
    (step - 1) / 24 + 1                             AS day,
    COUNT(*)                                        AS transactions,
    SUM(is_fraud)                                   AS fraud_cases,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4)      AS fraud_rate_pct
FROM transactions
GROUP BY day
ORDER BY day;
