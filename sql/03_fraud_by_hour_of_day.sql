-- Fraud rate by hour of day (step 1 is hour 0 of day 1).
SELECT
    (step - 1) % 24                                 AS hour_of_day,
    COUNT(*)                                        AS transactions,
    SUM(is_fraud)                                   AS fraud_cases,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4)      AS fraud_rate_pct
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;
