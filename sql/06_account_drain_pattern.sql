-- AML typology: "account drain" = a TRANSFER or CASH_OUT that takes the sender's whole balance.
SELECT
    type,
    CASE WHEN old_balance_orig > 0 AND amount >= old_balance_orig THEN 'Drains account' ELSE 'Partial amount' END AS pattern,
    COUNT(*)                                        AS transactions,
    SUM(is_fraud)                                   AS fraud_cases,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4)      AS fraud_rate_pct
FROM transactions
WHERE type IN ('TRANSFER', 'CASH_OUT')
GROUP BY type, pattern
ORDER BY type, pattern;
