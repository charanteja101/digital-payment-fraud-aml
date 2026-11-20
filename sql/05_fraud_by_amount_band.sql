-- Fraud rate and fraud money by amount band, for the two risky transaction types.
SELECT
    CASE
        WHEN amount < 1000    THEN '1. Under 1,000'
        WHEN amount < 10000   THEN '2. 1,000-10,000'
        WHEN amount < 100000  THEN '3. 10,000-100,000'
        WHEN amount < 200000  THEN '4. 100,000-200,000'
        ELSE                       '5. 200,000+'
    END                                                              AS amount_band,
    COUNT(*)                                                         AS transactions,
    SUM(is_fraud)                                                    AS fraud_cases,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4)                       AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 0)     AS fraud_amount
FROM transactions
WHERE type IN ('TRANSFER', 'CASH_OUT')
GROUP BY amount_band
ORDER BY amount_band;
