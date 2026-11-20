-- Destination accounts that received the most fraudulent transactions (candidates for mule-account review).
SELECT
    name_dest,
    COUNT(*)                         AS fraud_transactions,
    ROUND(SUM(amount), 0)            AS fraud_amount_received,
    MIN(step)                        AS first_step,
    MAX(step)                        AS last_step
FROM transactions
WHERE is_fraud = 1
GROUP BY name_dest
ORDER BY fraud_transactions DESC, fraud_amount_received DESC
LIMIT 20;
