-- Business view: fraud money caught, fraud money missed, legitimate money wrongly flagged.
SELECT
    outcome,
    COUNT(*)                    AS transactions,
    ROUND(SUM(amount), 0)       AS total_amount,
    ROUND(AVG(amount), 2)       AS avg_amount
FROM scored_transactions
WHERE outcome <> 'True negative'
GROUP BY outcome
ORDER BY CASE outcome WHEN 'True positive' THEN 1 WHEN 'False negative' THEN 2 ELSE 3 END;
