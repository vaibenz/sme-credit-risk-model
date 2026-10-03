-- Repayment behaviour from past instalments, one row per applicant.
-- Positive days_late means the payment landed after the due date.
WITH paid AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        AMT_INSTALMENT,
        AMT_PAYMENT,
        DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT AS days_late,
        DAYS_INSTALMENT
    FROM installments
)
SELECT
    SK_ID_CURR,
    COUNT(*)                                                          AS inst_count,
    COUNT(DISTINCT SK_ID_PREV)                                        AS inst_prev_loans,
    AVG(CASE WHEN days_late > 0 THEN 1.0 ELSE 0.0 END)                AS inst_late_ratio,
    AVG(CASE WHEN days_late > 30 THEN 1.0 ELSE 0.0 END)               AS inst_late30_ratio,
    AVG(MAX(days_late, 0))                                            AS inst_avg_days_late,
    MAX(days_late)                                                    AS inst_max_days_late,
    AVG(CASE WHEN AMT_PAYMENT < AMT_INSTALMENT THEN 1.0 ELSE 0.0 END) AS inst_underpay_ratio,
    SUM(AMT_PAYMENT) / NULLIF(SUM(AMT_INSTALMENT), 0)                 AS inst_paid_to_due,
    AVG(CASE WHEN DAYS_INSTALMENT >= -365 AND days_late > 0 THEN 1.0
             WHEN DAYS_INSTALMENT >= -365 THEN 0.0 END)               AS inst_late_ratio_12m
FROM paid
GROUP BY SK_ID_CURR;
