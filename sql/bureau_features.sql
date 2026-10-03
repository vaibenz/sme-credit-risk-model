-- Credit bureau footprint, one row per applicant. Thin-file applicants simply have no row.
SELECT
    SK_ID_CURR,
    COUNT(*)                                                    AS bureau_count,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END)   AS bureau_active_count,
    SUM(AMT_CREDIT_SUM_DEBT) / NULLIF(SUM(AMT_CREDIT_SUM), 0)   AS bureau_debt_ratio,
    MAX(CREDIT_DAY_OVERDUE)                                     AS bureau_max_overdue,
    AVG(CASE WHEN CREDIT_DAY_OVERDUE > 0 THEN 1.0 ELSE 0.0 END) AS bureau_overdue_ratio
FROM bureau
GROUP BY SK_ID_CURR;
