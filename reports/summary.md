# Results

Data source: **synthetic (Home Credit schema)**, 30,000 applicants, default rate 10.6%, thin file share 39.8%.

## Model performance (held-out test set)

| Model | AUC | Gini | KS |
|---|---|---|---|
| Logistic regression | 0.720 | 0.440 | 0.328 |
| LightGBM | 0.729 | 0.459 | 0.350 |

## Risk tiers

| tier   | pd_ceiling   |   limit_x_monthly_income |   max_tenure_months |   applicants | avg_pd   | observed_default_rate   | avg_apr   | share   |
|:-------|:-------------|-------------------------:|--------------------:|-------------:|:---------|:------------------------|:----------|:--------|
| A      | 5.7%         |                      3   |                  24 |         1780 | 4.8%     | 3.1%                    | 16.9%     | 29.7%   |
| B      | 8.0%         |                      2   |                  18 |         1613 | 6.7%     | 6.6%                    | 18.6%     | 26.9%   |
| C      | 11.8%        |                      1.5 |                  12 |         1133 | 9.5%     | 12.5%                   | 21.2%     | 18.9%   |
| D      | 21.4%        |                      1   |                   6 |          887 | 15.9%    | 15.9%                   | 27.9%     | 14.8%   |
| E      | 100.0%       |                      0   |                   0 |          587 | 33.0%    | 32.5%                   | nan       | 9.8%    |

## Recommended cut-off

Approve applicants with PD at or below **17.1%**: approval rate **85.0%**, expected loss rate **4.96%**, realised loss rate 4.75%, bad rate 7.4%.
