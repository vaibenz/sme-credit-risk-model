# Results

Data source: **Home Credit Default Risk**, 307,511 applicants, default rate 8.1%, thin file share 14.3%.

## Model performance (held-out test set)

| Model | AUC | Gini | KS |
|---|---|---|---|
| Logistic regression | 0.751 | 0.502 | 0.379 |
| LightGBM | 0.771 | 0.542 | 0.409 |

## Risk tiers

| tier   | pd_ceiling   |   limit_x_monthly_income |   max_tenure_months |   applicants | avg_pd   | observed_default_rate   | avg_apr   | share   |
|:-------|:-------------|-------------------------:|--------------------:|-------------:|:---------|:------------------------|:----------|:--------|
| A      | 3.0%         |                      3   |                  24 |        18502 | 1.9%     | 1.9%                    | 14.5%     | 30.1%   |
| B      | 5.8%         |                      2   |                  18 |        15496 | 4.3%     | 4.2%                    | 16.5%     | 25.2%   |
| C      | 10.2%        |                      1.5 |                  12 |        12331 | 7.7%     | 7.7%                    | 19.5%     | 20.0%   |
| D      | 18.6%        |                      1   |                   6 |         9225 | 13.6%    | 13.7%                   | 25.4%     | 15.0%   |
| E      | 100.0%       |                      0   |                   0 |         5949 | 28.8%    | 29.6%                   | nan       | 9.7%    |

## Recommended cut-off

Approve applicants with PD at or below **11.9%**: approval rate **80.0%**, expected loss rate **2.94%**, realised loss rate 2.91%, bad rate 4.6%.
