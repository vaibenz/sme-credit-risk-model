# SME Credit Scoring & Risk Based Pricing Model

A credit scoring and pricing engine for thin-file borrowers, built for **NTU PEAK (Monee Team 1)**.

Our PEAK proposal, TemanUsaha, partners a digital financial platform with Indonesian rural development banks (Bank Perekonomian Rakyat, or BPRs). The BPR underwrites and lends locally, and the platform supplies digital onboarding, bookkeeping data and repayment rails. That only works if the BPR can make a sound credit decision on a merchant with little or no bureau history. This project asks how far behavioural data (how someone has repaid before, how often they pay late, whether they underpay) can carry a credit decision, and how the resulting score turns into limits, tenures and prices a lender could actually offer.

## What it does

1. **Feature engineering in SQL.** Repayment behaviour (late payment ratio, days past due, underpayment ratio, paid to due ratio, last 12 months behaviour) and bureau footprint are aggregated with SQL queries in [`sql/`](sql), run on an in-memory SQLite database. Affordability ratios (annuity to income, credit to income, implied term) are added in pandas.
2. **Model comparison.** A regularised logistic regression baseline against a LightGBM challenger with early stopping, evaluated on a held-out test set with AUC, Gini and KS.
3. **Calibration.** LightGBM scores are Platt calibrated on the validation set so predicted PDs line up with observed default rates, which matters because prices and expected losses are computed from them.
4. **Five risk tiers.** Each tier gets a PD ceiling, a credit limit as a multiple of monthly income and a maximum tenure. Tier E is declined and referred back to bookkeeping and financial coaching, which mirrors the "not yet ready" path in the TemanUsaha app.
5. **Risk based pricing.** A break-even APR per applicant that covers cost of funds, servicing cost, target margin and expected loss, capped at a responsible lending ceiling.
6. **Cut-off simulation.** Approval rate against expected and realised loss across PD cut-offs, then a recommended cut-off at the highest approval rate inside a stated loss appetite.

All business assumptions (LGD, cost of funds, margin, APR cap, tier limits, loss appetite) sit in [`src/credit_risk/config.py`](src/credit_risk/config.py) so they can be challenged in one place. Protected attributes such as gender and family status are excluded from the model by design.

## Results

The charts and table below are from the **synthetic run** (`--synthetic`), which uses generated data with the same schema as Home Credit so the repo runs out of the box. Rerun on the real dataset to reproduce real-data results; the pipeline rewrites everything in [`reports/`](reports).

| Model | AUC | Gini | KS |
|---|---|---|---|
| Logistic regression | 0.720 | 0.440 | 0.328 |
| LightGBM (calibrated) | 0.729 | 0.459 | 0.350 |

| Tier | Share of applicants | Observed default rate | Avg APR | Limit (x monthly income) | Max tenure |
|---|---|---|---|---|---|
| A | 29.7% | 3.1% | 16.9% | 3.0 | 24 months |
| B | 26.9% | 6.6% | 18.6% | 2.0 | 18 months |
| C | 18.9% | 12.5% | 21.2% | 1.5 | 12 months |
| D | 14.8% | 15.9% | 27.9% | 1.0 | 6 months |
| E | 9.8% | 32.5% | declined | 0 | n/a |

Default rates rise monotonically across tiers, and calibrated PDs track observed rates closely in tiers B to E.

<p>
<img src="reports/roc.png" width="48%">
<img src="reports/tier_default_rates.png" width="48%">
</p>
<p>
<img src="reports/approval_vs_loss.png" width="48%">
<img src="reports/feature_importance.png" width="48%">
</p>

Repayment behaviour features (late payment ratio, underpayment ratio, average days late) rank just behind the external scores, which is the point of the exercise: for thin-file merchants, the bookkeeping and repayment data a platform like TemanUsaha collects is what lets a BPR lend responsibly.

## Run it

```bash
git clone https://github.com/vaibenz/sme-credit-risk-model.git
cd sme-credit-risk-model
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py --synthetic   # about a minute, no downloads needed
pytest -q                            # unit tests for SQL features, tiers and pricing
```

### On the real Home Credit data

1. Download the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) dataset from Kaggle (free account, accept the competition rules).
2. Put `application_train.csv`, `installments_payments.csv` and `bureau.csv` in `data/home-credit/`.
3. Run:

```bash
python run_pipeline.py --data-dir data/home-credit
# or a faster first pass:
python run_pipeline.py --data-dir data/home-credit --sample 100000
```

The full dataset has 307,511 applications and the instalments table is about 690 MB, so allow a few minutes and around 8 GB of RAM.

## Project structure

```
├── run_pipeline.py              # end to end entry point
├── sql/
│   ├── installments_features.sql
│   └── bureau_features.sql
├── src/credit_risk/
│   ├── config.py                # pricing, tier and risk appetite assumptions
│   ├── data.py                  # loading the Home Credit tables
│   ├── features.py              # SQL + pandas feature engineering
│   ├── models.py                # logistic regression, LightGBM, calibration, metrics
│   ├── tiers.py                 # PD to tier, limit and tenure
│   ├── policy.py                # risk based APR and cut-off simulation
│   └── report.py                # charts and summary.md
├── scripts/generate_synthetic_data.py
├── tests/test_policy.py
└── reports/                     # generated outputs
```

## Limitations

The Home Credit data covers consumer cash and revolving loans, not Indonesian MSME working capital loans, so it is a proxy for the behaviour a BPR would see rather than a like-for-like sample. LGD, funding cost and the APR cap are illustrative assumptions, not figures from any lender. A production model would also need out-of-time validation, reject inference and fairness testing across segments before any lending decision relied on it.

## Context

Built as part of the NTU PEAK Leadership Programme 2026 (Monee Team 1), responding to the problem statement: *How can we better utilise technology to increase access to sustainable financing for SMEs in Southeast Asia, to foster financial inclusion and long-term economic resilience?* The companion app is [TemanUsaha](https://github.com/vaibenz/temanusaha-bookkeeping-app).

## License

MIT
