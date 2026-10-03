"""Generate synthetic tables with the same schema as the Home Credit Default Risk dataset.

This lets the pipeline run end to end without a Kaggle download. The numbers it produces are
NOT the results on real data; see the README for how to reproduce the real-data results.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n: int, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.arange(100000, 100000 + n)
    z = rng.normal(size=n)                                   # latent credit risk, higher is riskier

    income = np.round(rng.lognormal(mean=11.9, sigma=0.5, size=n), -2)
    credit = np.round(income * rng.uniform(0.8, 5.0, size=n), -2)
    term_years = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.15, 0.25, 0.3, 0.2, 0.1])
    annuity = np.round(credit / term_years * rng.uniform(1.05, 1.25, size=n), 1)
    goods = np.round(credit * rng.uniform(0.8, 1.0, size=n), -2)

    def ext(noise, missing):
        v = sigmoid(-0.9 * z + rng.normal(scale=noise, size=n))
        v[rng.random(n) < missing] = np.nan
        return v

    days_employed = -rng.gamma(2.0, 900, size=n).round()
    days_employed[rng.random(n) < 0.15] = 365243
    app = pd.DataFrame({
        "SK_ID_CURR": ids,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods,
        "DAYS_BIRTH": -(rng.uniform(21, 65, size=n) * 365.25).round(),
        "DAYS_EMPLOYED": days_employed,
        "REGION_RATING_CLIENT": rng.choice([1, 2, 3], size=n, p=[0.1, 0.75, 0.15]),
        "EXT_SOURCE_1": ext(1.2, 0.55),
        "EXT_SOURCE_2": ext(0.9, 0.01),
        "EXT_SOURCE_3": ext(1.0, 0.20),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], size=n, p=[0.9, 0.1]),
        "NAME_INCOME_TYPE": rng.choice(["Working", "Commercial associate", "Pensioner", "State servant"],
                                       size=n, p=[0.52, 0.23, 0.18, 0.07]),
        "NAME_EDUCATION_TYPE": rng.choice(["Secondary / secondary special", "Higher education",
                                           "Incomplete higher", "Lower secondary"],
                                          size=n, p=[0.71, 0.24, 0.035, 0.015]),
    })

    # Repayment history: about 70% have past instalment loans, late behaviour driven by z.
    inst_rows = []
    late_p = sigmoid(-2.2 + 1.1 * z)
    has_hist = rng.random(n) < 0.7
    prev_id = 1000000
    for i in np.where(has_hist)[0]:
        for _ in range(rng.integers(1, 4)):
            prev_id += 1
            k = int(rng.integers(6, 25))
            due = -np.sort(rng.choice(np.arange(30, 1500), size=k, replace=False))[::-1]
            amt = float(np.round(rng.uniform(2000, 20000), 1))
            late = rng.random(k) < late_p[i]
            delay = np.where(late, rng.gamma(1.5, 12, size=k).round(), -rng.integers(0, 10, size=k))
            short = rng.random(k) < late_p[i] * 0.5
            paid = np.where(short, amt * rng.uniform(0.3, 0.95, size=k), amt)
            for j in range(k):
                inst_rows.append((prev_id, ids[i], j + 1, due[j], due[j] + delay[j], amt, round(float(paid[j]), 1)))
    inst = pd.DataFrame(inst_rows, columns=["SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_NUMBER",
                                            "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"])

    # Bureau: about 40% are thin file with no bureau record at all.
    b_rows = []
    for i in np.where(rng.random(n) >= 0.4)[0]:
        for _ in range(rng.integers(1, 6)):
            total = float(np.round(rng.uniform(20000, 600000), -2))
            debt_share = np.clip(rng.beta(2, 3) + 0.15 * z[i], 0, 1)
            overdue = int(rng.gamma(1.0, 25)) if rng.random() < sigmoid(-3 + 1.2 * z[i]) else 0
            b_rows.append((ids[i], rng.choice(["Active", "Closed"], p=[0.4, 0.6]), total,
                           round(total * debt_share, 1), overdue))
    bureau = pd.DataFrame(b_rows, columns=["SK_ID_CURR", "CREDIT_ACTIVE", "AMT_CREDIT_SUM",
                                           "AMT_CREDIT_SUM_DEBT", "CREDIT_DAY_OVERDUE"])

    # Default outcome: latent risk plus non-linear affordability, tenure and age effects,
    # calibrated to roughly 8% defaults (close to the real dataset's 8.1%).
    afford = annuity / income
    age = -app["DAYS_BIRTH"].values / 365.25
    years_emp = np.where(days_employed == 365243, np.nan, -days_employed / 365.25)
    new_job = np.nan_to_num(years_emp, nan=10) < 1
    logit = (-3.75 + 1.0 * z
             + 1.3 * (afford > 0.45)                      # payment burden threshold
             + 1.0 * ((age < 28) & new_job)               # young and newly employed
             + 0.5 * (app["REGION_RATING_CLIENT"].values == 3)
             - 0.4 * (age > 50)
             + rng.normal(scale=0.5, size=n))
    app.insert(1, "TARGET", (rng.random(n) < sigmoid(logit)).astype(int))
    return {"application_train": app, "installments_payments": inst, "bureau": bureau}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=30000)
    ap.add_argument("--out", default="data/synthetic")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in generate(args.rows).items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"wrote {out / name}.csv  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
