"""End to end: load data, engineer features, train and compare models, tier, price and pick a cut-off.

Usage:
    python run_pipeline.py --synthetic                 # runs out of the box on generated data
    python run_pipeline.py --data-dir data/home-credit # real Home Credit data from Kaggle
"""
import argparse
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent / "src"))

from credit_risk import report  # noqa: E402
from credit_risk.config import ID, RANDOM_STATE, TARGET, Config  # noqa: E402
from credit_risk.data import load_tables  # noqa: E402
from credit_risk.features import build_feature_table  # noqa: E402
from credit_risk.models import (PlattCalibrator, credit_metrics, lightgbm_model, logistic_baseline,  # noqa: E402
                                split_columns, to_lgb_frame)
from credit_risk.policy import cutoff_simulation, recommend_cutoff, risk_based_apr  # noqa: E402
from credit_risk.tiers import assign_tier, calibrate_tiers, tier_summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/home-credit")
    ap.add_argument("--synthetic", action="store_true", help="generate and use synthetic data")
    ap.add_argument("--sample", type=int, default=None, help="subsample applicants for a quick run")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    cfg = Config()

    if args.synthetic:
        from scripts.generate_synthetic_data import generate
        data_dir = Path("data/synthetic")
        if not (data_dir / "application_train.csv").exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            for name, df in generate(30000).items():
                df.to_csv(data_dir / f"{name}.csv", index=False)
        source = "synthetic (Home Credit schema)"
    else:
        data_dir = Path(args.data_dir)
        source = "Home Credit Default Risk"

    print("Loading tables...")
    tables = load_tables(data_dir, sample=args.sample)
    print("Engineering features (SQL + pandas)...")
    df = build_feature_table(tables)
    nums, cats = split_columns(df)
    X, y = df[nums + cats], df[TARGET].values

    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=cfg.test_size, stratify=y, random_state=RANDOM_STATE)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_tmp, y_tmp, test_size=cfg.valid_size / (1 - cfg.test_size), stratify=y_tmp, random_state=RANDOM_STATE)

    print("Training logistic regression baseline...")
    lr = logistic_baseline(nums, cats).fit(X_train, y_train)
    pd_lr = lr.predict_proba(X_test)[:, 1]

    print("Training LightGBM with early stopping...")
    gbm = lightgbm_model()
    gbm.fit(to_lgb_frame(X_train, cats), y_train,
            eval_set=[(to_lgb_frame(X_valid, cats), y_valid)], eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)])
    raw_valid = gbm.predict_proba(to_lgb_frame(X_valid, cats))[:, 1]
    calibrator = PlattCalibrator().fit(raw_valid, y_valid)
    pd_valid = calibrator.transform(raw_valid)
    pd_gbm = calibrator.transform(gbm.predict_proba(to_lgb_frame(X_test, cats))[:, 1])

    metrics = {"Logistic regression": credit_metrics(y_test, pd_lr).as_dict(),
               "LightGBM": credit_metrics(y_test, pd_gbm).as_dict()}
    for k, v in metrics.items():
        print(f"  {k:<20} AUC {v['AUC']:.3f}  Gini {v['Gini']:.3f}  KS {v['KS']:.3f}")

    print("Tiering, pricing and simulating cut-offs...")
    tiers = calibrate_tiers(pd_valid, cfg.tiers)
    scored = pd.DataFrame({ID: df.loc[X_test.index, ID].values, "pd": pd_gbm, "default": y_test,
                           "exposure": X_test["AMT_CREDIT"].values,
                           "monthly_income": X_test["AMT_INCOME_TOTAL"].values / 12})
    scored["tier"] = assign_tier(scored["pd"].values, tiers)
    scored["apr"] = risk_based_apr(scored["pd"], cfg.pricing)
    limits = {t.name: t.limit_income_multiple for t in tiers}
    tenure = {t.name: t.max_tenure_months for t in tiers}
    scored["credit_limit"] = scored["tier"].map(limits) * scored["monthly_income"]
    scored["max_tenure_months"] = scored["tier"].map(tenure)
    scored.loc[scored["tier"] == "E", "apr"] = np.nan

    summary = tier_summary(scored, tiers)
    sim = cutoff_simulation(scored["pd"], scored["default"], scored["exposure"], cfg.pricing)
    chosen = recommend_cutoff(sim, cfg.max_expected_loss_rate, max_pd=tiers[-2].max_pd)  # tier E is declined

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    summary.to_csv(out / "tier_summary.csv", index=False)
    sim.to_csv(out / "cutoff_simulation.csv", index=False)
    scored.head(200).to_csv(out / "scored_sample.csv", index=False)
    report.roc_chart(y_test, {"Logistic regression": pd_lr, "LightGBM": pd_gbm}, out / "roc.png")
    report.tier_chart(summary, out / "tier_default_rates.png")
    report.tradeoff_chart(sim, chosen, out / "approval_vs_loss.png")
    imp = pd.Series(gbm.booster_.feature_importance("gain"), index=gbm.booster_.feature_name())
    report.importance_chart(imp, out / "feature_importance.png")
    meta = {"source": source, "applicants": int(len(df)), "default_rate": float(y.mean()),
            "thin_file_share": float(df["THIN_FILE"].mean())}
    report.write_summary(out, meta, metrics, summary, chosen)
    print(f"Done. See {out}/summary.md")


if __name__ == "__main__":
    main()
