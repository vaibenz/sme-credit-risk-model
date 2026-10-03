import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import DEFAULT_TIERS, PricingAssumptions  # noqa: E402
from credit_risk.features import application_features, run_sql  # noqa: E402
from credit_risk.policy import cutoff_simulation, recommend_cutoff, risk_based_apr  # noqa: E402
from credit_risk.tiers import assign_tier  # noqa: E402


def test_apr_increases_with_pd_and_respects_cap():
    p = PricingAssumptions()
    apr = risk_based_apr([0.0, 0.02, 0.05, 0.5], p)
    assert np.isclose(apr[0], p.cost_of_funds + p.operating_cost + p.target_margin)
    assert apr[1] < apr[2]
    assert np.isnan(apr[3])          # far above the cap, so not priceable


def test_tier_assignment_boundaries():
    tiers = assign_tier(np.array([0.01, 0.03, 0.031, 0.5]), DEFAULT_TIERS)
    assert list(tiers) == ["A", "A", "B", "E"]


def test_cutoff_simulation_is_monotone_in_approval():
    rng = np.random.default_rng(0)
    pd_scores = rng.uniform(0, 0.4, 1000)
    defaults = rng.random(1000) < pd_scores
    sim = cutoff_simulation(pd_scores, defaults, np.ones(1000), PricingAssumptions())
    assert sim["approval_rate"].is_monotonic_increasing
    chosen = recommend_cutoff(sim, 0.05)
    assert chosen["expected_loss_rate"] <= 0.05


def test_installment_sql_late_ratio():
    inst = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 1, 2], "SK_ID_PREV": [10, 10, 11, 20],
        "DAYS_INSTALMENT": [-100, -70, -40, -50], "DAYS_ENTRY_PAYMENT": [-105, -60, -40, -10],
        "AMT_INSTALMENT": [100, 100, 100, 50], "AMT_PAYMENT": [100, 80, 100, 50],
    })
    out = run_sql({"installments": inst}, "installments_features.sql").set_index("SK_ID_CURR")
    assert np.isclose(out.loc[1, "inst_late_ratio"], 1 / 3)
    assert np.isclose(out.loc[1, "inst_underpay_ratio"], 1 / 3)
    assert out.loc[2, "inst_max_days_late"] == 40


def test_unemployed_placeholder_is_handled():
    app = pd.DataFrame({"SK_ID_CURR": [1], "DAYS_BIRTH": [-10000], "DAYS_EMPLOYED": [365243],
                        "AMT_INCOME_TOTAL": [100.0], "AMT_CREDIT": [300.0], "AMT_ANNUITY": [30.0],
                        "AMT_GOODS_PRICE": [280.0], "EXT_SOURCE_1": [np.nan], "EXT_SOURCE_2": [0.5],
                        "EXT_SOURCE_3": [0.4]})
    f = application_features(app)
    assert f.loc[0, "FLAG_NOT_EMPLOYED"] == 1 and np.isnan(f.loc[0, "YEARS_EMPLOYED"])
