"""Turning probabilities of default into five risk tiers with credit limits and tenures."""
import numpy as np
import pandas as pd

from .config import TIER_QUANTILES, TierPolicy


def calibrate_tiers(pd_valid: np.ndarray, tiers: list[TierPolicy]) -> list[TierPolicy]:
    """Set each tier's PD ceiling from validation quantiles so tier sizes stay stable."""
    bounds = list(np.quantile(pd_valid, TIER_QUANTILES)) + [1.0]
    return [TierPolicy(t.name, float(b), t.limit_income_multiple, t.max_tenure_months)
            for t, b in zip(tiers, bounds)]


def assign_tier(pd_scores: np.ndarray, tiers: list[TierPolicy]) -> np.ndarray:
    bounds = np.array([t.max_pd for t in tiers])
    idx = np.clip(np.searchsorted(bounds, pd_scores, side="left"), 0, len(tiers) - 1)
    return np.array([tiers[i].name for i in idx])


def tier_summary(df: pd.DataFrame, tiers: list[TierPolicy]) -> pd.DataFrame:
    """df needs columns: tier, pd, default, apr."""
    g = df.groupby("tier").agg(applicants=("pd", "size"), avg_pd=("pd", "mean"),
                               observed_default_rate=("default", "mean"), avg_apr=("apr", "mean"))
    g["share"] = g["applicants"] / g["applicants"].sum()
    policy = pd.DataFrame([{"tier": t.name, "pd_ceiling": t.max_pd,
                            "limit_x_monthly_income": t.limit_income_multiple,
                            "max_tenure_months": t.max_tenure_months} for t in tiers]).set_index("tier")
    return policy.join(g).reset_index()
