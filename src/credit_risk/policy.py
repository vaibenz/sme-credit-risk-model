"""Risk based pricing and approval cut-off simulation."""
import numpy as np
import pandas as pd

from .config import PricingAssumptions


def risk_based_apr(pd_scores, p: PricingAssumptions) -> np.ndarray:
    """Break-even APR so expected interest covers funding, servicing, margin and expected loss.

    With probability (1 - PD) the borrower repays principal plus interest r; with probability PD
    the lender recovers (1 - LGD). Setting expected return equal to cof + opex + margin gives
        r = (cof + opex + margin + PD * LGD) / (1 - PD).
    Loans whose required rate exceeds the responsible lending cap are not priceable (NaN).
    """
    pd_scores = np.asarray(pd_scores, dtype=float)
    r = (p.cost_of_funds + p.operating_cost + p.target_margin + pd_scores * p.lgd) / (1 - pd_scores)
    return np.where(r <= p.max_apr, r, np.nan)


def cutoff_simulation(pd_scores, defaults, exposure, p: PricingAssumptions, cutoffs=None) -> pd.DataFrame:
    """For each PD cut-off: approval rate, bad rate in the approved book, expected and realised loss rate."""
    pd_scores = np.asarray(pd_scores, dtype=float)
    defaults = np.asarray(defaults)
    exposure = np.asarray(exposure, dtype=float)
    if cutoffs is None:
        cutoffs = np.quantile(pd_scores, np.linspace(0.10, 1.0, 19))
    rows = []
    for c in cutoffs:
        ok = pd_scores <= c
        ead = exposure[ok].sum()
        rows.append({
            "pd_cutoff": float(c),
            "approval_rate": float(ok.mean()),
            "bad_rate_approved": float(defaults[ok].mean()) if ok.any() else np.nan,
            "expected_loss_rate": float((pd_scores[ok] * p.lgd * exposure[ok]).sum() / ead) if ead else np.nan,
            "realised_loss_rate": float((defaults[ok] * p.lgd * exposure[ok]).sum() / ead) if ead else np.nan,
        })
    return pd.DataFrame(rows)


def recommend_cutoff(sim: pd.DataFrame, max_loss_rate: float, max_pd: float = 1.0) -> pd.Series:
    """Highest approval rate whose expected loss rate stays within risk appetite,
    never approving beyond the PD ceiling of the last lending tier."""
    within = sim[(sim["expected_loss_rate"] <= max_loss_rate) & (sim["pd_cutoff"] <= max_pd)]
    if within.empty:
        return sim.sort_values("approval_rate").iloc[0]
    return within.sort_values("approval_rate").iloc[-1]
