import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.policy import recommend_cutoff  # noqa: E402


def test_cutoff_never_reaches_declined_tier():
    sim = pd.DataFrame({"pd_cutoff": [0.05, 0.10, 0.20, 0.50],
                        "approval_rate": [0.5, 0.7, 0.9, 1.0],
                        "expected_loss_rate": [0.01, 0.02, 0.025, 0.028]})
    chosen = recommend_cutoff(sim, max_loss_rate=0.03, max_pd=0.15)
    assert chosen["pd_cutoff"] == 0.10
