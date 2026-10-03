"""Loading the Home Credit tables (or the synthetic stand-ins with the same schema)."""
from pathlib import Path

import pandas as pd

from .config import APPLICATION_CATEGORICAL, APPLICATION_NUMERIC, ID, TARGET

INST_COLS = ["SK_ID_PREV", "SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"]
BUREAU_COLS = ["SK_ID_CURR", "CREDIT_ACTIVE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "CREDIT_DAY_OVERDUE"]


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Home Credit Default Risk data into this folder "
            "(see data/README.md) or run with --synthetic."
        )
    return path


def load_tables(data_dir: str | Path, sample: int | None = None) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    app = pd.read_csv(
        _require(data_dir / "application_train.csv"),
        usecols=[ID, TARGET, *APPLICATION_NUMERIC, *APPLICATION_CATEGORICAL],
    )
    if sample:
        app = app.sample(n=min(sample, len(app)), random_state=0)
    ids = set(app[ID])

    inst = pd.read_csv(_require(data_dir / "installments_payments.csv"), usecols=INST_COLS)
    bureau = pd.read_csv(_require(data_dir / "bureau.csv"), usecols=BUREAU_COLS)
    return {
        "application": app.reset_index(drop=True),
        "installments": inst[inst[ID].isin(ids)].reset_index(drop=True),
        "bureau": bureau[bureau[ID].isin(ids)].reset_index(drop=True),
    }
