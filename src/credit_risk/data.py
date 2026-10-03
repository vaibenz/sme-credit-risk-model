"""Loading the Home Credit tables (or the synthetic stand-ins with the same schema).

The instalments table has 13.6 million rows (about 720 MB on disk), so it is streamed in chunks
into a temporary SQLite database instead of being held in memory. That keeps the full run inside
the RAM of an ordinary laptop.
"""
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from .config import APPLICATION_CATEGORICAL, APPLICATION_NUMERIC, ID, TARGET

INST_DTYPES = {"SK_ID_PREV": "int64", "SK_ID_CURR": "int64", "DAYS_INSTALMENT": "float32",
               "DAYS_ENTRY_PAYMENT": "float32", "AMT_INSTALMENT": "float32", "AMT_PAYMENT": "float32"}
BUREAU_DTYPES = {"SK_ID_CURR": "int64", "CREDIT_ACTIVE": "object", "AMT_CREDIT_SUM": "float32",
                 "AMT_CREDIT_SUM_DEBT": "float32", "CREDIT_DAY_OVERDUE": "float32"}
CHUNK = 1_000_000


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Home Credit Default Risk data into this folder "
            "(see data/README.md) or run with --synthetic."
        )
    return path


def _stream_to_sqlite(csv: Path, table: str, dtypes: dict, ids: set, conn: sqlite3.Connection) -> int:
    rows = 0
    for chunk in pd.read_csv(csv, usecols=list(dtypes), dtype=dtypes, chunksize=CHUNK):
        chunk = chunk[chunk[ID].isin(ids)]
        chunk.to_sql(table, conn, index=False, if_exists="append")
        rows += len(chunk)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_id ON {table}({ID})")
    return rows


def load_tables(data_dir: str | Path, sample: int | None = None) -> dict:
    """Returns the application table as a DataFrame and a path to a SQLite file holding
    the instalments and bureau tables for those applicants."""
    data_dir = Path(data_dir)
    app = pd.read_csv(
        _require(data_dir / "application_train.csv"),
        usecols=[ID, TARGET, *APPLICATION_NUMERIC, *APPLICATION_CATEGORICAL],
    )
    if sample:
        app = app.sample(n=min(sample, len(app)), random_state=0)
    ids = set(app[ID])

    db_path = Path(tempfile.mkdtemp()) / "behaviour.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        n_inst = _stream_to_sqlite(_require(data_dir / "installments_payments.csv"), "installments",
                                   INST_DTYPES, ids, conn)
        n_bur = _stream_to_sqlite(_require(data_dir / "bureau.csv"), "bureau", BUREAU_DTYPES, ids, conn)
        conn.commit()
    finally:
        conn.close()
    print(f"  {len(app):,} applications, {n_inst:,} instalment rows, {n_bur:,} bureau rows")
    return {"application": app.reset_index(drop=True), "db": db_path}
