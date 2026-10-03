"""Feature engineering. Behavioural aggregates are written in SQL and run on an in-memory SQLite database."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ID

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"


def run_sql(source, sql_file: str) -> pd.DataFrame:
    """Run a query from sql/ against either a SQLite file path or a dict of DataFrames."""
    query = (SQL_DIR / sql_file).read_text()
    conn = sqlite3.connect(str(source) if not isinstance(source, dict) else ":memory:")
    try:
        if isinstance(source, dict):
            for name, df in source.items():
                df.to_sql(name, conn, index=False)
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def application_features(app: pd.DataFrame) -> pd.DataFrame:
    df = app.copy()
    # DAYS_EMPLOYED uses 365243 as a placeholder for "not employed": keep a flag, treat value as missing.
    not_emp = df["DAYS_EMPLOYED"] == 365243
    df["FLAG_NOT_EMPLOYED"] = not_emp.astype(int)
    df.loc[not_emp, "DAYS_EMPLOYED"] = np.nan
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365.25
    df["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM_MONTHS"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["GOODS_TO_CREDIT"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_MISSING"] = ext.isna().sum(axis=1)
    return df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED"])


def build_feature_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = application_features(tables["application"])
    inst = run_sql(tables["db"], "installments_features.sql")
    bureau = run_sql(tables["db"], "bureau_features.sql")

    df = base.merge(inst, on=ID, how="left").merge(bureau, on=ID, how="left")
    df["HAS_REPAYMENT_HISTORY"] = df["inst_count"].notna().astype(int)
    df["THIN_FILE"] = df["bureau_count"].isna().astype(int)
    df[["bureau_count", "bureau_active_count"]] = df[["bureau_count", "bureau_active_count"]].fillna(0)
    return df
