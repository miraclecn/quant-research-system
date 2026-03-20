from __future__ import annotations

import pandas as pd

from aqt.config import DataConfig


def apply_universe_filters(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    out = df.copy()
    amount_20 = (
        out.groupby("symbol", observed=True)["amount"]
        .rolling(20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out["amount_20d"] = amount_20

    out["tradable_universe"] = (
        out["in_universe"]
        & ~out["is_st"]
        & (out["listed_days"] >= cfg.min_listed_days)
        & (out["amount_20d"] >= cfg.min_amount_20d)
    )
    out["amount_20d"] = out["amount_20d"].astype("float32")
    return out
