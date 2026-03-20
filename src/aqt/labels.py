from __future__ import annotations

import pandas as pd


def add_labels(df: pd.DataFrame, primary_horizon: int, secondary_horizon: int) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("symbol", group_keys=False, observed=True)

    out[f"future_return_{primary_horizon}d"] = (
        g["close"].shift(-primary_horizon) / g["open"].shift(-1) - 1.0
    )
    out[f"future_return_{secondary_horizon}d"] = (
        g["close"].shift(-secondary_horizon) / g["open"].shift(-1) - 1.0
    )
    out[f"future_return_{primary_horizon}d"] = out[f"future_return_{primary_horizon}d"].astype("float32")
    out[f"future_return_{secondary_horizon}d"] = out[f"future_return_{secondary_horizon}d"].astype("float32")
    return out
