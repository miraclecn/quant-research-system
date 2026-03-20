from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "vol_10",
    "vol_20",
    "amp_5",
    "amp_20",
    "turnover_5",
    "turnover_20",
    "amount_ratio_5_20",
    "amount_ratio_1_20",
    "close_to_ma_5",
    "close_to_ma_8",
    "close_to_ma_13",
    "close_to_ma_16",
    "close_to_ma_20",
    "close_to_ma_21",
    "close_to_ma_32",
    "close_to_ma_34",
    "close_to_ma_55",
    "close_to_ma_60",
    "gap_1",
    "intraday_ret_1",
    "overnight_ret_5",
    "trend_5_20",
    "trend_5_13",
    "trend_8_16",
    "trend_8_21",
    "trend_13_34",
    "trend_16_48",
    "trend_20_60",
    "trend_21_55",
    "vol_ratio_10_20",
    "turnover_ratio_5_20",
    "channel_pos_20",
    "channel_pos_16",
    "close_to_high_20",
    "close_to_high_16",
    "close_to_low_20",
    "close_to_low_16",
    "channel_pos_60",
    "distance_to_low_20",
    "distance_to_high_60",
    "range_pct_1",
    "vol_compress_5_20",
    "vol_compress_10_60",
    "ret_reversal_5_20",
    "ret_reversal_10_60",
    "turnover_shock_1_20",
    "turnover_shock_5_60",
    "price_volume_div_5",
    "price_volume_div_20",
    "rebound_from_low_20",
    "amount_trend_5",
    "float_mv_log",
]


def _infer_factor_family(feature: str) -> str:
    if feature.startswith("close_to_ma_") or feature.startswith("trend_"):
        return "sma"
    if feature.startswith("ret_") or feature in {"gap_1", "intraday_ret_1", "overnight_ret_5", "rebound_from_low_20"}:
        return "reversal"
    if feature.startswith("turnover_") or feature.startswith("amount_") or feature.startswith("price_volume_div_"):
        return "liquidity"
    if feature.startswith("vol_") or feature.startswith("amp_") or feature.startswith("range_") or feature.startswith("vol_compress_"):
        return "volatility"
    if feature.startswith("channel_") or feature.startswith("close_to_high_") or feature.startswith("close_to_low_") or feature.startswith("distance_to_"):
        return "path"
    if feature.startswith("float_mv"):
        return "size"
    return "misc"


def _infer_factor_expression(feature: str) -> str:
    if match := re.fullmatch(r"close_to_ma_(\d+)", feature):
        window = int(match.group(1))
        return f"close / ma({window}) - 1"
    if match := re.fullmatch(r"trend_(\d+)_(\d+)", feature):
        fast, slow = (int(group) for group in match.groups())
        return f"ma({fast}) / ma({slow}) - 1"
    if match := re.fullmatch(r"ret_(\d+)", feature):
        window = int(match.group(1))
        return f"pct_change(close, {window})"
    if match := re.fullmatch(r"ret_reversal_(\d+)_(\d+)", feature):
        fast, slow = (int(group) for group in match.groups())
        return f"ret({fast}) - ret({slow})"
    if match := re.fullmatch(r"vol_(\d+)", feature):
        window = int(match.group(1))
        return f"std(daily_ret, {window})"
    if match := re.fullmatch(r"amp_(\d+)", feature):
        window = int(match.group(1))
        return f"rolling_high({window}) / rolling_low({window}) - 1"
    if match := re.fullmatch(r"turnover_(\d+)", feature):
        window = int(match.group(1))
        return f"mean(turnover_rate, {window})"
    return feature


def _infer_factor_params(feature: str) -> dict:
    params = [int(value) for value in re.findall(r"\d+", feature)]
    if not params:
        return {}
    if len(params) == 1:
        return {"window": params[0]}
    return {"windows": params}


def build_factor_registry(features: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for feature in features or FEATURE_COLUMNS:
        params = _infer_factor_params(feature)
        rows.append(
            {
                "feature": feature,
                "family": _infer_factor_family(feature),
                "expression": _infer_factor_expression(feature),
                "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                "direction_hint": "unknown",
                "status": "candidate",
                "notes": "",
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "feature"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("symbol", group_keys=False, observed=True)

    out["ret_1"] = g["close"].pct_change(1)
    out["ret_3"] = g["close"].pct_change(3)
    out["ret_5"] = g["close"].pct_change(5)
    out["ret_10"] = g["close"].pct_change(10)
    out["ret_20"] = g["close"].pct_change(20)
    out["ret_60"] = g["close"].pct_change(60)

    daily_ret = g["close"].pct_change()
    out["vol_10"] = daily_ret.groupby(out["symbol"], observed=True).rolling(10).std().reset_index(level=0, drop=True)
    out["vol_20"] = daily_ret.groupby(out["symbol"], observed=True).rolling(20).std().reset_index(level=0, drop=True)

    out["amp_5"] = (g["high"].rolling(5).max().reset_index(level=0, drop=True) /
                    g["low"].rolling(5).min().reset_index(level=0, drop=True) - 1.0)
    out["amp_20"] = (g["high"].rolling(20).max().reset_index(level=0, drop=True) /
                     g["low"].rolling(20).min().reset_index(level=0, drop=True) - 1.0)

    out["turnover_5"] = g["turnover_rate"].rolling(5).mean().reset_index(level=0, drop=True)
    out["turnover_20"] = g["turnover_rate"].rolling(20).mean().reset_index(level=0, drop=True)
    amount_ma_5 = g["amount"].rolling(5).mean().reset_index(level=0, drop=True)
    amount_ma_20 = g["amount"].rolling(20).mean().reset_index(level=0, drop=True)
    out["amount_ratio_5_20"] = amount_ma_5 / amount_ma_20.replace(0, np.nan)
    out["amount_ratio_1_20"] = out["amount"] / amount_ma_20.replace(0, np.nan) - 1.0

    ma_5 = g["close"].rolling(5).mean().reset_index(level=0, drop=True)
    ma_8 = g["close"].rolling(8).mean().reset_index(level=0, drop=True)
    ma_13 = g["close"].rolling(13).mean().reset_index(level=0, drop=True)
    ma_16 = g["close"].rolling(16).mean().reset_index(level=0, drop=True)
    ma_20 = g["close"].rolling(20).mean().reset_index(level=0, drop=True)
    ma_21 = g["close"].rolling(21).mean().reset_index(level=0, drop=True)
    ma_32 = g["close"].rolling(32).mean().reset_index(level=0, drop=True)
    ma_34 = g["close"].rolling(34).mean().reset_index(level=0, drop=True)
    ma_48 = g["close"].rolling(48).mean().reset_index(level=0, drop=True)
    ma_55 = g["close"].rolling(55).mean().reset_index(level=0, drop=True)
    ma_60 = g["close"].rolling(60).mean().reset_index(level=0, drop=True)
    out["close_to_ma_5"] = out["close"] / ma_5 - 1.0
    out["close_to_ma_8"] = out["close"] / ma_8 - 1.0
    out["close_to_ma_13"] = out["close"] / ma_13 - 1.0
    out["close_to_ma_16"] = out["close"] / ma_16 - 1.0
    out["close_to_ma_20"] = out["close"] / ma_20 - 1.0
    out["close_to_ma_21"] = out["close"] / ma_21 - 1.0
    out["close_to_ma_32"] = out["close"] / ma_32 - 1.0
    out["close_to_ma_34"] = out["close"] / ma_34 - 1.0
    out["close_to_ma_55"] = out["close"] / ma_55 - 1.0
    out["close_to_ma_60"] = out["close"] / ma_60 - 1.0

    prev_close = g["close"].shift(1)
    out["gap_1"] = out["open"] / prev_close.replace(0, np.nan) - 1.0
    out["intraday_ret_1"] = out["close"] / out["open"].replace(0, np.nan) - 1.0
    out["overnight_ret_5"] = out["gap_1"].groupby(out["symbol"], observed=True).rolling(5).mean().reset_index(level=0, drop=True)
    out["trend_5_20"] = ma_5 / ma_20.replace(0, np.nan) - 1.0
    out["trend_5_13"] = ma_5 / ma_13.replace(0, np.nan) - 1.0
    out["trend_8_16"] = ma_8 / ma_16.replace(0, np.nan) - 1.0
    out["trend_8_21"] = ma_8 / ma_21.replace(0, np.nan) - 1.0
    out["trend_13_34"] = ma_13 / ma_34.replace(0, np.nan) - 1.0
    out["trend_16_48"] = ma_16 / ma_48.replace(0, np.nan) - 1.0
    out["trend_20_60"] = ma_20 / ma_60.replace(0, np.nan) - 1.0
    out["trend_21_55"] = ma_21 / ma_55.replace(0, np.nan) - 1.0
    out["vol_ratio_10_20"] = out["vol_10"] / out["vol_20"].replace(0, np.nan) - 1.0
    out["turnover_ratio_5_20"] = out["turnover_5"] / out["turnover_20"].replace(0, np.nan) - 1.0

    rolling_high_16 = g["high"].rolling(16).max().reset_index(level=0, drop=True)
    rolling_low_16 = g["low"].rolling(16).min().reset_index(level=0, drop=True)
    channel_width_16 = (rolling_high_16 - rolling_low_16).replace(0, np.nan)
    rolling_high_20 = g["high"].rolling(20).max().reset_index(level=0, drop=True)
    rolling_low_20 = g["low"].rolling(20).min().reset_index(level=0, drop=True)
    channel_width_20 = (rolling_high_20 - rolling_low_20).replace(0, np.nan)
    out["channel_pos_16"] = (out["close"] - rolling_low_16) / channel_width_16 - 0.5
    out["channel_pos_20"] = (out["close"] - rolling_low_20) / channel_width_20 - 0.5
    out["close_to_high_16"] = out["close"] / rolling_high_16.replace(0, np.nan) - 1.0
    out["close_to_high_20"] = out["close"] / rolling_high_20.replace(0, np.nan) - 1.0
    out["close_to_low_16"] = out["close"] / rolling_low_16.replace(0, np.nan) - 1.0
    out["close_to_low_20"] = out["close"] / rolling_low_20.replace(0, np.nan) - 1.0
    rolling_high_60 = g["high"].rolling(60).max().reset_index(level=0, drop=True)
    rolling_low_60 = g["low"].rolling(60).min().reset_index(level=0, drop=True)
    channel_width_60 = (rolling_high_60 - rolling_low_60).replace(0, np.nan)
    out["channel_pos_60"] = (out["close"] - rolling_low_60) / channel_width_60 - 0.5
    out["distance_to_low_20"] = (out["close"] - rolling_low_20) / rolling_low_20.replace(0, np.nan)
    out["distance_to_high_60"] = out["close"] / rolling_high_60.replace(0, np.nan) - 1.0
    out["range_pct_1"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["vol_compress_5_20"] = -(out["vol_10"] / out["vol_20"].replace(0, np.nan) - 1.0).abs()
    vol_60 = daily_ret.groupby(out["symbol"], observed=True).rolling(60).std().reset_index(level=0, drop=True)
    out["vol_compress_10_60"] = -(out["vol_10"] / vol_60.replace(0, np.nan) - 1.0).abs()
    out["ret_reversal_5_20"] = out["ret_5"] - out["ret_20"]
    out["ret_reversal_10_60"] = out["ret_10"] - out["ret_60"]
    out["turnover_shock_1_20"] = out["turnover_rate"] / out["turnover_20"].replace(0, np.nan) - 1.0
    turnover_60 = g["turnover_rate"].rolling(60).mean().reset_index(level=0, drop=True)
    out["turnover_shock_5_60"] = out["turnover_5"] / turnover_60.replace(0, np.nan) - 1.0
    out["price_volume_div_5"] = out["ret_5"] - (amount_ma_5 / amount_ma_20.replace(0, np.nan) - 1.0)
    amount_ma_60 = g["amount"].rolling(60).mean().reset_index(level=0, drop=True)
    out["price_volume_div_20"] = out["ret_20"] - (amount_ma_20 / amount_ma_60.replace(0, np.nan) - 1.0)
    out["rebound_from_low_20"] = out["close"] / rolling_low_20.replace(0, np.nan) - 1.0 - out["ret_20"]
    out["amount_trend_5"] = amount_ma_5 / amount_ma_20.replace(0, np.nan) - 1.0

    out["float_mv_log"] = np.log(out["float_mv"].clip(lower=1.0))

    for col in FEATURE_COLUMNS:
        out[col] = out[col].astype("float32")

    return out
