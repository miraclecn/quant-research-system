from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

EXTENDED_CLOSE_TO_MA_WINDOWS = [10, 89, 120]
EXTENDED_TREND_PAIRS = [(10, 20), (5, 34), (13, 55), (20, 120), (34, 89), (55, 120)]
MA_SPREAD_TRIPLES = [(5, 20, 60), (5, 34, 89), (8, 21, 55), (10, 20, 60), (13, 34, 89), (20, 60, 120)]
EXTENDED_VOL_RATIO_PAIRS = [(5, 20), (20, 60)]
EXTENDED_TURNOVER_RATIO_PAIRS = [(5, 60), (20, 60)]
EXTENDED_CHANNEL_WINDOWS = [120]
EXTENDED_DISTANCE_HIGH_WINDOWS = [20, 120]
EXTENDED_DISTANCE_LOW_WINDOWS = [60, 120]
PRICE_ZSCORE_WINDOWS = [20, 60, 120]
EFFICIENCY_WINDOWS = [10, 20, 60]
ATR_WINDOWS = [14, 20]
ATR_RATIO_PAIRS = [(14, 60), (20, 60)]
BREAKOUT_WINDOWS = [20, 60, 120]
AMOUNT_ZSCORE_WINDOWS = [20, 60]
TURNOVER_ZSCORE_WINDOWS = [20, 60]

FUNDAMENTAL_FEATURE_COLUMNS = [
    "value_earnings_yield",
    "value_book_to_price",
    "value_sales_yield",
    "value_dividend_yield",
    "quality_roe",
    "quality_roe_dt",
    "quality_roa",
    "quality_roic",
    "quality_grossprofit_margin",
    "quality_netprofit_margin",
    "quality_ocfps",
    "quality_cfps",
    "quality_profit_dedt_to_mv",
    "leverage_debt_to_assets",
    "solvency_current_ratio",
    "solvency_quick_ratio",
    "growth_basic_eps_yoy",
    "growth_dt_eps_yoy",
    "growth_netprofit_yoy",
    "growth_dt_netprofit_yoy",
    "growth_ocf_yoy",
    "growth_revenue_yoy",
    "growth_operating_revenue_yoy",
    "growth_q_sales_yoy",
    "growth_q_op_qoq",
]


BASE_FEATURE_COLUMNS = [
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

EXTENDED_FEATURE_COLUMNS = [
    *[f"close_to_ma_{window}" for window in EXTENDED_CLOSE_TO_MA_WINDOWS],
    *[f"trend_{fast}_{slow}" for fast, slow in EXTENDED_TREND_PAIRS],
    *[f"ma_spread_{fast}_{mid}_{slow}" for fast, mid, slow in MA_SPREAD_TRIPLES],
    *[f"vol_ratio_{fast}_{slow}" for fast, slow in EXTENDED_VOL_RATIO_PAIRS],
    *[f"turnover_ratio_{fast}_{slow}" for fast, slow in EXTENDED_TURNOVER_RATIO_PAIRS],
    *[f"channel_pos_{window}" for window in EXTENDED_CHANNEL_WINDOWS],
    *[f"distance_to_high_{window}" for window in EXTENDED_DISTANCE_HIGH_WINDOWS],
    *[f"distance_to_low_{window}" for window in EXTENDED_DISTANCE_LOW_WINDOWS],
    *[f"price_zscore_{window}" for window in PRICE_ZSCORE_WINDOWS],
    *[f"efficiency_ratio_{window}" for window in EFFICIENCY_WINDOWS],
    *[f"atr_{window}" for window in ATR_WINDOWS],
    *[f"atr_ratio_{fast}_{slow}" for fast, slow in ATR_RATIO_PAIRS],
    *[f"breakout_ratio_{window}" for window in BREAKOUT_WINDOWS],
    *[f"amount_zscore_{window}" for window in AMOUNT_ZSCORE_WINDOWS],
    *[f"turnover_zscore_{window}" for window in TURNOVER_ZSCORE_WINDOWS],
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + EXTENDED_FEATURE_COLUMNS + FUNDAMENTAL_FEATURE_COLUMNS


def _infer_factor_family(feature: str) -> str:
    if feature.startswith("value_"):
        return "value"
    if feature.startswith("quality_"):
        return "quality"
    if feature.startswith("leverage_") or feature.startswith("solvency_"):
        return "quality"
    if feature.startswith("growth_"):
        return "growth"
    if feature.startswith("close_to_ma_") or feature.startswith("trend_"):
        return "sma"
    if feature.startswith("ma_spread_"):
        return "sma"
    if feature.startswith(("price_zscore_", "breakout_ratio_")):
        return "path"
    if feature.startswith("ret_") or feature in {"gap_1", "intraday_ret_1", "overnight_ret_5", "rebound_from_low_20"}:
        return "reversal"
    if feature.startswith("turnover_") or feature.startswith("amount_") or feature.startswith("price_volume_div_"):
        return "liquidity"
    if feature.startswith("vol_") or feature.startswith("amp_") or feature.startswith("range_") or feature.startswith("vol_compress_") or feature.startswith(("atr_", "efficiency_ratio_")):
        return "volatility"
    if feature.startswith("channel_") or feature.startswith("close_to_high_") or feature.startswith("close_to_low_") or feature.startswith("distance_to_"):
        return "path"
    if feature.startswith("float_mv"):
        return "size"
    return "misc"


def _infer_factor_expression(feature: str) -> str:
    fundamental_expressions = {
        "value_earnings_yield": "1 / pe_ttm",
        "value_book_to_price": "1 / pb",
        "value_sales_yield": "1 / ps_ttm",
        "value_dividend_yield": "dv_ttm",
        "quality_roe": "roe",
        "quality_roe_dt": "roe_dt",
        "quality_roa": "roa",
        "quality_roic": "roic",
        "quality_grossprofit_margin": "grossprofit_margin",
        "quality_netprofit_margin": "netprofit_margin",
        "quality_ocfps": "ocfps",
        "quality_cfps": "cfps",
        "quality_profit_dedt_to_mv": "profit_dedt / float_mv",
        "leverage_debt_to_assets": "-debt_to_assets",
        "solvency_current_ratio": "current_ratio",
        "solvency_quick_ratio": "quick_ratio",
        "growth_basic_eps_yoy": "basic_eps_yoy",
        "growth_dt_eps_yoy": "dt_eps_yoy",
        "growth_netprofit_yoy": "netprofit_yoy",
        "growth_dt_netprofit_yoy": "dt_netprofit_yoy",
        "growth_ocf_yoy": "ocf_yoy",
        "growth_revenue_yoy": "tr_yoy",
        "growth_operating_revenue_yoy": "or_yoy",
        "growth_q_sales_yoy": "q_sales_yoy",
        "growth_q_op_qoq": "q_op_qoq",
    }
    if feature in fundamental_expressions:
        return fundamental_expressions[feature]
    if match := re.fullmatch(r"close_to_ma_(\d+)", feature):
        window = int(match.group(1))
        return f"close / ma({window}) - 1"
    if match := re.fullmatch(r"trend_(\d+)_(\d+)", feature):
        fast, slow = (int(group) for group in match.groups())
        return f"ma({fast}) / ma({slow}) - 1"
    if match := re.fullmatch(r"ma_spread_(\d+)_(\d+)_(\d+)", feature):
        fast, mid, slow = (int(group) for group in match.groups())
        return f"(ma({fast}) - ma({mid})) / ma({slow})"
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
    if match := re.fullmatch(r"vol_ratio_(\d+)_(\d+)", feature):
        fast, slow = (int(group) for group in match.groups())
        return f"vol({fast}) / vol({slow}) - 1"
    if match := re.fullmatch(r"turnover_ratio_(\d+)_(\d+)", feature):
        fast, slow = (int(group) for group in match.groups())
        return f"turnover({fast}) / turnover({slow}) - 1"
    if match := re.fullmatch(r"channel_pos_(\d+)", feature):
        window = int(match.group(1))
        return f"(close - rolling_low({window})) / (rolling_high({window}) - rolling_low({window})) - 0.5"
    if match := re.fullmatch(r"distance_to_high_(\d+)", feature):
        window = int(match.group(1))
        return f"close / rolling_high({window}) - 1"
    if match := re.fullmatch(r"distance_to_low_(\d+)", feature):
        window = int(match.group(1))
        return f"(close - rolling_low({window})) / rolling_low({window})"
    if match := re.fullmatch(r"price_zscore_(\d+)", feature):
        window = int(match.group(1))
        return f"(close - ma({window})) / std(close, {window})"
    if match := re.fullmatch(r"efficiency_ratio_(\d+)", feature):
        window = int(match.group(1))
        return f"abs(close - close_lag({window})) / sum(abs(diff(close)), {window})"
    if match := re.fullmatch(r"atr_(\d+)", feature):
        window = int(match.group(1))
        return f"mean(true_range, {window}) / close"
    if match := re.fullmatch(r"atr_ratio_(\d+)_(\d+)", feature):
        fast, slow = (int(group) for group in match.groups())
        return f"atr({fast}) / atr({slow}) - 1"
    if match := re.fullmatch(r"breakout_ratio_(\d+)", feature):
        window = int(match.group(1))
        return f"(close - rolling_low({window})) / (rolling_high({window}) - rolling_low({window}))"
    if match := re.fullmatch(r"amount_zscore_(\d+)", feature):
        window = int(match.group(1))
        return f"(amount - mean(amount, {window})) / std(amount, {window})"
    if match := re.fullmatch(r"turnover_zscore_(\d+)", feature):
        window = int(match.group(1))
        return f"(turnover_rate - mean(turnover_rate, {window})) / std(turnover_rate, {window})"
    return feature


def _infer_factor_params(feature: str) -> dict:
    params = [int(value) for value in re.findall(r"\d+", feature)]
    if not params:
        return {}
    if len(params) == 1:
        return {"window": params[0]}
    return {"windows": params}


def _infer_factor_template(feature: str) -> str:
    if feature.startswith("value_"):
        return "fundamental_value"
    if feature.startswith("quality_"):
        return "fundamental_quality"
    if feature.startswith("leverage_"):
        return "fundamental_leverage"
    if feature.startswith("solvency_"):
        return "fundamental_solvency"
    if feature.startswith("growth_"):
        return "fundamental_growth"
    if feature.startswith("close_to_ma_"):
        return "close_to_ma"
    if feature.startswith("trend_"):
        return "trend_ratio"
    if feature.startswith("ma_spread_"):
        return "ma_spread"
    if feature.startswith("price_zscore_"):
        return "price_zscore"
    if re.fullmatch(r"ret_\d+", feature):
        return "return"
    if feature.startswith("ret_reversal_"):
        return "return_spread"
    if feature.startswith("vol_ratio_"):
        return "vol_ratio"
    if feature.startswith("efficiency_ratio_"):
        return "efficiency_ratio"
    if feature.startswith("atr_ratio_"):
        return "atr_ratio"
    if feature.startswith("atr_"):
        return "atr"
    if feature.startswith("turnover_ratio_"):
        return "turnover_ratio"
    if feature.startswith("channel_pos_"):
        return "channel_position"
    if feature.startswith("breakout_ratio_"):
        return "breakout_ratio"
    if feature.startswith("distance_to_high_"):
        return "distance_to_high"
    if feature.startswith("distance_to_low_"):
        return "distance_to_low"
    if feature.startswith("amount_zscore_"):
        return "amount_zscore"
    if feature.startswith("turnover_zscore_"):
        return "turnover_zscore"
    return "direct"


def _infer_factor_subfamily(feature: str) -> str:
    family = _infer_factor_family(feature)
    if family == "value":
        return "valuation"
    if family == "quality":
        if feature.startswith("leverage_"):
            return "leverage"
        if feature.startswith("solvency_"):
            return "solvency"
        return "profitability"
    if family == "growth":
        return "fundamental_growth"
    if family == "sma":
        if feature.startswith("close_to_ma_"):
            return "price_vs_ma"
        if feature.startswith("trend_"):
            return "ma_ratio"
        if feature.startswith("ma_spread_"):
            return "ma_spread"
    if family == "reversal":
        return "return_reversal"
    if family == "liquidity":
        if feature.startswith("amount_zscore_"):
            return "amount_anomaly"
        if feature.startswith("turnover_zscore_"):
            return "turnover_anomaly"
        return "turnover_amount"
    if family == "volatility":
        if feature.startswith("efficiency_ratio_"):
            return "trend_efficiency"
        if feature.startswith("atr_"):
            return "true_range"
        return "vol_amp"
    if family == "path":
        if feature.startswith("price_zscore_"):
            return "price_standardized"
        if feature.startswith("breakout_ratio_"):
            return "breakout"
        return "price_position"
    return family


def _infer_factor_dependencies(feature: str) -> list[str]:
    fundamental_dependencies = {
        "value_earnings_yield": ["pe_ttm"],
        "value_book_to_price": ["pb"],
        "value_sales_yield": ["ps_ttm"],
        "value_dividend_yield": ["dv_ttm"],
        "quality_roe": ["roe"],
        "quality_roe_dt": ["roe_dt"],
        "quality_roa": ["roa"],
        "quality_roic": ["roic"],
        "quality_grossprofit_margin": ["grossprofit_margin"],
        "quality_netprofit_margin": ["netprofit_margin"],
        "quality_ocfps": ["ocfps"],
        "quality_cfps": ["cfps"],
        "quality_profit_dedt_to_mv": ["profit_dedt", "float_mv"],
        "leverage_debt_to_assets": ["debt_to_assets"],
        "solvency_current_ratio": ["current_ratio"],
        "solvency_quick_ratio": ["quick_ratio"],
        "growth_basic_eps_yoy": ["basic_eps_yoy"],
        "growth_dt_eps_yoy": ["dt_eps_yoy"],
        "growth_netprofit_yoy": ["netprofit_yoy"],
        "growth_dt_netprofit_yoy": ["dt_netprofit_yoy"],
        "growth_ocf_yoy": ["ocf_yoy"],
        "growth_revenue_yoy": ["tr_yoy"],
        "growth_operating_revenue_yoy": ["or_yoy"],
        "growth_q_sales_yoy": ["q_sales_yoy"],
        "growth_q_op_qoq": ["q_op_qoq"],
    }
    if feature in fundamental_dependencies:
        return fundamental_dependencies[feature]
    if feature.startswith(("close_to_ma_", "trend_", "ma_spread_", "ret_", "channel_pos_", "distance_to_", "price_zscore_", "breakout_ratio_")):
        return ["close", "high", "low"]
    if feature.startswith(("turnover_", "amount_", "price_volume_div_")):
        return ["amount", "turnover_rate", "close"]
    if feature.startswith(("vol_", "amp_", "range_", "efficiency_ratio_", "atr_")):
        return ["close", "high", "low", "open"]
    if feature.startswith("float_mv"):
        return ["float_mv"]
    return ["close"]


def build_factor_registry(features: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for feature in features or FEATURE_COLUMNS:
        params = _infer_factor_params(feature)
        rows.append(
            {
                "feature": feature,
                "family": _infer_factor_family(feature),
                "subfamily": _infer_factor_subfamily(feature),
                "template": _infer_factor_template(feature),
                "expression": _infer_factor_expression(feature),
                "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                "data_dependencies": json.dumps(_infer_factor_dependencies(feature), ensure_ascii=False),
                "calc_source": "internal",
                "production_ready": True,
                "direction_hint": "unknown",
                "status": "candidate",
                "notes": "",
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "feature"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("symbol", group_keys=False, observed=True)
    features: dict[str, pd.Series] = {}

    features["ret_1"] = g["close"].pct_change(1)
    features["ret_3"] = g["close"].pct_change(3)
    features["ret_5"] = g["close"].pct_change(5)
    features["ret_10"] = g["close"].pct_change(10)
    features["ret_20"] = g["close"].pct_change(20)
    features["ret_60"] = g["close"].pct_change(60)

    daily_ret = g["close"].pct_change()
    features["vol_10"] = daily_ret.groupby(out["symbol"], observed=True).rolling(10).std().reset_index(level=0, drop=True)
    features["vol_20"] = daily_ret.groupby(out["symbol"], observed=True).rolling(20).std().reset_index(level=0, drop=True)
    vol_60 = daily_ret.groupby(out["symbol"], observed=True).rolling(60).std().reset_index(level=0, drop=True)

    features["amp_5"] = (g["high"].rolling(5).max().reset_index(level=0, drop=True) /
                         g["low"].rolling(5).min().reset_index(level=0, drop=True) - 1.0)
    features["amp_20"] = (g["high"].rolling(20).max().reset_index(level=0, drop=True) /
                          g["low"].rolling(20).min().reset_index(level=0, drop=True) - 1.0)

    features["turnover_5"] = g["turnover_rate"].rolling(5).mean().reset_index(level=0, drop=True)
    features["turnover_20"] = g["turnover_rate"].rolling(20).mean().reset_index(level=0, drop=True)
    amount_ma_5 = g["amount"].rolling(5).mean().reset_index(level=0, drop=True)
    amount_ma_20 = g["amount"].rolling(20).mean().reset_index(level=0, drop=True)
    features["amount_ratio_5_20"] = amount_ma_5 / amount_ma_20.replace(0, np.nan)
    features["amount_ratio_1_20"] = out["amount"] / amount_ma_20.replace(0, np.nan) - 1.0

    ma_5 = g["close"].rolling(5).mean().reset_index(level=0, drop=True)
    ma_8 = g["close"].rolling(8).mean().reset_index(level=0, drop=True)
    ma_10 = g["close"].rolling(10).mean().reset_index(level=0, drop=True)
    ma_13 = g["close"].rolling(13).mean().reset_index(level=0, drop=True)
    ma_16 = g["close"].rolling(16).mean().reset_index(level=0, drop=True)
    ma_20 = g["close"].rolling(20).mean().reset_index(level=0, drop=True)
    ma_21 = g["close"].rolling(21).mean().reset_index(level=0, drop=True)
    ma_32 = g["close"].rolling(32).mean().reset_index(level=0, drop=True)
    ma_34 = g["close"].rolling(34).mean().reset_index(level=0, drop=True)
    ma_48 = g["close"].rolling(48).mean().reset_index(level=0, drop=True)
    ma_55 = g["close"].rolling(55).mean().reset_index(level=0, drop=True)
    ma_60 = g["close"].rolling(60).mean().reset_index(level=0, drop=True)
    ma_89 = g["close"].rolling(89).mean().reset_index(level=0, drop=True)
    ma_120 = g["close"].rolling(120).mean().reset_index(level=0, drop=True)
    close_std_20 = g["close"].rolling(20).std().reset_index(level=0, drop=True)
    close_std_60 = g["close"].rolling(60).std().reset_index(level=0, drop=True)
    close_std_120 = g["close"].rolling(120).std().reset_index(level=0, drop=True)
    features["close_to_ma_5"] = out["close"] / ma_5 - 1.0
    features["close_to_ma_8"] = out["close"] / ma_8 - 1.0
    features["close_to_ma_10"] = out["close"] / ma_10 - 1.0
    features["close_to_ma_13"] = out["close"] / ma_13 - 1.0
    features["close_to_ma_16"] = out["close"] / ma_16 - 1.0
    features["close_to_ma_20"] = out["close"] / ma_20 - 1.0
    features["close_to_ma_21"] = out["close"] / ma_21 - 1.0
    features["close_to_ma_32"] = out["close"] / ma_32 - 1.0
    features["close_to_ma_34"] = out["close"] / ma_34 - 1.0
    features["close_to_ma_55"] = out["close"] / ma_55 - 1.0
    features["close_to_ma_60"] = out["close"] / ma_60 - 1.0
    features["close_to_ma_89"] = out["close"] / ma_89 - 1.0
    features["close_to_ma_120"] = out["close"] / ma_120 - 1.0

    prev_close = g["close"].shift(1)
    features["gap_1"] = out["open"] / prev_close.replace(0, np.nan) - 1.0
    features["intraday_ret_1"] = out["close"] / out["open"].replace(0, np.nan) - 1.0
    features["overnight_ret_5"] = features["gap_1"].groupby(out["symbol"], observed=True).rolling(5).mean().reset_index(level=0, drop=True)
    features["trend_5_20"] = ma_5 / ma_20.replace(0, np.nan) - 1.0
    features["trend_5_13"] = ma_5 / ma_13.replace(0, np.nan) - 1.0
    features["trend_8_16"] = ma_8 / ma_16.replace(0, np.nan) - 1.0
    features["trend_8_21"] = ma_8 / ma_21.replace(0, np.nan) - 1.0
    features["trend_10_20"] = ma_10 / ma_20.replace(0, np.nan) - 1.0
    features["trend_5_34"] = ma_5 / ma_34.replace(0, np.nan) - 1.0
    features["trend_13_34"] = ma_13 / ma_34.replace(0, np.nan) - 1.0
    features["trend_13_55"] = ma_13 / ma_55.replace(0, np.nan) - 1.0
    features["trend_16_48"] = ma_16 / ma_48.replace(0, np.nan) - 1.0
    features["trend_20_60"] = ma_20 / ma_60.replace(0, np.nan) - 1.0
    features["trend_20_120"] = ma_20 / ma_120.replace(0, np.nan) - 1.0
    features["trend_21_55"] = ma_21 / ma_55.replace(0, np.nan) - 1.0
    features["trend_34_89"] = ma_34 / ma_89.replace(0, np.nan) - 1.0
    features["trend_55_120"] = ma_55 / ma_120.replace(0, np.nan) - 1.0
    features["ma_spread_5_20_60"] = (ma_5 - ma_20) / ma_60.replace(0, np.nan)
    features["ma_spread_5_34_89"] = (ma_5 - ma_34) / ma_89.replace(0, np.nan)
    features["ma_spread_8_21_55"] = (ma_8 - ma_21) / ma_55.replace(0, np.nan)
    features["ma_spread_10_20_60"] = (ma_10 - ma_20) / ma_60.replace(0, np.nan)
    features["ma_spread_13_34_89"] = (ma_13 - ma_34) / ma_89.replace(0, np.nan)
    features["ma_spread_20_60_120"] = (ma_20 - ma_60) / ma_120.replace(0, np.nan)
    features["vol_ratio_10_20"] = features["vol_10"] / features["vol_20"].replace(0, np.nan) - 1.0
    vol_5 = daily_ret.groupby(out["symbol"], observed=True).rolling(5).std().reset_index(level=0, drop=True)
    features["vol_ratio_5_20"] = vol_5 / features["vol_20"].replace(0, np.nan) - 1.0
    features["vol_ratio_20_60"] = features["vol_20"] / vol_60.replace(0, np.nan) - 1.0
    features["turnover_ratio_5_20"] = features["turnover_5"] / features["turnover_20"].replace(0, np.nan) - 1.0

    rolling_high_16 = g["high"].rolling(16).max().reset_index(level=0, drop=True)
    rolling_low_16 = g["low"].rolling(16).min().reset_index(level=0, drop=True)
    channel_width_16 = (rolling_high_16 - rolling_low_16).replace(0, np.nan)
    rolling_high_20 = g["high"].rolling(20).max().reset_index(level=0, drop=True)
    rolling_low_20 = g["low"].rolling(20).min().reset_index(level=0, drop=True)
    channel_width_20 = (rolling_high_20 - rolling_low_20).replace(0, np.nan)
    features["channel_pos_16"] = (out["close"] - rolling_low_16) / channel_width_16 - 0.5
    features["channel_pos_20"] = (out["close"] - rolling_low_20) / channel_width_20 - 0.5
    features["close_to_high_16"] = out["close"] / rolling_high_16.replace(0, np.nan) - 1.0
    features["close_to_high_20"] = out["close"] / rolling_high_20.replace(0, np.nan) - 1.0
    features["close_to_low_16"] = out["close"] / rolling_low_16.replace(0, np.nan) - 1.0
    features["close_to_low_20"] = out["close"] / rolling_low_20.replace(0, np.nan) - 1.0
    rolling_high_60 = g["high"].rolling(60).max().reset_index(level=0, drop=True)
    rolling_low_60 = g["low"].rolling(60).min().reset_index(level=0, drop=True)
    rolling_high_120 = g["high"].rolling(120).max().reset_index(level=0, drop=True)
    rolling_low_120 = g["low"].rolling(120).min().reset_index(level=0, drop=True)
    channel_width_60 = (rolling_high_60 - rolling_low_60).replace(0, np.nan)
    channel_width_120 = (rolling_high_120 - rolling_low_120).replace(0, np.nan)
    features["channel_pos_60"] = (out["close"] - rolling_low_60) / channel_width_60 - 0.5
    features["channel_pos_120"] = (out["close"] - rolling_low_120) / channel_width_120 - 0.5
    features["breakout_ratio_20"] = (out["close"] - rolling_low_20) / channel_width_20
    features["breakout_ratio_60"] = (out["close"] - rolling_low_60) / channel_width_60
    features["breakout_ratio_120"] = (out["close"] - rolling_low_120) / channel_width_120
    features["distance_to_low_20"] = (out["close"] - rolling_low_20) / rolling_low_20.replace(0, np.nan)
    features["distance_to_low_60"] = (out["close"] - rolling_low_60) / rolling_low_60.replace(0, np.nan)
    features["distance_to_low_120"] = (out["close"] - rolling_low_120) / rolling_low_120.replace(0, np.nan)
    features["distance_to_high_20"] = out["close"] / rolling_high_20.replace(0, np.nan) - 1.0
    features["distance_to_high_60"] = out["close"] / rolling_high_60.replace(0, np.nan) - 1.0
    features["distance_to_high_120"] = out["close"] / rolling_high_120.replace(0, np.nan) - 1.0
    features["price_zscore_20"] = (out["close"] - ma_20) / close_std_20.replace(0, np.nan)
    features["price_zscore_60"] = (out["close"] - ma_60) / close_std_60.replace(0, np.nan)
    features["price_zscore_120"] = (out["close"] - ma_120) / close_std_120.replace(0, np.nan)
    features["range_pct_1"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    features["vol_compress_5_20"] = -(features["vol_10"] / features["vol_20"].replace(0, np.nan) - 1.0).abs()
    features["vol_compress_10_60"] = -(features["vol_10"] / vol_60.replace(0, np.nan) - 1.0).abs()
    features["ret_reversal_5_20"] = features["ret_5"] - features["ret_20"]
    features["ret_reversal_10_60"] = features["ret_10"] - features["ret_60"]
    features["turnover_shock_1_20"] = out["turnover_rate"] / features["turnover_20"].replace(0, np.nan) - 1.0
    turnover_60 = g["turnover_rate"].rolling(60).mean().reset_index(level=0, drop=True)
    features["turnover_shock_5_60"] = features["turnover_5"] / turnover_60.replace(0, np.nan) - 1.0
    features["turnover_ratio_5_60"] = features["turnover_5"] / turnover_60.replace(0, np.nan) - 1.0
    features["turnover_ratio_20_60"] = features["turnover_20"] / turnover_60.replace(0, np.nan) - 1.0
    features["price_volume_div_5"] = features["ret_5"] - (amount_ma_5 / amount_ma_20.replace(0, np.nan) - 1.0)
    amount_ma_60 = g["amount"].rolling(60).mean().reset_index(level=0, drop=True)
    amount_std_20 = g["amount"].rolling(20).std().reset_index(level=0, drop=True)
    amount_std_60 = g["amount"].rolling(60).std().reset_index(level=0, drop=True)
    turnover_std_20 = g["turnover_rate"].rolling(20).std().reset_index(level=0, drop=True)
    turnover_std_60 = g["turnover_rate"].rolling(60).std().reset_index(level=0, drop=True)
    features["price_volume_div_20"] = features["ret_20"] - (amount_ma_20 / amount_ma_60.replace(0, np.nan) - 1.0)
    features["rebound_from_low_20"] = out["close"] / rolling_low_20.replace(0, np.nan) - 1.0 - features["ret_20"]
    features["amount_trend_5"] = amount_ma_5 / amount_ma_20.replace(0, np.nan) - 1.0
    features["amount_zscore_20"] = (out["amount"] - amount_ma_20) / amount_std_20.replace(0, np.nan)
    features["amount_zscore_60"] = (out["amount"] - amount_ma_60) / amount_std_60.replace(0, np.nan)
    features["turnover_zscore_20"] = (out["turnover_rate"] - features["turnover_20"]) / turnover_std_20.replace(0, np.nan)
    features["turnover_zscore_60"] = (out["turnover_rate"] - turnover_60) / turnover_std_60.replace(0, np.nan)

    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_14 = true_range.groupby(out["symbol"], observed=True).rolling(14).mean().reset_index(level=0, drop=True)
    atr_20 = true_range.groupby(out["symbol"], observed=True).rolling(20).mean().reset_index(level=0, drop=True)
    atr_60 = true_range.groupby(out["symbol"], observed=True).rolling(60).mean().reset_index(level=0, drop=True)
    features["atr_14"] = atr_14 / out["close"].replace(0, np.nan)
    features["atr_20"] = atr_20 / out["close"].replace(0, np.nan)
    features["atr_ratio_14_60"] = atr_14 / atr_60.replace(0, np.nan) - 1.0
    features["atr_ratio_20_60"] = atr_20 / atr_60.replace(0, np.nan) - 1.0

    abs_diff = g["close"].diff().abs()
    path_10 = abs_diff.groupby(out["symbol"], observed=True).rolling(10).sum().reset_index(level=0, drop=True)
    path_20 = abs_diff.groupby(out["symbol"], observed=True).rolling(20).sum().reset_index(level=0, drop=True)
    path_60 = abs_diff.groupby(out["symbol"], observed=True).rolling(60).sum().reset_index(level=0, drop=True)
    features["efficiency_ratio_10"] = (out["close"] - g["close"].shift(10)).abs() / path_10.replace(0, np.nan)
    features["efficiency_ratio_20"] = (out["close"] - g["close"].shift(20)).abs() / path_20.replace(0, np.nan)
    features["efficiency_ratio_60"] = (out["close"] - g["close"].shift(60)).abs() / path_60.replace(0, np.nan)

    features["float_mv_log"] = np.log(out["float_mv"].clip(lower=1.0))

    # Fundamental features are point-in-time fields already aligned in the panel.
    features["value_earnings_yield"] = 1.0 / out["pe_ttm"].replace(0, np.nan)
    features["value_book_to_price"] = 1.0 / out["pb"].replace(0, np.nan)
    features["value_sales_yield"] = 1.0 / out["ps_ttm"].replace(0, np.nan)
    features["value_dividend_yield"] = out["dv_ttm"]

    features["quality_roe"] = out["roe"]
    features["quality_roe_dt"] = out["roe_dt"]
    features["quality_roa"] = out["roa"]
    features["quality_roic"] = out["roic"]
    features["quality_grossprofit_margin"] = out["grossprofit_margin"]
    features["quality_netprofit_margin"] = out["netprofit_margin"]
    features["quality_ocfps"] = out["ocfps"]
    features["quality_cfps"] = out["cfps"]
    features["quality_profit_dedt_to_mv"] = out["profit_dedt"] / out["float_mv"].replace(0, np.nan)

    features["leverage_debt_to_assets"] = -out["debt_to_assets"]
    features["solvency_current_ratio"] = out["current_ratio"]
    features["solvency_quick_ratio"] = out["quick_ratio"]

    features["growth_basic_eps_yoy"] = out["basic_eps_yoy"]
    features["growth_dt_eps_yoy"] = out["dt_eps_yoy"]
    features["growth_netprofit_yoy"] = out["netprofit_yoy"]
    features["growth_dt_netprofit_yoy"] = out["dt_netprofit_yoy"]
    features["growth_ocf_yoy"] = out["ocf_yoy"]
    features["growth_revenue_yoy"] = out["tr_yoy"]
    features["growth_operating_revenue_yoy"] = out["or_yoy"]
    features["growth_q_sales_yoy"] = out["q_sales_yoy"]
    features["growth_q_op_qoq"] = out["q_op_qoq"]

    feature_frame = pd.DataFrame(features, index=out.index)
    feature_frame = feature_frame.astype("float32")
    return pd.concat([out, feature_frame], axis=1)
