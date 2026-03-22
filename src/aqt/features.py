from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

try:
    import ta_cn.talib as ta_talib
except Exception:  # pragma: no cover - optional acceleration dependency
    ta_talib = None

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


def infer_feature_dependencies(features: list[str] | None = None) -> list[str]:
    deps: set[str] = set()
    for feature in features or FEATURE_COLUMNS:
        deps.update(_infer_factor_dependencies(feature))
    return sorted(deps)


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


def add_features(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("symbol", group_keys=False, observed=True)
    selected_order = list(features or FEATURE_COLUMNS)
    selected = set(selected_order)
    if not selected_order:
        return out

    computed: dict[str, pd.Series] = {}
    ret_cache: dict[int, pd.Series] = {}
    ma_cache: dict[int, pd.Series] = {}
    close_std_cache: dict[int, pd.Series] = {}
    high_cache: dict[int, pd.Series] = {}
    low_cache: dict[int, pd.Series] = {}
    turnover_mean_cache: dict[int, pd.Series] = {}
    turnover_std_cache: dict[int, pd.Series] = {}
    amount_mean_cache: dict[int, pd.Series] = {}
    amount_std_cache: dict[int, pd.Series] = {}
    vol_cache: dict[int, pd.Series] = {}
    atr_cache: dict[int, pd.Series] = {}
    path_cache: dict[int, pd.Series] = {}
    shifted_close_cache: dict[int, pd.Series] = {}
    ta_close_wide_cache: pd.DataFrame | None = None
    ta_close_indexer: pd.MultiIndex | None = None
    ta_lib = None
    ta_ma_cache: dict[int, pd.Series] = {}
    ta_std_cache: dict[int, pd.Series] = {}

    daily_ret_cache: pd.Series | None = None
    prev_close_cache: pd.Series | None = None
    gap_1_cache: pd.Series | None = None
    intraday_ret_1_cache: pd.Series | None = None
    true_range_cache: pd.Series | None = None
    abs_diff_cache: pd.Series | None = None

    ma_feature_requested = any(
        feature.startswith(("close_to_ma_", "trend_", "ma_spread_", "price_zscore_"))
        for feature in selected_order
    )

    def ta_close_wide() -> pd.DataFrame | None:
        nonlocal ta_close_wide_cache, ta_close_indexer, ta_lib
        if ta_talib is None or not ma_feature_requested:
            return None
        if ta_close_wide_cache is None:
            ta_close_wide_cache = out.pivot(index="date", columns="symbol", values="close").sort_index(axis=0).sort_index(axis=1)
            ta_close_indexer = pd.MultiIndex.from_arrays([out["date"].to_numpy(), out["symbol"].to_numpy()])
            ta_lib = ta_talib.init(mode=2, skipna=False, to_globals=False)
        return ta_close_wide_cache

    def ta_series_from_matrix(values: np.ndarray, name: str) -> pd.Series:
        assert ta_close_wide_cache is not None
        assert ta_close_indexer is not None
        long = (
            pd.DataFrame(values, index=ta_close_wide_cache.index, columns=ta_close_wide_cache.columns)
            .stack(future_stack=True)
            .rename(name)
        )
        long.index = long.index.set_names(["date", "symbol"])
        return long.reindex(ta_close_indexer).reset_index(drop=True)

    def pct_return(window: int) -> pd.Series:
        if window not in ret_cache:
            ret_cache[window] = g["close"].pct_change(window)
        return ret_cache[window]

    def daily_ret() -> pd.Series:
        nonlocal daily_ret_cache
        if daily_ret_cache is None:
            daily_ret_cache = g["close"].pct_change()
        return daily_ret_cache

    def moving_average(window: int) -> pd.Series:
        if window not in ma_cache:
            wide = ta_close_wide()
            if wide is not None and ta_lib is not None:
                if window not in ta_ma_cache:
                    ta_ma_cache[window] = ta_series_from_matrix(
                        ta_lib.SMA(wide.to_numpy(dtype=float), timeperiod=window),
                        f"ma_{window}",
                    )
                ma_cache[window] = ta_ma_cache[window]
            else:
                ma_cache[window] = g["close"].rolling(window).mean().reset_index(level=0, drop=True)
        return ma_cache[window]

    def close_std(window: int) -> pd.Series:
        if window not in close_std_cache:
            wide = ta_close_wide()
            if wide is not None and ta_lib is not None:
                if window not in ta_std_cache:
                    sample_scale = np.sqrt(window / max(window - 1, 1))
                    ta_std_cache[window] = ta_series_from_matrix(
                        ta_lib.STDDEV(wide.to_numpy(dtype=float), timeperiod=window) * sample_scale,
                        f"std_{window}",
                    )
                close_std_cache[window] = ta_std_cache[window]
            else:
                close_std_cache[window] = g["close"].rolling(window).std().reset_index(level=0, drop=True)
        return close_std_cache[window]

    def rolling_high(window: int) -> pd.Series:
        if window not in high_cache:
            high_cache[window] = g["high"].rolling(window).max().reset_index(level=0, drop=True)
        return high_cache[window]

    def rolling_low(window: int) -> pd.Series:
        if window not in low_cache:
            low_cache[window] = g["low"].rolling(window).min().reset_index(level=0, drop=True)
        return low_cache[window]

    def turnover_mean(window: int) -> pd.Series:
        if window not in turnover_mean_cache:
            turnover_mean_cache[window] = g["turnover_rate"].rolling(window).mean().reset_index(level=0, drop=True)
        return turnover_mean_cache[window]

    def turnover_std(window: int) -> pd.Series:
        if window not in turnover_std_cache:
            turnover_std_cache[window] = g["turnover_rate"].rolling(window).std().reset_index(level=0, drop=True)
        return turnover_std_cache[window]

    def amount_mean(window: int) -> pd.Series:
        if window not in amount_mean_cache:
            amount_mean_cache[window] = g["amount"].rolling(window).mean().reset_index(level=0, drop=True)
        return amount_mean_cache[window]

    def amount_std(window: int) -> pd.Series:
        if window not in amount_std_cache:
            amount_std_cache[window] = g["amount"].rolling(window).std().reset_index(level=0, drop=True)
        return amount_std_cache[window]

    def volatility(window: int) -> pd.Series:
        if window not in vol_cache:
            vol_cache[window] = daily_ret().groupby(out["symbol"], observed=True).rolling(window).std().reset_index(level=0, drop=True)
        return vol_cache[window]

    def shifted_close(window: int) -> pd.Series:
        if window not in shifted_close_cache:
            shifted_close_cache[window] = g["close"].shift(window)
        return shifted_close_cache[window]

    def previous_close() -> pd.Series:
        nonlocal prev_close_cache
        if prev_close_cache is None:
            prev_close_cache = shifted_close(1)
        return prev_close_cache

    def gap_1() -> pd.Series:
        nonlocal gap_1_cache
        if gap_1_cache is None:
            gap_1_cache = out["open"] / previous_close().replace(0, np.nan) - 1.0
        return gap_1_cache

    def intraday_ret_1() -> pd.Series:
        nonlocal intraday_ret_1_cache
        if intraday_ret_1_cache is None:
            intraday_ret_1_cache = out["close"] / out["open"].replace(0, np.nan) - 1.0
        return intraday_ret_1_cache

    def true_range() -> pd.Series:
        nonlocal true_range_cache
        if true_range_cache is None:
            true_range_cache = pd.concat(
                [
                    out["high"] - out["low"],
                    (out["high"] - previous_close()).abs(),
                    (out["low"] - previous_close()).abs(),
                ],
                axis=1,
            ).max(axis=1)
        return true_range_cache

    def atr(window: int) -> pd.Series:
        if window not in atr_cache:
            atr_cache[window] = true_range().groupby(out["symbol"], observed=True).rolling(window).mean().reset_index(level=0, drop=True)
        return atr_cache[window]

    def abs_diff() -> pd.Series:
        nonlocal abs_diff_cache
        if abs_diff_cache is None:
            abs_diff_cache = g["close"].diff().abs()
        return abs_diff_cache

    def path_sum(window: int) -> pd.Series:
        if window not in path_cache:
            path_cache[window] = abs_diff().groupby(out["symbol"], observed=True).rolling(window).sum().reset_index(level=0, drop=True)
        return path_cache[window]

    def channel_width(window: int) -> pd.Series:
        return (rolling_high(window) - rolling_low(window)).replace(0, np.nan)

    fundamental_values = {
        "value_earnings_yield": lambda: 1.0 / out["pe_ttm"].replace(0, np.nan),
        "value_book_to_price": lambda: 1.0 / out["pb"].replace(0, np.nan),
        "value_sales_yield": lambda: 1.0 / out["ps_ttm"].replace(0, np.nan),
        "value_dividend_yield": lambda: out["dv_ttm"],
        "quality_roe": lambda: out["roe"],
        "quality_roe_dt": lambda: out["roe_dt"],
        "quality_roa": lambda: out["roa"],
        "quality_roic": lambda: out["roic"],
        "quality_grossprofit_margin": lambda: out["grossprofit_margin"],
        "quality_netprofit_margin": lambda: out["netprofit_margin"],
        "quality_ocfps": lambda: out["ocfps"],
        "quality_cfps": lambda: out["cfps"],
        "quality_profit_dedt_to_mv": lambda: out["profit_dedt"] / out["float_mv"].replace(0, np.nan),
        "leverage_debt_to_assets": lambda: -out["debt_to_assets"],
        "solvency_current_ratio": lambda: out["current_ratio"],
        "solvency_quick_ratio": lambda: out["quick_ratio"],
        "growth_basic_eps_yoy": lambda: out["basic_eps_yoy"],
        "growth_dt_eps_yoy": lambda: out["dt_eps_yoy"],
        "growth_netprofit_yoy": lambda: out["netprofit_yoy"],
        "growth_dt_netprofit_yoy": lambda: out["dt_netprofit_yoy"],
        "growth_ocf_yoy": lambda: out["ocf_yoy"],
        "growth_revenue_yoy": lambda: out["tr_yoy"],
        "growth_operating_revenue_yoy": lambda: out["or_yoy"],
        "growth_q_sales_yoy": lambda: out["q_sales_yoy"],
        "growth_q_op_qoq": lambda: out["q_op_qoq"],
    }

    for feature in selected_order:
        if feature in computed:
            continue
        if feature in fundamental_values:
            computed[feature] = fundamental_values[feature]()
            continue
        if match := re.fullmatch(r"ret_(\d+)", feature):
            computed[feature] = pct_return(int(match.group(1)))
            continue
        if match := re.fullmatch(r"close_to_ma_(\d+)", feature):
            window = int(match.group(1))
            computed[feature] = out["close"] / moving_average(window) - 1.0
            continue
        if match := re.fullmatch(r"trend_(\d+)_(\d+)", feature):
            fast, slow = (int(group) for group in match.groups())
            computed[feature] = moving_average(fast) / moving_average(slow).replace(0, np.nan) - 1.0
            continue
        if match := re.fullmatch(r"ma_spread_(\d+)_(\d+)_(\d+)", feature):
            fast, mid, slow = (int(group) for group in match.groups())
            computed[feature] = (moving_average(fast) - moving_average(mid)) / moving_average(slow).replace(0, np.nan)
            continue
        if match := re.fullmatch(r"vol_(\d+)", feature):
            computed[feature] = volatility(int(match.group(1)))
            continue
        if match := re.fullmatch(r"amp_(\d+)", feature):
            window = int(match.group(1))
            computed[feature] = rolling_high(window) / rolling_low(window).replace(0, np.nan) - 1.0
            continue
        if match := re.fullmatch(r"turnover_(\d+)", feature):
            computed[feature] = turnover_mean(int(match.group(1)))
            continue
        if match := re.fullmatch(r"ret_reversal_(\d+)_(\d+)", feature):
            fast, slow = (int(group) for group in match.groups())
            computed[feature] = pct_return(fast) - pct_return(slow)
            continue
        if match := re.fullmatch(r"vol_ratio_(\d+)_(\d+)", feature):
            fast, slow = (int(group) for group in match.groups())
            computed[feature] = volatility(fast) / volatility(slow).replace(0, np.nan) - 1.0
            continue
        if match := re.fullmatch(r"turnover_ratio_(\d+)_(\d+)", feature):
            fast, slow = (int(group) for group in match.groups())
            computed[feature] = turnover_mean(fast) / turnover_mean(slow).replace(0, np.nan) - 1.0
            continue
        if match := re.fullmatch(r"channel_pos_(\d+)", feature):
            window = int(match.group(1))
            computed[feature] = (out["close"] - rolling_low(window)) / channel_width(window) - 0.5
            continue
        if match := re.fullmatch(r"close_to_high_(\d+)", feature):
            window = int(match.group(1))
            computed[feature] = out["close"] / rolling_high(window).replace(0, np.nan) - 1.0
            continue
        if match := re.fullmatch(r"close_to_low_(\d+)", feature):
            window = int(match.group(1))
            computed[feature] = out["close"] / rolling_low(window).replace(0, np.nan) - 1.0
            continue
        if match := re.fullmatch(r"distance_to_high_(\d+)", feature):
            window = int(match.group(1))
            computed[feature] = out["close"] / rolling_high(window).replace(0, np.nan) - 1.0
            continue
        if match := re.fullmatch(r"distance_to_low_(\d+)", feature):
            window = int(match.group(1))
            computed[feature] = (out["close"] - rolling_low(window)) / rolling_low(window).replace(0, np.nan)
            continue
        if match := re.fullmatch(r"price_zscore_(\d+)", feature):
            window = int(match.group(1))
            computed[feature] = (out["close"] - moving_average(window)) / close_std(window).replace(0, np.nan)
            continue
        if match := re.fullmatch(r"efficiency_ratio_(\d+)", feature):
            window = int(match.group(1))
            computed[feature] = (out["close"] - shifted_close(window)).abs() / path_sum(window).replace(0, np.nan)
            continue
        if match := re.fullmatch(r"atr_(\d+)", feature):
            window = int(match.group(1))
            computed[feature] = atr(window) / out["close"].replace(0, np.nan)
            continue
        if match := re.fullmatch(r"atr_ratio_(\d+)_(\d+)", feature):
            fast, slow = (int(group) for group in match.groups())
            computed[feature] = atr(fast) / atr(slow).replace(0, np.nan) - 1.0
            continue
        if match := re.fullmatch(r"breakout_ratio_(\d+)", feature):
            window = int(match.group(1))
            computed[feature] = (out["close"] - rolling_low(window)) / channel_width(window)
            continue
        if match := re.fullmatch(r"amount_zscore_(\d+)", feature):
            window = int(match.group(1))
            computed[feature] = (out["amount"] - amount_mean(window)) / amount_std(window).replace(0, np.nan)
            continue
        if match := re.fullmatch(r"turnover_zscore_(\d+)", feature):
            window = int(match.group(1))
            computed[feature] = (out["turnover_rate"] - turnover_mean(window)) / turnover_std(window).replace(0, np.nan)
            continue

        if feature == "amount_ratio_5_20":
            computed[feature] = amount_mean(5) / amount_mean(20).replace(0, np.nan)
        elif feature == "amount_ratio_1_20":
            computed[feature] = out["amount"] / amount_mean(20).replace(0, np.nan) - 1.0
        elif feature == "gap_1":
            computed[feature] = gap_1()
        elif feature == "intraday_ret_1":
            computed[feature] = intraday_ret_1()
        elif feature == "overnight_ret_5":
            computed[feature] = gap_1().groupby(out["symbol"], observed=True).rolling(5).mean().reset_index(level=0, drop=True)
        elif feature == "range_pct_1":
            computed[feature] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
        elif feature == "vol_compress_5_20":
            computed[feature] = -(volatility(10) / volatility(20).replace(0, np.nan) - 1.0).abs()
        elif feature == "vol_compress_10_60":
            computed[feature] = -(volatility(10) / volatility(60).replace(0, np.nan) - 1.0).abs()
        elif feature == "turnover_shock_1_20":
            computed[feature] = out["turnover_rate"] / turnover_mean(20).replace(0, np.nan) - 1.0
        elif feature == "turnover_shock_5_60":
            computed[feature] = turnover_mean(5) / turnover_mean(60).replace(0, np.nan) - 1.0
        elif feature == "price_volume_div_5":
            computed[feature] = pct_return(5) - (amount_mean(5) / amount_mean(20).replace(0, np.nan) - 1.0)
        elif feature == "price_volume_div_20":
            computed[feature] = pct_return(20) - (amount_mean(20) / amount_mean(60).replace(0, np.nan) - 1.0)
        elif feature == "rebound_from_low_20":
            computed[feature] = out["close"] / rolling_low(20).replace(0, np.nan) - 1.0 - pct_return(20)
        elif feature == "amount_trend_5":
            computed[feature] = amount_mean(5) / amount_mean(20).replace(0, np.nan) - 1.0
        elif feature == "float_mv_log":
            computed[feature] = np.log(out["float_mv"].clip(lower=1.0))

    feature_frame = pd.DataFrame({name: computed[name] for name in selected_order if name in computed}, index=out.index)
    feature_frame = feature_frame.astype("float32")
    return pd.concat([out, feature_frame], axis=1)
