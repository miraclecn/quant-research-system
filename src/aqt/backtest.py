from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aqt.config import PortfolioConfig


def select_rebalance_dates(dates: pd.Series, weekday: int) -> pd.Series:
    calendar = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates).drop_duplicates())})
    calendar = calendar.sort_values("date").reset_index(drop=True)
    iso = calendar["date"].dt.isocalendar()
    calendar["year"] = iso.year
    calendar["week"] = iso.week
    calendar["is_target_weekday"] = calendar["date"].dt.weekday.eq(weekday)
    calendar = calendar.sort_values(["year", "week", "is_target_weekday", "date"])
    chosen = calendar.groupby(["year", "week"], as_index=False).tail(1)
    return chosen["date"].sort_values().reset_index(drop=True)


def build_positions(pred_df: pd.DataFrame, cfg: PortfolioConfig) -> pd.DataFrame:
    rebalance_dates = select_rebalance_dates(pred_df["date"], cfg.rebalance_weekday)
    ranked = pred_df.loc[
        pred_df["date"].isin(rebalance_dates) & pred_df["tradable_universe"]
    ].copy()
    ranked["rank"] = ranked.groupby("date")["score"].rank(method="first", ascending=False)
    positions = ranked.loc[ranked["rank"] <= cfg.top_n].copy()
    actual_counts = positions.groupby("date")["symbol"].transform("count").clip(lower=1)
    positions["target_weight"] = 1.0 / actual_counts
    positions["target_weight"] = positions["target_weight"].clip(upper=cfg.max_weight)
    return positions[["date", "symbol", "target_weight"]]


def run_backtest(panel: pd.DataFrame, positions: pd.DataFrame, cfg: PortfolioConfig) -> tuple[pd.DataFrame, dict]:
    panel = panel.sort_values(["date", "symbol"]).copy()
    merged = positions.copy()

    next_dates = sorted(panel["date"].drop_duplicates())
    next_date_map = {next_dates[i]: next_dates[i + 1] for i in range(len(next_dates) - 1)}
    merged["trade_date"] = merged["date"].map(next_date_map)

    trade_open = panel[["date", "symbol", "open"]].rename(columns={"date": "trade_date", "open": "entry_open"})
    merged = merged.merge(trade_open, on=["trade_date", "symbol"], how="left")

    exit_reb = positions[["date"]].drop_duplicates().sort_values("date")
    exit_reb["exit_trade_date"] = exit_reb["date"].shift(-1).map(next_date_map)
    merged = merged.merge(exit_reb, on="date", how="left")
    exit_open = panel[["date", "symbol", "open"]].rename(columns={"date": "exit_trade_date", "open": "exit_open"})
    merged = merged.merge(exit_open, on=["exit_trade_date", "symbol"], how="left")

    tradable = (~panel["is_paused"]) & (~panel["is_limit_up"])
    tradable_df = panel.loc[tradable, ["date", "symbol"]].copy()
    tradable_df["can_enter"] = True
    merged = merged.merge(
        tradable_df.rename(columns={"date": "trade_date"}),
        on=["trade_date", "symbol"],
        how="left",
    )
    merged["can_enter"] = merged["can_enter"].eq(True)

    merged = merged.loc[merged["can_enter"] & merged["entry_open"].notna() & merged["exit_open"].notna()].copy()
    gross_ret = merged["exit_open"] / merged["entry_open"] - 1.0
    cost = 2.0 * (cfg.fee_rate + cfg.slippage_rate)
    merged["net_ret"] = gross_ret - cost
    merged["weighted_ret"] = merged["target_weight"] * merged["net_ret"]

    portfolio = merged.groupby("trade_date", as_index=False)["weighted_ret"].sum()
    portfolio = portfolio.rename(columns={"trade_date": "date", "weighted_ret": "portfolio_ret"})
    portfolio["equity"] = (1.0 + portfolio["portfolio_ret"]).cumprod()

    metrics = summarize_metrics(portfolio)
    return portfolio, metrics


def summarize_metrics(portfolio: pd.DataFrame) -> dict:
    if portfolio.empty:
        return {"status": "empty"}

    ret = portfolio["portfolio_ret"].fillna(0.0)
    dates = pd.to_datetime(portfolio["date"], errors="coerce").sort_values()
    gaps = dates.diff().dt.days.dropna()
    if gaps.empty:
        periods_per_year = 52.0
    else:
        median_gap_days = float(gaps.median())
        periods_per_year = 365.25 / median_gap_days if median_gap_days > 0 else 52.0
    ann_ret = (1.0 + ret.mean()) ** periods_per_year - 1.0
    ann_vol = ret.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    equity = portfolio["equity"]
    drawdown = equity / equity.cummax() - 1.0

    return {
        "status": "ok",
        "periods": int(len(portfolio)),
        "periods_per_year_est": float(periods_per_year),
        "annual_return_est": float(ann_ret),
        "annual_vol_est": float(ann_vol),
        "sharpe_est": float(sharpe),
        "max_drawdown": float(drawdown.min()),
    }


def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
