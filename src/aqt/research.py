from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from aqt.backtest import run_backtest, select_rebalance_dates, summarize_metrics
from aqt.config import AppConfig, PortfolioConfig
from aqt.data import default_benchmark_output_path


def summarize_feature_importance(
    feature_names: list[str],
    ridge_importances: list[np.ndarray],
    lgbm_importances: list[np.ndarray],
) -> pd.DataFrame:
    ridge_mean = np.mean(np.vstack(ridge_importances), axis=0) if ridge_importances else np.zeros(len(feature_names))
    lgbm_mean = np.mean(np.vstack(lgbm_importances), axis=0) if lgbm_importances else np.zeros(len(feature_names))

    out = pd.DataFrame(
        {
            "feature": feature_names,
            "ridge_abs_coef_mean": np.abs(ridge_mean),
            "ridge_coef_mean": ridge_mean,
            "lgbm_gain_proxy_mean": lgbm_mean,
        }
    )
    return out.sort_values(["lgbm_gain_proxy_mean", "ridge_abs_coef_mean"], ascending=False).reset_index(drop=True)


def summarize_ridge_coefficients(
    selected_features: pd.DataFrame,
    ridge_coef: np.ndarray,
) -> pd.DataFrame:
    if selected_features.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "direction",
                "sign_flipped",
                "train_mean_rank_ic",
                "train_rank_ic_ir",
                "train_positive_rank_ic_ratio",
                "train_mean_top_bottom_spread",
                "ridge_oriented_coef",
                "ridge_oriented_abs_coef",
                "ridge_original_coef",
                "ridge_original_abs_coef",
                "ridge_oriented_positive",
            ]
        )

    out = selected_features.copy().reset_index(drop=True)
    coef = np.asarray(ridge_coef, dtype=float)
    out["ridge_oriented_coef"] = coef
    out["ridge_oriented_abs_coef"] = np.abs(coef)
    out["ridge_original_coef"] = coef * out["direction"].astype(float)
    out["ridge_original_abs_coef"] = np.abs(out["ridge_original_coef"])
    out["ridge_oriented_positive"] = out["ridge_oriented_coef"] > 0
    return out.sort_values("ridge_oriented_abs_coef", ascending=False).reset_index(drop=True)


def summarize_selected_feature_frequency(output_dir: Path, split_ids: list[int]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for split_id in split_ids:
        path = output_dir / f"split_{split_id:02d}" / "selected_features.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["split_id"] = split_id
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    split_count = int(combined["split_id"].nunique())
    summary = (
        combined.groupby("feature", observed=False)
        .agg(
            selection_count=("split_id", "nunique"),
            direction_mean=("direction", "mean"),
            train_mean_rank_ic_mean=("train_mean_rank_ic", "mean"),
            train_mean_rank_ic_abs_mean=("train_mean_rank_ic", lambda s: np.abs(s).mean()),
            train_rank_ic_ir_mean=("train_rank_ic_ir", "mean"),
            train_rank_ic_ir_abs_mean=("train_rank_ic_ir", lambda s: np.abs(s).mean()),
            train_positive_rank_ic_ratio_mean=("train_positive_rank_ic_ratio", "mean"),
            train_mean_top_bottom_spread_mean=("train_mean_top_bottom_spread", "mean"),
        )
        .reset_index()
    )
    summary["selection_rate"] = summary["selection_count"] / split_count
    return summary.sort_values(
        ["selection_count", "train_mean_rank_ic_abs_mean", "train_rank_ic_ir_abs_mean"],
        ascending=False,
    ).reset_index(drop=True)


def summarize_lgbm_selected_feature_frequency(output_dir: Path, split_ids: list[int]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for split_id in split_ids:
        path = output_dir / f"split_{split_id:02d}" / "lgbm_selected_features.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["split_id"] = split_id
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    split_count = int(combined["split_id"].nunique())
    summary = (
        combined.groupby("feature", observed=False)
        .agg(
            lgbm_selection_count=("split_id", "nunique"),
            lgbm_selection_score_mean=("ridge_split_score", "mean"),
            lgbm_selection_quality_score_mean=("quality_score", "mean"),
            ridge_oriented_abs_coef_mean=("ridge_oriented_abs_coef", "mean"),
            ridge_oriented_positive_ratio=("ridge_oriented_positive", "mean"),
        )
        .reset_index()
    )
    summary["lgbm_selection_rate"] = summary["lgbm_selection_count"] / split_count
    return summary.sort_values(
        ["lgbm_selection_count", "lgbm_selection_score_mean", "ridge_oriented_abs_coef_mean"],
        ascending=False,
    ).reset_index(drop=True)


def summarize_ridge_coefficient_stability(output_dir: Path, split_ids: list[int]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for split_id in split_ids:
        path = output_dir / f"split_{split_id:02d}" / "ridge_coefficients.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["split_id"] = split_id
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    split_count = int(combined["split_id"].nunique())
    summary = (
        combined.groupby("feature", observed=False)
        .agg(
            selection_count=("split_id", "nunique"),
            direction_mean=("direction", "mean"),
            train_mean_rank_ic_mean=("train_mean_rank_ic", "mean"),
            ridge_oriented_coef_mean=("ridge_oriented_coef", "mean"),
            ridge_oriented_coef_std=("ridge_oriented_coef", "std"),
            ridge_oriented_abs_coef_mean=("ridge_oriented_abs_coef", "mean"),
            ridge_oriented_positive_ratio=("ridge_oriented_positive", "mean"),
            ridge_original_coef_mean=("ridge_original_coef", "mean"),
            ridge_original_coef_std=("ridge_original_coef", "std"),
            ridge_original_abs_coef_mean=("ridge_original_abs_coef", "mean"),
        )
        .reset_index()
    )
    summary["selection_rate"] = summary["selection_count"] / split_count
    summary["ridge_oriented_coef_cv"] = summary["ridge_oriented_coef_std"] / summary["ridge_oriented_abs_coef_mean"].replace(0, np.nan)
    summary["ridge_original_coef_cv"] = summary["ridge_original_coef_std"] / summary["ridge_original_abs_coef_mean"].replace(0, np.nan)
    return summary.sort_values(
        ["selection_count", "ridge_oriented_abs_coef_mean", "ridge_oriented_positive_ratio"],
        ascending=False,
    ).reset_index(drop=True)


def compute_signal_diagnostics(
    scored: pd.DataFrame,
    score_col: str,
    target_col: str,
    bucket_count: int = 5,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    ic_df, bucket_df, _, labels = _compute_daily_bucket_stats(scored, score_col, target_col, bucket_count)
    bucket_df = bucket_df.rename(columns={"avg_target": "avg_future_return"}) if not bucket_df.empty else pd.DataFrame(columns=["bucket", "avg_future_return", "date"])

    summary = {
        "coverage_dates": int(len(ic_df)),
        "mean_pearson_ic": float(ic_df["pearson_ic"].mean()) if not ic_df.empty else np.nan,
        "mean_rank_ic": float(ic_df["rank_ic"].mean()) if not ic_df.empty else np.nan,
        "ic_ir": float(ic_df["pearson_ic"].mean() / ic_df["pearson_ic"].std(ddof=0)) if not ic_df.empty and ic_df["pearson_ic"].std(ddof=0) > 0 else 0.0,
        "rank_ic_ir": float(ic_df["rank_ic"].mean() / ic_df["rank_ic"].std(ddof=0)) if not ic_df.empty and ic_df["rank_ic"].std(ddof=0) > 0 else 0.0,
        "mean_top_bottom_spread": float(ic_df["top_bottom_spread"].mean()) if not ic_df.empty else np.nan,
        "positive_rank_ic_ratio": float((ic_df["rank_ic"] > 0).mean()) if not ic_df.empty else np.nan,
    }
    return ic_df, summary, bucket_df


def winsorize_and_zscore_by_date(
    df: pd.DataFrame,
    value_col: str,
    output_col: str,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    out = df.copy()
    eligible = out["tradable_universe"] & out[value_col].notna()
    out[output_col] = np.nan
    if not eligible.any():
        return out

    work = out.loc[eligible, ["date", value_col]].copy()
    work[value_col] = work[value_col].astype(float)

    quantiles = (
        work.groupby("date", sort=False, observed=False)[value_col]
        .quantile([lower_quantile, upper_quantile])
        .unstack(level=-1)
        .rename(columns={lower_quantile: "_lo", upper_quantile: "_hi"})
    )
    work = work.join(quantiles, on="date")
    work["_clipped"] = work[value_col].clip(lower=work["_lo"], upper=work["_hi"])

    clipped_stats = (
        work.groupby("date", sort=False, observed=False)["_clipped"]
        .agg(_mean="mean", _std=lambda s: s.std(ddof=0))
    )
    work = work.join(clipped_stats, on="date")

    standardized = np.where(
        work["_std"].fillna(0.0).to_numpy() > 0,
        (work["_clipped"].to_numpy() - work["_mean"].to_numpy()) / work["_std"].to_numpy(),
        0.0,
    )
    out.loc[work.index, output_col] = standardized.astype("float32")
    return out


def compute_single_factor_group_backtest(
    scored: pd.DataFrame,
    score_col: str,
    target_col: str,
    bucket_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    curve_rows: list[dict] = []
    _, _, returns_df, labels = _compute_daily_bucket_stats(scored, score_col, target_col, bucket_count)
    top_label = labels[-1]
    bottom_label = labels[0]
    if returns_df.empty:
        return returns_df, pd.DataFrame(), {"status": "empty"}

    equity_cols = labels + ["long_short"]
    curve = returns_df[["date"]].copy()
    for col in equity_cols:
        curve[f"{col}_equity"] = (1.0 + returns_df[col].fillna(0.0)).cumprod()
        portfolio = pd.DataFrame(
            {
                "date": returns_df["date"],
                "portfolio_ret": returns_df[col].fillna(0.0),
                "equity": curve[f"{col}_equity"],
            }
        )
        metrics = summarize_metrics(portfolio)
        curve_rows.append(
            {
                "bucket": col,
                "annual_return_est": metrics.get("annual_return_est"),
                "sharpe_est": metrics.get("sharpe_est"),
                "max_drawdown": metrics.get("max_drawdown"),
            }
        )

    bucket_means = returns_df[labels].mean()
    bucket_index = np.arange(1, len(labels) + 1, dtype=float)
    monotonic_spearman = pd.Series(bucket_index).corr(bucket_means.reset_index(drop=True), method="spearman")
    diffs = np.diff(bucket_means.to_numpy(dtype=float))

    summary = {
        "bucket_count": bucket_count,
        "coverage_dates": int(len(returns_df)),
        "top_bucket": top_label,
        "bottom_bucket": bottom_label,
        "long_short_annual_return_est": next((row["annual_return_est"] for row in curve_rows if row["bucket"] == "long_short"), np.nan),
        "long_short_sharpe_est": next((row["sharpe_est"] for row in curve_rows if row["bucket"] == "long_short"), np.nan),
        "top_bucket_annual_return_est": next((row["annual_return_est"] for row in curve_rows if row["bucket"] == top_label), np.nan),
        "bottom_bucket_annual_return_est": next((row["annual_return_est"] for row in curve_rows if row["bucket"] == bottom_label), np.nan),
        "bucket_return_spearman": float(monotonic_spearman) if pd.notna(monotonic_spearman) else np.nan,
        "bucket_return_monotonic_increasing": bool(np.all(diffs >= 0)) if len(diffs) else False,
        "bucket_return_monotonic_decreasing": bool(np.all(diffs <= 0)) if len(diffs) else False,
    }
    return returns_df, pd.DataFrame(curve_rows), summary


def _compute_daily_bucket_stats(
    scored: pd.DataFrame,
    score_col: str,
    target_col: str,
    bucket_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    ic_rows: list[dict] = []
    bucket_rows: list[pd.DataFrame] = []
    return_rows: list[dict] = []

    base = scored.loc[scored["tradable_universe"]].copy()
    base = base[["date", "symbol", score_col, target_col]].dropna(subset=[score_col, target_col])
    labels = [f"Q{i}" for i in range(1, bucket_count + 1)]
    top_label = labels[-1]
    bottom_label = labels[0]

    for date, day_df in base.groupby("date", sort=True):
        if len(day_df) < bucket_count:
            continue

        pearson_ic = day_df[score_col].corr(day_df[target_col], method="pearson")
        rank_ic = day_df[score_col].corr(day_df[target_col], method="spearman")

        ranked = day_df.copy()
        ranked["bucket"] = pd.qcut(
            ranked[score_col].rank(method="first"),
            q=bucket_count,
            labels=labels,
        )
        grouped = ranked.groupby("bucket", observed=False)[target_col].mean().reindex(labels)
        bucket_mean = grouped.rename("avg_target").reset_index()
        bucket_mean["date"] = date
        bucket_rows.append(bucket_mean)

        top_bucket = grouped.loc[top_label]
        bottom_bucket = grouped.loc[bottom_label]
        ic_rows.append(
            {
                "date": date,
                "n": int(len(day_df)),
                "pearson_ic": float(pearson_ic) if pd.notna(pearson_ic) else np.nan,
                "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
                "top_bottom_spread": float(top_bucket - bottom_bucket) if pd.notna(top_bucket) and pd.notna(bottom_bucket) else np.nan,
            }
        )

        row = {"date": date}
        for label in labels:
            value = grouped.loc[label]
            row[label] = float(value) if pd.notna(value) else np.nan
        row["long_short"] = row[top_label] - row[bottom_label] if pd.notna(row[top_label]) and pd.notna(row[bottom_label]) else np.nan
        return_rows.append(row)

    ic_df = pd.DataFrame(ic_rows)
    bucket_df = pd.concat(bucket_rows, ignore_index=True) if bucket_rows else pd.DataFrame(columns=["bucket", "avg_target", "date"])
    returns_df = pd.DataFrame(return_rows)
    return ic_df, bucket_df, returns_df, labels


def summarize_feature_diagnostics(
    scored: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
) -> pd.DataFrame:
    keep_cols = ["date", "symbol", "tradable_universe", target_col, *feature_names]
    base = scored.loc[:, [col for col in keep_cols if col in scored.columns]].copy()
    feature_names = [feature for feature in feature_names if feature in base.columns]
    if not feature_names:
        return pd.DataFrame()

    daily_rows: list[pd.DataFrame] = []
    for date, day_df in (
        base.loc[base["tradable_universe"], ["date", target_col, *feature_names]]
        .dropna(subset=[target_col])
        .groupby("date", sort=True)
    ):
        feature_frame = day_df[feature_names].dropna(axis=1, how="all")
        if len(day_df) < 2 or feature_frame.empty:
            continue

        target = day_df[target_col].astype(float)
        pearson_ic = feature_frame.corrwith(target, method="pearson")
        rank_ic = feature_frame.rank(method="average").corrwith(target.rank(method="average"), method="pearson")

        spread_series = pd.Series(np.nan, index=feature_frame.columns, dtype=float)
        if len(day_df) >= 5:
            valid_counts = feature_frame.notna().sum()
            spread_features = valid_counts[valid_counts >= 5].index.tolist()
            if spread_features:
                spread_frame = feature_frame.loc[:, spread_features]
                ranks = spread_frame.rank(method="first")
                rank_values = ranks.to_numpy(dtype=float)
                value_mask = spread_frame.notna().to_numpy()
                count_values = valid_counts.loc[spread_features].to_numpy(dtype=float)

                bucket_ids = np.full(rank_values.shape, np.nan, dtype=float)
                for sample_count in np.unique(count_values.astype(int)):
                    col_mask = count_values.astype(int) == sample_count
                    if sample_count < 5:
                        continue
                    rank_to_bucket = (
                        pd.qcut(pd.Series(np.arange(1, sample_count + 1)), q=5, labels=False).to_numpy(dtype=float) + 1.0
                    )
                    col_ranks = rank_values[:, col_mask]
                    col_valid_mask = value_mask[:, col_mask]
                    col_bucket_ids = np.full(col_ranks.shape, np.nan, dtype=float)
                    integer_ranks = col_ranks[col_valid_mask].astype(int) - 1
                    col_bucket_ids[col_valid_mask] = rank_to_bucket[integer_ranks]
                    bucket_ids[:, col_mask] = col_bucket_ids

                target_values = target.to_numpy(dtype=float)[:, None]
                top_mask = bucket_ids == 5.0
                bottom_mask = bucket_ids == 1.0
                top_counts = top_mask.sum(axis=0)
                bottom_counts = bottom_mask.sum(axis=0)
                top_means = np.divide(
                    (top_mask * target_values).sum(axis=0),
                    top_counts,
                    out=np.full(len(spread_features), np.nan, dtype=float),
                    where=top_counts > 0,
                )
                bottom_means = np.divide(
                    (bottom_mask * target_values).sum(axis=0),
                    bottom_counts,
                    out=np.full(len(spread_features), np.nan, dtype=float),
                    where=bottom_counts > 0,
                )
                spread_series.loc[spread_features] = top_means - bottom_means

        day_stats = pd.DataFrame(
            {
                "feature": feature_frame.columns,
                "date": date,
                "pearson_ic": pearson_ic.reindex(feature_frame.columns).to_numpy(),
                "rank_ic": rank_ic.reindex(feature_frame.columns).to_numpy(),
                "top_bottom_spread": spread_series.reindex(feature_frame.columns).to_numpy(),
            }
        )
        daily_rows.append(day_stats)

    if not daily_rows:
        return pd.DataFrame()

    daily_df = pd.concat(daily_rows, ignore_index=True)
    grouped = daily_df.groupby("feature", observed=False)
    out = grouped.agg(
        coverage_dates=("date", "nunique"),
        mean_pearson_ic=("pearson_ic", "mean"),
        mean_rank_ic=("rank_ic", "mean"),
        pearson_ic_std=("pearson_ic", lambda s: s.std(ddof=0)),
        rank_ic_std=("rank_ic", lambda s: s.std(ddof=0)),
        positive_rank_ic_ratio=("rank_ic", lambda s: float((s > 0).mean())),
        mean_top_bottom_spread=("top_bottom_spread", "mean"),
    ).reset_index()
    out["ic_ir"] = out["mean_pearson_ic"] / out["pearson_ic_std"].replace(0, np.nan)
    out["rank_ic_ir"] = out["mean_rank_ic"] / out["rank_ic_std"].replace(0, np.nan)
    out["ic_ir"] = out["ic_ir"].fillna(0.0)
    out["rank_ic_ir"] = out["rank_ic_ir"].fillna(0.0)
    out = out.drop(columns=["pearson_ic_std", "rank_ic_std"])
    if out.empty:
        return out
    return out.sort_values(
        ["mean_rank_ic", "rank_ic_ir", "mean_top_bottom_spread"],
        ascending=False,
    ).reset_index(drop=True)


def build_universe_benchmark_positions(scored: pd.DataFrame, cfg: PortfolioConfig) -> pd.DataFrame:
    rebalance_dates = select_rebalance_dates(scored["date"], cfg.rebalance_weekday)
    eligible = scored.loc[
        scored["date"].isin(rebalance_dates) & scored["tradable_universe"]
    ][["date", "symbol"]].copy()
    counts = eligible.groupby("date")["symbol"].transform("count").clip(lower=1)
    eligible["target_weight"] = 1.0 / counts
    return eligible


def compare_to_benchmark(strategy_portfolio: pd.DataFrame, benchmark_portfolio: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    merged = strategy_portfolio.merge(
        benchmark_portfolio[["date", "portfolio_ret"]].rename(columns={"portfolio_ret": "benchmark_ret"}),
        on="date",
        how="inner",
    )
    merged["excess_ret"] = merged["portfolio_ret"] - merged["benchmark_ret"]
    merged["excess_equity"] = (1.0 + merged["excess_ret"]).cumprod()

    excess_portfolio = merged[["date", "excess_ret", "excess_equity"]].rename(
        columns={"excess_ret": "portfolio_ret", "excess_equity": "equity"}
    )
    excess_metrics = summarize_metrics(excess_portfolio)
    excess_metrics["benchmark_periods"] = int(len(merged))
    return merged, excess_metrics


def load_official_index_benchmark(
    cfg: AppConfig,
    benchmark_dates: pd.Series,
) -> pd.DataFrame:
    benchmark_dates = pd.to_datetime(benchmark_dates)
    benchmark_path = default_benchmark_output_path(cfg.data.index_code)
    if benchmark_path.exists():
        benchmark = pd.read_parquet(benchmark_path, columns=["date", "benchmark_ret"])
        benchmark["date"] = pd.to_datetime(benchmark["date"])
        benchmark = benchmark.loc[benchmark["date"].isin(benchmark_dates), ["date", "benchmark_ret"]].copy()
        benchmark["benchmark_equity"] = (1.0 + benchmark["benchmark_ret"].fillna(0.0)).cumprod()
        return benchmark.reset_index(drop=True)

    input_path = cfg.data.input_path if cfg.data.input_path.suffix == ".duckdb" else Path("stock_data.duckdb")
    if not input_path.exists():
        return pd.DataFrame(columns=["date", "benchmark_ret", "benchmark_equity"])

    with duckdb.connect(str(input_path), read_only=True) as con:
        exists = bool(
            con.execute(
                "SELECT COUNT(*) > 0 FROM duckdb_tables() WHERE schema_name = 'main' AND table_name = 'index_daily'"
            ).fetchone()[0]
        )
        if not exists:
            return pd.DataFrame(columns=["date", "benchmark_ret", "benchmark_equity"])

        df = con.execute(
            """
SELECT
    strptime(trade_date, '%Y%m%d') AS date,
    close,
    pre_close
FROM index_daily
WHERE ts_code = ?
ORDER BY trade_date
""",
            [cfg.data.index_code],
        ).df()

    if df.empty:
        return pd.DataFrame(columns=["date", "benchmark_ret", "benchmark_equity"])

    df["benchmark_ret"] = df["close"] / df["pre_close"].replace(0, np.nan) - 1.0
    benchmark = df.loc[df["date"].isin(benchmark_dates), ["date", "benchmark_ret"]].copy()
    benchmark["benchmark_equity"] = (1.0 + benchmark["benchmark_ret"].fillna(0.0)).cumprod()
    return benchmark.reset_index(drop=True)


def compare_to_official_index(
    strategy_portfolio: pd.DataFrame,
    official_benchmark: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    if official_benchmark.empty:
        return pd.DataFrame(columns=["date", "portfolio_ret", "benchmark_ret", "excess_ret", "excess_equity"]), {"status": "empty"}

    merged = strategy_portfolio.merge(
        official_benchmark[["date", "benchmark_ret"]],
        on="date",
        how="inner",
    )
    merged["excess_ret"] = merged["portfolio_ret"] - merged["benchmark_ret"]
    merged["excess_equity"] = (1.0 + merged["excess_ret"]).cumprod()
    excess_portfolio = merged[["date", "excess_ret", "excess_equity"]].rename(
        columns={"excess_ret": "portfolio_ret", "excess_equity": "equity"}
    )
    metrics = summarize_metrics(excess_portfolio)
    metrics["benchmark_periods"] = int(len(merged))
    return merged, metrics


def summarize_official_index_benchmark(official_benchmark: pd.DataFrame) -> dict:
    if official_benchmark.empty:
        return {"status": "empty"}
    benchmark_portfolio = official_benchmark[["date", "benchmark_ret", "benchmark_equity"]].rename(
        columns={"benchmark_ret": "portfolio_ret", "benchmark_equity": "equity"}
    )
    return summarize_metrics(benchmark_portfolio)


def compute_exposure_diagnostics(
    scored: pd.DataFrame,
    positions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    base_cols = ["date", "symbol", "tradable_universe", "industry", "float_mv"]
    base = scored.loc[:, [col for col in base_cols if col in scored.columns]].copy()
    if "industry" in base.columns:
        base["industry"] = base["industry"].astype("string").fillna("UNKNOWN")
    else:
        base["industry"] = "UNKNOWN"
    base["float_mv"] = pd.to_numeric(base.get("float_mv", 0.0), errors="coerce").fillna(0.0)
    base["float_mv_clipped"] = base["float_mv"].clip(lower=1.0)
    base["float_mv_log"] = np.log(base["float_mv_clipped"])

    pos = positions.merge(
        base[["date", "symbol", "industry", "float_mv", "float_mv_log"]],
        on=["date", "symbol"],
        how="left",
    )
    pos["industry"] = pos["industry"].fillna("UNKNOWN")
    pos["float_mv"] = pos["float_mv"].fillna(0.0)
    pos["float_mv_log"] = pos["float_mv_log"].fillna(0.0)

    universe = base.loc[base["tradable_universe"]].copy()
    universe_counts = universe.groupby("date")["symbol"].transform("count").clip(lower=1)
    universe["universe_weight"] = 1.0 / universe_counts

    strategy_industry = (
        pos.groupby(["date", "industry"], observed=False)["target_weight"]
        .sum()
        .rename("strategy_weight")
        .reset_index()
    )
    universe_industry = (
        universe.groupby(["date", "industry"], observed=False)["universe_weight"]
        .sum()
        .rename("universe_weight")
        .reset_index()
    )
    industry_exposure = strategy_industry.merge(
        universe_industry,
        on=["date", "industry"],
        how="outer",
    ).fillna({"strategy_weight": 0.0, "universe_weight": 0.0})
    industry_exposure["active_weight"] = industry_exposure["strategy_weight"] - industry_exposure["universe_weight"]
    industry_exposure = industry_exposure.sort_values(["date", "active_weight"], ascending=[True, False]).reset_index(drop=True)

    industry_summary_df = (
        industry_exposure.groupby("industry", observed=False)
        .agg(
            mean_active_weight=("active_weight", "mean"),
            mean_abs_active_weight=("active_weight", lambda s: np.abs(s).mean()),
            max_overweight=("active_weight", "max"),
            max_underweight=("active_weight", "min"),
            coverage_dates=("date", "nunique"),
        )
        .reset_index()
        .sort_values("mean_abs_active_weight", ascending=False)
        .reset_index(drop=True)
    )
    industry_summary = {
        "top_overweights": industry_summary_df.nlargest(10, "mean_active_weight").to_dict(orient="records"),
        "top_underweights": industry_summary_df.nsmallest(10, "mean_active_weight").to_dict(orient="records"),
        "top_abs_bias": industry_summary_df.head(10).to_dict(orient="records"),
    }

    strategy_size = _summarize_size_exposure(
        pos,
        weight_col="target_weight",
        avg_float_mv_col="strategy_avg_float_mv",
        avg_log_float_mv_col="strategy_avg_log_float_mv",
        median_float_mv_col="strategy_median_float_mv",
    )
    universe_size = _summarize_size_exposure(
        universe,
        weight_col="universe_weight",
        avg_float_mv_col="universe_avg_float_mv",
        avg_log_float_mv_col="universe_avg_log_float_mv",
        median_float_mv_col="universe_median_float_mv",
    )
    size_exposure = strategy_size.merge(universe_size, on="date", how="inner")
    size_exposure["active_avg_float_mv"] = size_exposure["strategy_avg_float_mv"] - size_exposure["universe_avg_float_mv"]
    size_exposure["active_avg_log_float_mv"] = size_exposure["strategy_avg_log_float_mv"] - size_exposure["universe_avg_log_float_mv"]
    size_exposure["active_median_float_mv"] = size_exposure["strategy_median_float_mv"] - size_exposure["universe_median_float_mv"]

    size_summary = {
        "coverage_dates": int(len(size_exposure)),
        "mean_active_avg_float_mv": float(size_exposure["active_avg_float_mv"].mean()) if not size_exposure.empty else np.nan,
        "mean_active_avg_log_float_mv": float(size_exposure["active_avg_log_float_mv"].mean()) if not size_exposure.empty else np.nan,
        "mean_active_median_float_mv": float(size_exposure["active_median_float_mv"].mean()) if not size_exposure.empty else np.nan,
        "latest_active_avg_float_mv": float(size_exposure["active_avg_float_mv"].iloc[-1]) if not size_exposure.empty else np.nan,
        "latest_active_avg_log_float_mv": float(size_exposure["active_avg_log_float_mv"].iloc[-1]) if not size_exposure.empty else np.nan,
    }
    return industry_exposure, industry_summary, size_exposure, size_summary


def neutralize_score_cross_sectionally(
    scored: pd.DataFrame,
    score_col: str,
    *,
    by_industry: bool = True,
    by_size: bool = True,
    output_col: str | None = None,
) -> pd.DataFrame:
    if not by_industry and not by_size:
        return scored.copy()

    out = scored.copy()
    neutral_col = output_col or f"{score_col}_neutralized"
    out[neutral_col] = out[score_col].astype("float32")

    eligible = out["tradable_universe"] & out[score_col].notna()
    if not eligible.any():
        return out

    work = out.loc[eligible, ["date", "industry", "float_mv", score_col]].copy()
    work["industry"] = work["industry"].fillna("UNKNOWN")
    work["float_mv"] = pd.to_numeric(work["float_mv"], errors="coerce").fillna(0.0)
    work["_neutral"] = work[score_col].astype(float)

    if by_industry:
        industry_mean = work.groupby(["date", "industry"], observed=False)["_neutral"].transform("mean")
        work["_neutral"] = work["_neutral"] - industry_mean

    if by_size:
        work["_log_float_mv"] = np.log(work["float_mv"].clip(lower=1.0))
        adjusted_groups: list[pd.DataFrame] = []
        for _, day in work.groupby("date", observed=False, sort=True):
            x = day["_log_float_mv"].to_numpy(dtype=float)
            y = day["_neutral"].to_numpy(dtype=float)
            if len(day) >= 3 and np.nanstd(x) > 0:
                beta = np.cov(x, y, ddof=0)[0, 1] / np.var(x)
                y = y - beta * (x - np.mean(x))
            day = day.copy()
            day["_neutral"] = y
            adjusted_groups.append(day)
        work = pd.concat(adjusted_groups, ignore_index=False) if adjusted_groups else work

    work["_neutral"] = work.groupby("date", observed=False)["_neutral"].transform(_safe_zscore)
    out.loc[work.index, neutral_col] = work["_neutral"].astype("float32")
    return out


def _summarize_size_exposure(
    df: pd.DataFrame,
    weight_col: str,
    avg_float_mv_col: str,
    avg_log_float_mv_col: str,
    median_float_mv_col: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for date, day in df.groupby("date", observed=False, sort=True):
        weights = day[weight_col].to_numpy(dtype=float)
        float_mv = day["float_mv"].to_numpy(dtype=float)
        log_float_mv = day["float_mv_log"].to_numpy(dtype=float)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            continue
        norm_weights = weights / weight_sum
        rows.append(
            {
                "date": date,
                avg_float_mv_col: float(np.average(float_mv, weights=norm_weights)),
                avg_log_float_mv_col: float(np.average(log_float_mv, weights=norm_weights)),
                median_float_mv_col: float(pd.Series(float_mv).median()),
            }
        )
    return pd.DataFrame(rows)


def _safe_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std <= 0:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    return (series - series.mean()) / std


def write_research_summary(
    output_path: Path,
    strategy_name: str,
    metrics: dict,
    benchmark_metrics: dict,
    diagnostics: dict,
    exposure_summary: dict | None = None,
    official_benchmark_metrics: dict | None = None,
) -> None:
    lines = [
        f"strategy={strategy_name}",
        f"annual_return_est={metrics.get('annual_return_est')}",
        f"annual_vol_est={metrics.get('annual_vol_est')}",
        f"sharpe_est={metrics.get('sharpe_est')}",
        f"max_drawdown={metrics.get('max_drawdown')}",
        f"excess_annual_return_est={benchmark_metrics.get('annual_return_est')}",
        f"excess_sharpe_est={benchmark_metrics.get('sharpe_est')}",
        f"mean_pearson_ic={diagnostics.get('mean_pearson_ic')}",
        f"mean_rank_ic={diagnostics.get('mean_rank_ic')}",
        f"rank_ic_ir={diagnostics.get('rank_ic_ir')}",
        f"mean_top_bottom_spread={diagnostics.get('mean_top_bottom_spread')}",
    ]
    if exposure_summary:
        lines.append(f"mean_active_avg_log_float_mv={exposure_summary.get('mean_active_avg_log_float_mv')}")
        lines.append(f"latest_active_avg_log_float_mv={exposure_summary.get('latest_active_avg_log_float_mv')}")
    if official_benchmark_metrics:
        lines.append(f"official_index_excess_annual_return_est={official_benchmark_metrics.get('annual_return_est')}")
        lines.append(f"official_index_excess_sharpe_est={official_benchmark_metrics.get('sharpe_est')}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_json(data: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
