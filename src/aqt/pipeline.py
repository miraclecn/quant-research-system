from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import gc

import pandas as pd

from aqt.backtest import build_positions, run_backtest, save_metrics, select_rebalance_dates
from aqt.config import AppConfig
from aqt.data import load_panel
from aqt.features import FEATURE_COLUMNS, add_features, build_factor_registry
from aqt.labels import add_labels
from aqt.models import fit_predict_models
from aqt.research import (
    build_universe_benchmark_positions,
    compare_to_benchmark,
    compare_to_official_index,
    compute_single_factor_group_backtest,
    compute_exposure_diagnostics,
    compute_signal_diagnostics,
    load_official_index_benchmark,
    neutralize_score_cross_sectionally,
    save_json,
    summarize_official_index_benchmark,
    summarize_feature_diagnostics,
    summarize_feature_importance,
    summarize_ridge_coefficients,
    summarize_ridge_coefficient_stability,
    summarize_selected_feature_frequency,
    winsorize_and_zscore_by_date,
    write_research_summary,
)
from aqt.universe import apply_universe_filters


def _write_run_config(cfg: AppConfig, output_path: Path, extra: dict | None = None) -> None:
    payload = asdict(cfg)
    payload["data"]["input_path"] = str(cfg.data.input_path)
    payload["output_dir"] = str(cfg.output_dir)
    if extra:
        payload["run_context"] = extra
    save_json(payload, output_path)


def _build_split_summary(split_results: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for split in split_results:
        base = {
            "split_id": split["split_id"],
            **split["window"],
        }
        for phase in ["valid", "test"]:
            phase_metrics = split.get(phase, {})
            for model_name, model_metrics in phase_metrics.items():
                rows.append(
                    {
                        **base,
                        "phase": phase,
                        "model": model_name,
                        "annual_return_est": model_metrics.get("annual_return_est"),
                        "annual_vol_est": model_metrics.get("annual_vol_est"),
                        "sharpe_est": model_metrics.get("sharpe_est"),
                        "max_drawdown": model_metrics.get("max_drawdown"),
                        "excess_annual_return_est": model_metrics.get("excess", {}).get("annual_return_est"),
                        "excess_sharpe_est": model_metrics.get("excess", {}).get("sharpe_est"),
                        "official_index_annual_return_est": model_metrics.get("official_index_benchmark", {}).get("annual_return_est"),
                        "official_index_sharpe_est": model_metrics.get("official_index_benchmark", {}).get("sharpe_est"),
                        "official_index_excess_annual_return_est": model_metrics.get("official_index_excess", {}).get("annual_return_est"),
                        "official_index_excess_sharpe_est": model_metrics.get("official_index_excess", {}).get("sharpe_est"),
                        "mean_rank_ic": model_metrics.get("diagnostics", {}).get("mean_rank_ic"),
                        "rank_ic_ir": model_metrics.get("diagnostics", {}).get("rank_ic_ir"),
                        "positive_rank_ic_ratio": model_metrics.get("diagnostics", {}).get("positive_rank_ic_ratio"),
                        "mean_top_bottom_spread": model_metrics.get("diagnostics", {}).get("mean_top_bottom_spread"),
                        "mean_active_avg_log_float_mv": model_metrics.get("size_bias", {}).get("mean_active_avg_log_float_mv"),
                    }
                )
    return pd.DataFrame(rows)


def _build_feature_screening_summary(output_dir: Path, split_ids: list[int]) -> pd.DataFrame:
    rows: list[dict] = []
    for split_id in split_ids:
        split_dir = output_dir / f"split_{split_id:02d}"
        for phase in ["valid", "test"]:
            path = split_dir / f"feature_diagnostics_{phase}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            best = df.sort_values(["mean_rank_ic", "rank_ic_ir"], ascending=False).iloc[0]
            rows.append(
                {
                    "split_id": split_id,
                    "phase": phase,
                    "best_feature": best.get("feature"),
                    "best_feature_mean_rank_ic": best.get("mean_rank_ic"),
                    "best_feature_rank_ic_ir": best.get("rank_ic_ir"),
                    "best_feature_positive_rank_ic_ratio": best.get("positive_rank_ic_ratio"),
                    "best_feature_mean_top_bottom_spread": best.get("mean_top_bottom_spread"),
                }
            )
    return pd.DataFrame(rows)


def _safe_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(0.0, index=series.index, dtype=float)
    return numeric.rank(pct=True, ascending=ascending, method="average").fillna(0.0)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _build_factor_evaluation(
    annual_report: pd.DataFrame,
    registry: pd.DataFrame,
    cfg: AppConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if annual_report.empty:
        raise ValueError("No annual single-factor reports were generated for evaluation.")

    factor_eval_cfg = cfg.train.factor_eval
    weights = factor_eval_cfg.weights
    year_count = int(annual_report["year"].nunique())
    min_years_required = (
        factor_eval_cfg.min_years_required_single_window
        if year_count <= 1
        else max(
            factor_eval_cfg.min_years_required_multi_window,
            min(factor_eval_cfg.max_years_required, year_count),
        )
    )

    stability = (
        annual_report.groupby("feature", observed=False)
        .agg(
            years_covered=("year", "nunique"),
            direction=("direction", lambda s: int(pd.Series(s).mode().iloc[0])),
            sign_flipped=("sign_flipped", "max"),
            mean_rank_ic_mean=("mean_rank_ic", "mean"),
            mean_rank_ic_std=("mean_rank_ic", "std"),
            rank_ic_ir_mean=("rank_ic_ir", "mean"),
            positive_rank_ic_ratio_mean=("positive_rank_ic_ratio", "mean"),
            mean_top_bottom_spread_mean=("mean_top_bottom_spread", "mean"),
            long_short_annual_return_est_mean=("long_short_annual_return_est", "mean"),
            long_short_annual_return_est_std=("long_short_annual_return_est", "std"),
            top_bucket_official_index_excess_annual_return_est_mean=("top_bucket_official_index_excess_annual_return_est", "mean"),
            top_bucket_official_index_excess_annual_return_est_std=("top_bucket_official_index_excess_annual_return_est", "std"),
            bucket_return_spearman_mean=("bucket_return_spearman", "mean"),
            monotonic_year_ratio=("bucket_return_monotonic_increasing", lambda s: float(s.fillna(False).mean())),
            positive_ic_year_ratio=("mean_rank_ic", lambda s: float((s > 0).mean())),
            positive_long_short_year_ratio=("long_short_annual_return_est", lambda s: float((s > 0).mean())),
            positive_top_bucket_excess_year_ratio=("top_bucket_official_index_excess_annual_return_est", lambda s: float((s > 0).mean())),
            latest_year=("year", "max"),
        )
        .reset_index()
    )
    latest_rows = annual_report.sort_values(["feature", "year"]).groupby("feature", as_index=False).tail(1)
    latest_rows = latest_rows.rename(
        columns={
            "mean_rank_ic": "latest_mean_rank_ic",
            "rank_ic_ir": "latest_rank_ic_ir",
            "long_short_annual_return_est": "latest_long_short_annual_return_est",
            "long_short_sharpe_est": "latest_long_short_sharpe_est",
            "top_bucket_official_index_excess_annual_return_est": "latest_top_bucket_official_index_excess_annual_return_est",
            "bucket_return_spearman": "latest_bucket_return_spearman",
        }
    )[[
        "feature",
        "latest_mean_rank_ic",
        "latest_rank_ic_ir",
        "latest_long_short_annual_return_est",
        "latest_long_short_sharpe_est",
        "latest_top_bucket_official_index_excess_annual_return_est",
        "latest_bucket_return_spearman",
    ]]
    evaluation = stability.merge(latest_rows, on="feature", how="left")
    evaluation = registry.merge(evaluation, on="feature", how="left")

    evaluation["abs_mean_rank_ic_mean"] = evaluation["mean_rank_ic_mean"].abs()
    evaluation["abs_rank_ic_ir_mean"] = evaluation["rank_ic_ir_mean"].abs()
    evaluation["quality_score"] = (
        weights.abs_rank_ic_ir_mean * _safe_rank(evaluation["abs_rank_ic_ir_mean"], ascending=True)
        + weights.abs_mean_rank_ic_mean * _safe_rank(evaluation["abs_mean_rank_ic_mean"], ascending=True)
        + weights.bucket_return_spearman_mean * _safe_rank(evaluation["bucket_return_spearman_mean"], ascending=True)
        + weights.positive_top_bucket_excess_year_ratio * _safe_rank(evaluation["positive_top_bucket_excess_year_ratio"], ascending=True)
        + weights.latest_top_bucket_official_index_excess_annual_return_est * _safe_rank(evaluation["latest_top_bucket_official_index_excess_annual_return_est"], ascending=True)
        + weights.monotonic_year_ratio * _safe_rank(evaluation["monotonic_year_ratio"], ascending=True)
        + weights.positive_ic_year_ratio * _safe_rank(evaluation["positive_ic_year_ratio"], ascending=True)
        - weights.mean_rank_ic_std_penalty * _safe_rank(evaluation["mean_rank_ic_std"], ascending=True)
    )
    evaluation["quality_score"] = evaluation["quality_score"].fillna(0.0).round(6)
    evaluation["quality_tier"] = "watch"
    evaluation.loc[
        evaluation["quality_score"] >= evaluation["quality_score"].quantile(factor_eval_cfg.core_quantile),
        "quality_tier",
    ] = "core"
    evaluation.loc[
        (evaluation["quality_tier"] == "watch")
        & (evaluation["quality_score"] >= evaluation["quality_score"].quantile(factor_eval_cfg.candidate_quantile)),
        "quality_tier",
    ] = "candidate"
    evaluation["pass_single_factor_gate"] = (
        evaluation["years_covered"].fillna(0) >= min_years_required
    ) & (
        evaluation["abs_rank_ic_ir_mean"].fillna(0.0) >= factor_eval_cfg.min_abs_rank_ic_ir
    ) & (
        evaluation["bucket_return_spearman_mean"].fillna(0.0) >= factor_eval_cfg.min_bucket_return_spearman
    ) & (
        evaluation["positive_top_bucket_excess_year_ratio"].fillna(0.0) >= factor_eval_cfg.min_positive_top_bucket_excess_year_ratio
    ) & (
        evaluation["latest_top_bucket_official_index_excess_annual_return_est"].fillna(-1.0) > factor_eval_cfg.min_latest_top_bucket_excess
    )
    evaluation = evaluation.sort_values(
        ["pass_single_factor_gate", "quality_score", "abs_rank_ic_ir_mean", "bucket_return_spearman_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    whitelist = evaluation.loc[evaluation["pass_single_factor_gate"]].copy()
    whitelist["priority_rank"] = range(1, len(whitelist) + 1)
    whitelist["selected_reason"] = "pass_single_factor_gate"
    return evaluation, whitelist


def _select_features_from_diagnostics(
    diagnostics: pd.DataFrame,
    top_k: int,
    min_rank_ic: float,
) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame(columns=["feature", "train_mean_rank_ic", "train_rank_ic_ir", "direction", "sign_flipped"])
    ranked = diagnostics.copy()
    ranked["abs_mean_rank_ic"] = ranked["mean_rank_ic"].abs()
    ranked["abs_rank_ic_ir"] = ranked["rank_ic_ir"].abs()
    ranked["direction"] = ranked["mean_rank_ic"].apply(lambda x: 1 if pd.isna(x) or x >= 0 else -1)
    ranked["sign_flipped"] = ranked["direction"].eq(-1)
    ranked = ranked.loc[ranked["abs_mean_rank_ic"].fillna(0.0) >= min_rank_ic].copy()
    if ranked.empty:
        ranked = diagnostics.copy()
        ranked["abs_mean_rank_ic"] = ranked["mean_rank_ic"].abs()
        ranked["abs_rank_ic_ir"] = ranked["rank_ic_ir"].abs()
        ranked["direction"] = ranked["mean_rank_ic"].apply(lambda x: 1 if pd.isna(x) or x >= 0 else -1)
        ranked["sign_flipped"] = ranked["direction"].eq(-1)
    ranked = ranked.sort_values(
        ["abs_mean_rank_ic", "abs_rank_ic_ir", "mean_top_bottom_spread"],
        ascending=False,
    )
    selected = ranked.head(top_k).copy()
    return selected[[
        "feature",
        "mean_rank_ic",
        "rank_ic_ir",
        "positive_rank_ic_ratio",
        "mean_top_bottom_spread",
        "direction",
        "sign_flipped",
    ]].rename(
        columns={
            "mean_rank_ic": "train_mean_rank_ic",
            "rank_ic_ir": "train_rank_ic_ir",
            "positive_rank_ic_ratio": "train_positive_rank_ic_ratio",
            "mean_top_bottom_spread": "train_mean_top_bottom_spread",
        }
    )


def _apply_feature_directions(
    df: pd.DataFrame,
    selected_features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    oriented_features: list[str] = []
    for row in selected_features.to_dict(orient="records"):
        feature = row["feature"]
        direction = int(row.get("direction", 1))
        oriented_name = f"{feature}_oriented"
        out[oriented_name] = out[feature].astype("float32") * direction
        oriented_features.append(oriented_name)
    return out, oriented_features


def _deduplicate_selected_features(
    train_df: pd.DataFrame,
    selected_features: pd.DataFrame,
    max_corr: float,
) -> pd.DataFrame:
    if selected_features.empty or max_corr >= 1.0:
        return selected_features

    keep_rows: list[dict] = []
    kept_features: list[str] = []
    for row in selected_features.to_dict(orient="records"):
        feature = row["feature"]
        candidate = train_df[feature]
        is_duplicate = False
        for kept_feature in kept_features:
            corr = candidate.corr(train_df[kept_feature], method="pearson")
            if pd.notna(corr) and abs(corr) >= max_corr:
                is_duplicate = True
                break
        if not is_duplicate:
            keep_rows.append(row)
            kept_features.append(feature)

    return pd.DataFrame(keep_rows, columns=selected_features.columns)


def _run_strategy(
    merged: pd.DataFrame,
    target_col: str,
    score_col: str,
    cfg: AppConfig,
    output_stem: str,
    write_default_alias: bool = False,
) -> dict:
    scored = merged.copy()
    scored["score"] = scored[score_col]

    positions = build_positions(scored, cfg.portfolio)
    portfolio, metrics = run_backtest(scored, positions, cfg.portfolio)
    benchmark_positions = build_universe_benchmark_positions(scored, cfg.portfolio)
    benchmark_portfolio, benchmark_metrics = run_backtest(scored, benchmark_positions, cfg.portfolio)
    excess_curve, excess_metrics = compare_to_benchmark(portfolio, benchmark_portfolio)
    official_benchmark = load_official_index_benchmark(cfg, portfolio["date"])
    official_benchmark_metrics = summarize_official_index_benchmark(official_benchmark)
    official_excess_curve, official_excess_metrics = compare_to_official_index(portfolio, official_benchmark)
    ic_df, diagnostics, bucket_df = compute_signal_diagnostics(scored, score_col=score_col, target_col=target_col)
    industry_exposure, industry_summary, size_exposure, size_summary = compute_exposure_diagnostics(scored, positions)

    positions.sort_values(["date", "target_weight"], ascending=[True, False]).to_csv(
        cfg.output_dir / f"positions_{output_stem}.csv",
        index=False,
    )
    portfolio.to_csv(cfg.output_dir / f"backtest_equity_{output_stem}.csv", index=False)
    benchmark_portfolio.to_csv(cfg.output_dir / f"benchmark_equity_{output_stem}.csv", index=False)
    excess_curve.to_csv(cfg.output_dir / f"excess_return_{output_stem}.csv", index=False)
    official_benchmark.to_csv(cfg.output_dir / f"official_index_benchmark_{output_stem}.csv", index=False)
    official_excess_curve.to_csv(cfg.output_dir / f"official_index_excess_{output_stem}.csv", index=False)
    ic_df.to_csv(cfg.output_dir / f"ic_timeseries_{output_stem}.csv", index=False)
    bucket_df.to_csv(cfg.output_dir / f"bucket_returns_{output_stem}.csv", index=False)
    industry_exposure.to_csv(cfg.output_dir / f"industry_exposure_{output_stem}.csv", index=False)
    size_exposure.to_csv(cfg.output_dir / f"size_exposure_{output_stem}.csv", index=False)
    save_json(diagnostics, cfg.output_dir / f"diagnostics_{output_stem}.json")
    save_json(benchmark_metrics, cfg.output_dir / f"benchmark_metrics_{output_stem}.json")
    save_json(excess_metrics, cfg.output_dir / f"excess_metrics_{output_stem}.json")
    save_json(official_benchmark_metrics, cfg.output_dir / f"official_index_benchmark_metrics_{output_stem}.json")
    save_json(official_excess_metrics, cfg.output_dir / f"official_index_excess_metrics_{output_stem}.json")
    save_json(industry_summary, cfg.output_dir / f"industry_bias_summary_{output_stem}.json")
    save_json(size_summary, cfg.output_dir / f"size_bias_summary_{output_stem}.json")
    write_research_summary(
        cfg.output_dir / f"research_summary_{output_stem}.txt",
        strategy_name=output_stem,
        metrics=metrics,
        benchmark_metrics=excess_metrics,
        diagnostics=diagnostics,
        exposure_summary=size_summary,
        official_benchmark_metrics=official_excess_metrics,
    )

    latest_positions = positions.loc[positions["date"] == positions["date"].max()].copy()
    latest_positions.to_csv(cfg.output_dir / f"latest_signals_{output_stem}.csv", index=False)

    if write_default_alias:
        positions.sort_values(["date", "target_weight"], ascending=[True, False]).to_csv(
            cfg.output_dir / "positions.csv",
            index=False,
        )
        portfolio.to_csv(cfg.output_dir / "backtest_equity.csv", index=False)
        benchmark_portfolio.to_csv(cfg.output_dir / "benchmark_equity.csv", index=False)
        excess_curve.to_csv(cfg.output_dir / "excess_return.csv", index=False)
        official_benchmark.to_csv(cfg.output_dir / "official_index_benchmark.csv", index=False)
        official_excess_curve.to_csv(cfg.output_dir / "official_index_excess.csv", index=False)
        industry_exposure.to_csv(cfg.output_dir / "industry_exposure.csv", index=False)
        size_exposure.to_csv(cfg.output_dir / "size_exposure.csv", index=False)
        latest_positions.to_csv(cfg.output_dir / "latest_signals.csv", index=False)

    return {
        **metrics,
        "benchmark": benchmark_metrics,
        "excess": excess_metrics,
        "official_index_benchmark": official_benchmark_metrics,
        "official_index_excess": official_excess_metrics,
        "diagnostics": diagnostics,
        "industry_bias": industry_summary,
        "size_bias": size_summary,
    }


def run_pipeline(cfg: AppConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(cfg, cfg.output_dir / "run_config.json", extra={"command": "run"})

    panel = load_panel(
        cfg.data.input_path,
        index_code=cfg.data.index_code,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
    )
    panel = add_features(panel)
    panel = add_labels(panel, cfg.labels.primary_horizon, cfg.labels.secondary_horizon)
    panel = apply_universe_filters(panel, cfg.data)

    features = cfg.train.features or FEATURE_COLUMNS
    target_col = f"future_return_{cfg.labels.primary_horizon}d"
    panel_columns_for_merge = [
        "date",
        "symbol",
        "open",
        "is_paused",
        "is_limit_up",
        "tradable_universe",
        "industry",
        "float_mv",
        target_col,
    ]

    modeling = panel.loc[panel["tradable_universe"]].copy()
    modeling = modeling[["date", "symbol", "tradable_universe", target_col, *features]].dropna(subset=[target_col]).reset_index(drop=True)

    unique_dates = sorted(modeling["date"].drop_duplicates())
    if len(unique_dates) < cfg.train.train_days:
        raise ValueError("Not enough dates for the configured training window.")

    unique_date_set = set(unique_dates)
    date_to_index = {date: idx for idx, date in enumerate(unique_dates)}
    rebalance_dates = select_rebalance_dates(pd.Series(unique_dates), cfg.portfolio.rebalance_weekday)
    rebalance_dates = [date for date in rebalance_dates if date in unique_date_set]
    if not rebalance_dates:
        raise ValueError("No rebalance dates were generated from the available data.")

    predictions = []
    ridge_importances: list = []
    lgbm_importances: list = []
    for rebalance_date in rebalance_dates:
        split_end = date_to_index[rebalance_date]
        if split_end < cfg.train.train_days:
            continue

        train_dates = unique_dates[split_end - cfg.train.train_days: split_end]
        test_dates = [rebalance_date]

        train_df = modeling.loc[modeling["date"].isin(train_dates)].copy()
        test_df = modeling.loc[modeling["date"].isin(test_dates)].copy()

        if len(train_df) < cfg.train.min_train_rows or test_df.empty:
            continue

        outputs = fit_predict_models(
            train_df=train_df,
            test_df=test_df,
            features=features,
            target_col=target_col,
            random_state=cfg.train.random_state,
            lgbm_cfg=cfg.train.lgbm,
        )
        fold_pred = test_df[["date", "symbol", "tradable_universe"]].copy()
        fold_pred["ridge_score"] = outputs.ridge_pred
        fold_pred["lgbm_score"] = outputs.lgbm_pred
        predictions.append(fold_pred)
        ridge_importances.append(outputs.ridge_coef)
        lgbm_importances.append(outputs.lgbm_feature_importance)

    if not predictions:
        raise ValueError("No prediction folds were generated. Check your data volume and config.")

    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df = pred_df.drop_duplicates(subset=["date", "symbol"], keep="last")

    merge_source = panel[panel_columns_for_merge].copy()
    merged = merge_source.merge(pred_df, on=["date", "symbol", "tradable_universe"], how="inner")
    merged["ridge_score"] = merged["ridge_score"].astype("float32")
    merged["lgbm_score"] = merged["lgbm_score"].astype("float32")
    del panel
    del modeling
    del merge_source
    del pred_df
    gc.collect()

    merged.to_parquet(cfg.output_dir / "predictions.parquet", index=False)
    feature_importance = summarize_feature_importance(features, ridge_importances, lgbm_importances)
    feature_importance.to_csv(cfg.output_dir / "feature_importance_summary.csv", index=False)
    metrics = {
        "ridge": _run_strategy(merged, target_col, "ridge_score", cfg, "ridge"),
        "lgbm": _run_strategy(merged, target_col, "lgbm_score", cfg, "lgbm", write_default_alias=True),
    }
    if cfg.train.neutralize_scores:
        neutralized = neutralize_score_cross_sectionally(
            merged,
            "lgbm_score",
            by_industry=cfg.train.neutralize_industry,
            by_size=cfg.train.neutralize_size,
            output_col="lgbm_score_neutralized",
        )
        metrics["lgbm_neutralized"] = _run_strategy(
            neutralized,
            target_col,
            "lgbm_score_neutralized",
            cfg,
            "lgbm_neutralized",
        )
    save_metrics(metrics, cfg.output_dir / "metrics.json")


def run_research_pipeline(
    cfg: AppConfig,
    research_start: str,
    research_end: str,
    train_months: int,
    valid_months: int,
    test_months: int,
    step_months: int | None = None,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(
        cfg,
        cfg.output_dir / "run_config.json",
        extra={
            "command": "research-run",
            "research_start": research_start,
            "research_end": research_end,
            "train_months": train_months,
            "valid_months": valid_months,
            "test_months": test_months,
            "step_months": step_months or valid_months,
        },
    )

    panel = load_panel(
        cfg.data.input_path,
        index_code=cfg.data.index_code,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
    )
    panel = add_features(panel)
    panel = add_labels(panel, cfg.labels.primary_horizon, cfg.labels.secondary_horizon)
    panel = apply_universe_filters(panel, cfg.data)

    features = cfg.train.features or FEATURE_COLUMNS
    target_col = f"future_return_{cfg.labels.primary_horizon}d"
    panel_columns_for_merge = [
        "date",
        "symbol",
        "open",
        "is_paused",
        "is_limit_up",
        "tradable_universe",
        "industry",
        "float_mv",
        target_col,
    ]

    modeling = panel.loc[panel["tradable_universe"]].copy()
    modeling = modeling[["date", "symbol", "tradable_universe", target_col, *features]].dropna(subset=[target_col]).reset_index(drop=True)

    unique_dates = sorted(modeling["date"].drop_duplicates())
    if len(unique_dates) < cfg.train.train_days:
        raise ValueError("Not enough dates for the configured training window.")

    unique_date_set = set(unique_dates)
    date_to_index = {date: idx for idx, date in enumerate(unique_dates)}
    rebalance_dates = select_rebalance_dates(pd.Series(unique_dates), cfg.portfolio.rebalance_weekday)
    rebalance_dates = [date for date in rebalance_dates if date in unique_date_set]
    if not rebalance_dates:
        raise ValueError("No rebalance dates were generated from the available data.")

    step_months = step_months or valid_months
    research_start_ts = pd.Timestamp(research_start)
    research_end_ts = pd.Timestamp(research_end)
    merge_source = panel[panel_columns_for_merge].copy()

    split_results = []
    split_id = 1
    cursor = research_start_ts
    while True:
        train_start_ts = cursor
        train_end_ts = train_start_ts + pd.DateOffset(months=train_months)
        valid_start_ts = train_end_ts
        valid_end_ts = valid_start_ts + pd.DateOffset(months=valid_months)
        test_start_ts = valid_end_ts
        test_end_ts = test_start_ts + pd.DateOffset(months=test_months)

        if test_end_ts > research_end_ts + pd.Timedelta(days=1):
            break

        train_df = modeling.loc[
            (modeling["date"] >= train_start_ts) & (modeling["date"] < train_end_ts)
        ].copy()
        eval_df = modeling.loc[
            (modeling["date"] >= valid_start_ts) & (modeling["date"] < test_end_ts)
        ].copy()
        valid_dates_df = modeling.loc[
            (modeling["date"] >= valid_start_ts) & (modeling["date"] < valid_end_ts), ["date"]
        ]
        test_dates_df = modeling.loc[
            (modeling["date"] >= test_start_ts) & (modeling["date"] < test_end_ts), ["date"]
        ]

        if len(train_df) < cfg.train.min_train_rows or eval_df.empty or valid_dates_df.empty or test_dates_df.empty:
            cursor = cursor + pd.DateOffset(months=step_months)
            split_id += 1
            continue

        outputs = fit_predict_models(
            train_df=train_df,
            test_df=eval_df,
            features=features,
            target_col=target_col,
            random_state=cfg.train.random_state,
            lgbm_cfg=cfg.train.lgbm,
        )
        pred_df = eval_df[["date", "symbol", "tradable_universe"]].copy()
        pred_df["ridge_score"] = outputs.ridge_pred.astype("float32")
        pred_df["lgbm_score"] = outputs.lgbm_pred.astype("float32")

        merged = merge_source.merge(pred_df, on=["date", "symbol", "tradable_universe"], how="inner")
        valid_merged = merged.loc[(merged["date"] >= valid_start_ts) & (merged["date"] < valid_end_ts)].copy()
        test_merged = merged.loc[(merged["date"] >= test_start_ts) & (merged["date"] < test_end_ts)].copy()
        if valid_merged.empty or test_merged.empty:
            cursor = cursor + pd.DateOffset(months=step_months)
            split_id += 1
            continue

        split_dir = cfg.output_dir / f"split_{split_id:02d}"
        split_dir.mkdir(parents=True, exist_ok=True)
        split_cfg = replace(cfg, output_dir=split_dir)

        valid_merged.to_parquet(split_dir / "predictions_valid.parquet", index=False)
        test_merged.to_parquet(split_dir / "predictions_test.parquet", index=False)
        feature_importance = summarize_feature_importance(
            features,
            [outputs.ridge_coef],
            [outputs.lgbm_feature_importance],
        )
        feature_importance.to_csv(split_dir / "feature_importance_summary.csv", index=False)
        valid_feature_diagnostics = summarize_feature_diagnostics(valid_merged, features, target_col)
        test_feature_diagnostics = summarize_feature_diagnostics(test_merged, features, target_col)
        valid_feature_diagnostics.to_csv(split_dir / "feature_diagnostics_valid.csv", index=False)
        test_feature_diagnostics.to_csv(split_dir / "feature_diagnostics_test.csv", index=False)

        split_metrics = {
            "window": {
                "train_start": train_start_ts.strftime("%Y-%m-%d"),
                "train_end": (train_end_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "valid_start": valid_start_ts.strftime("%Y-%m-%d"),
                "valid_end": (valid_end_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "test_start": test_start_ts.strftime("%Y-%m-%d"),
                "test_end": (test_end_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            },
            "valid": {
                "ridge": _run_strategy(valid_merged, target_col, "ridge_score", split_cfg, "valid_ridge"),
                "lgbm": _run_strategy(valid_merged, target_col, "lgbm_score", split_cfg, "valid_lgbm"),
            },
            "test": {
                "ridge": _run_strategy(test_merged, target_col, "ridge_score", split_cfg, "test_ridge"),
                "lgbm": _run_strategy(test_merged, target_col, "lgbm_score", split_cfg, "test_lgbm"),
            },
        }
        if cfg.train.neutralize_scores:
            valid_neutralized = neutralize_score_cross_sectionally(
                valid_merged,
                "lgbm_score",
                by_industry=cfg.train.neutralize_industry,
                by_size=cfg.train.neutralize_size,
                output_col="lgbm_score_neutralized",
            )
            test_neutralized = neutralize_score_cross_sectionally(
                test_merged,
                "lgbm_score",
                by_industry=cfg.train.neutralize_industry,
                by_size=cfg.train.neutralize_size,
                output_col="lgbm_score_neutralized",
            )
            split_metrics["valid"]["lgbm_neutralized"] = _run_strategy(
                valid_neutralized,
                target_col,
                "lgbm_score_neutralized",
                split_cfg,
                "valid_lgbm_neutralized",
            )
            split_metrics["test"]["lgbm_neutralized"] = _run_strategy(
                test_neutralized,
                target_col,
                "lgbm_score_neutralized",
                split_cfg,
                "test_lgbm_neutralized",
            )
        save_metrics(split_metrics, split_dir / "research_metrics.json")
        split_results.append({"split_id": split_id, **split_metrics})

        del pred_df
        del merged
        del valid_merged
        del test_merged
        gc.collect()
        cursor = cursor + pd.DateOffset(months=step_months)
        split_id += 1

    del panel
    del modeling
    del merge_source
    gc.collect()

    if not split_results:
        raise ValueError("No rolling research splits were generated. Check the research window and window sizes.")

    save_metrics(
        {
            "research_start": research_start,
            "research_end": research_end,
            "train_months": train_months,
            "valid_months": valid_months,
            "test_months": test_months,
            "step_months": step_months,
            "splits": split_results,
        },
        cfg.output_dir / "research_metrics.json",
    )
    split_ids = [split["split_id"] for split in split_results]
    _build_split_summary(split_results).to_csv(cfg.output_dir / "split_summary.csv", index=False)
    _build_feature_screening_summary(cfg.output_dir, split_ids).to_csv(
        cfg.output_dir / "feature_screening_summary.csv",
        index=False,
    )


def run_factor_chain_pipeline(
    cfg: AppConfig,
    research_start: str,
    research_end: str,
    train_months: int,
    valid_months: int,
    test_months: int,
    step_months: int | None = None,
    factor_top_k: int = 10,
    factor_min_rank_ic: float = 0.0,
    factor_max_corr: float = 0.9,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(
        cfg,
        cfg.output_dir / "run_config.json",
        extra={
            "command": "factor-chain-run",
            "research_start": research_start,
            "research_end": research_end,
            "train_months": train_months,
            "valid_months": valid_months,
            "test_months": test_months,
            "step_months": step_months or valid_months,
            "factor_top_k": factor_top_k,
            "factor_min_rank_ic": factor_min_rank_ic,
            "factor_max_corr": factor_max_corr,
        },
    )

    panel = load_panel(
        cfg.data.input_path,
        index_code=cfg.data.index_code,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
    )
    panel = add_features(panel)
    panel = add_labels(panel, cfg.labels.primary_horizon, cfg.labels.secondary_horizon)
    panel = apply_universe_filters(panel, cfg.data)

    all_features = cfg.train.features or FEATURE_COLUMNS
    factor_eval_dir = cfg.output_dir / "factor_research"
    factor_eval_cfg = replace(cfg, output_dir=factor_eval_dir)
    run_single_factor_pipeline(
        factor_eval_cfg,
        research_start=research_start,
        research_end=research_end,
        bucket_count=5,
        top_k=max(factor_top_k, 20),
    )
    factor_registry = pd.read_csv(factor_eval_dir / "factor_registry.csv")
    factor_evaluation = pd.read_csv(factor_eval_dir / "factor_evaluation.csv")
    factor_whitelist = pd.read_csv(factor_eval_dir / "factor_whitelist.csv")
    if factor_whitelist.empty:
        fallback_top_n = cfg.train.factor_eval.fallback_top_n
        fallback_count = max(1, factor_top_k) if factor_top_k > 0 else min(fallback_top_n, len(factor_evaluation))
        factor_whitelist = factor_evaluation.head(fallback_count).copy()
        factor_whitelist["pass_single_factor_gate"] = False
        factor_whitelist["selected_reason"] = "fallback_top_quality_score"
    elif "selected_reason" not in factor_whitelist.columns:
        factor_whitelist["selected_reason"] = "pass_single_factor_gate"
    factor_registry.to_csv(cfg.output_dir / "factor_registry.csv", index=False)
    factor_evaluation.to_csv(cfg.output_dir / "factor_evaluation.csv", index=False)
    factor_whitelist.to_csv(cfg.output_dir / "factor_whitelist.csv", index=False)
    whitelist_candidates = factor_whitelist.loc[
        factor_whitelist["feature"].isin(all_features)
    ].copy()
    if factor_top_k > 0:
        whitelist_candidates = whitelist_candidates.head(factor_top_k).copy()
    if whitelist_candidates.empty:
        raise ValueError("No whitelist factors overlap with the configured training feature set.")

    target_col = f"future_return_{cfg.labels.primary_horizon}d"
    panel_columns_for_merge = [
        "date",
        "symbol",
        "open",
        "is_paused",
        "is_limit_up",
        "tradable_universe",
        "industry",
        "float_mv",
        target_col,
        *all_features,
    ]

    modeling = panel.loc[panel["tradable_universe"]].copy()
    modeling = modeling[["date", "symbol", "tradable_universe", target_col, *all_features]].dropna(subset=[target_col]).reset_index(drop=True)

    step_months = step_months or valid_months
    research_start_ts = pd.Timestamp(research_start)
    research_end_ts = pd.Timestamp(research_end)
    merge_source = panel[panel_columns_for_merge].copy()

    split_results = []
    split_id = 1
    cursor = research_start_ts
    while True:
        train_start_ts = cursor
        train_end_ts = train_start_ts + pd.DateOffset(months=train_months)
        valid_start_ts = train_end_ts
        valid_end_ts = valid_start_ts + pd.DateOffset(months=valid_months)
        test_start_ts = valid_end_ts
        test_end_ts = test_start_ts + pd.DateOffset(months=test_months)

        if test_end_ts > research_end_ts + pd.Timedelta(days=1):
            break

        train_df = modeling.loc[
            (modeling["date"] >= train_start_ts) & (modeling["date"] < train_end_ts)
        ].copy()
        eval_df = modeling.loc[
            (modeling["date"] >= valid_start_ts) & (modeling["date"] < test_end_ts)
        ].copy()
        if len(train_df) < cfg.train.min_train_rows or eval_df.empty:
            cursor = cursor + pd.DateOffset(months=step_months)
            split_id += 1
            continue

        split_dir = cfg.output_dir / f"split_{split_id:02d}"
        split_dir.mkdir(parents=True, exist_ok=True)
        split_cfg = replace(cfg, output_dir=split_dir)

        selected_features = whitelist_candidates[[
            "feature",
            "direction",
            "sign_flipped",
            "quality_score",
            "quality_tier",
            "mean_rank_ic_mean",
            "rank_ic_ir_mean",
            "positive_rank_ic_ratio_mean",
            "mean_top_bottom_spread_mean",
        ]].rename(
            columns={
                "mean_rank_ic_mean": "train_mean_rank_ic",
                "rank_ic_ir_mean": "train_rank_ic_ir",
                "positive_rank_ic_ratio_mean": "train_positive_rank_ic_ratio",
                "mean_top_bottom_spread_mean": "train_mean_top_bottom_spread",
            }
        ).copy()
        train_feature_diagnostics = summarize_feature_diagnostics(train_df, selected_features["feature"].tolist(), target_col)
        train_feature_diagnostics.to_csv(split_dir / "feature_diagnostics_train.csv", index=False)
        selected_features = selected_features.loc[
            selected_features["train_mean_rank_ic"].abs().fillna(0.0) >= factor_min_rank_ic
        ].copy()
        if selected_features.empty:
            selected_features = whitelist_candidates[[
                "feature",
                "direction",
                "sign_flipped",
                "quality_score",
                "quality_tier",
                "mean_rank_ic_mean",
                "rank_ic_ir_mean",
                "positive_rank_ic_ratio_mean",
                "mean_top_bottom_spread_mean",
            ]].rename(
                columns={
                    "mean_rank_ic_mean": "train_mean_rank_ic",
                    "rank_ic_ir_mean": "train_rank_ic_ir",
                    "positive_rank_ic_ratio_mean": "train_positive_rank_ic_ratio",
                    "mean_top_bottom_spread_mean": "train_mean_top_bottom_spread",
                }
            ).copy()
        selected_features = _deduplicate_selected_features(train_df, selected_features, factor_max_corr)
        if selected_features.empty:
            cursor = cursor + pd.DateOffset(months=step_months)
            split_id += 1
            continue

        selected_features.to_csv(split_dir / "selected_features.csv", index=False)
        train_oriented, oriented_features = _apply_feature_directions(train_df, selected_features)
        eval_oriented, _ = _apply_feature_directions(eval_df, selected_features)
        outputs = fit_predict_models(
            train_df=train_oriented,
            test_df=eval_oriented,
            features=oriented_features,
            target_col=target_col,
            random_state=cfg.train.random_state,
            lgbm_cfg=cfg.train.lgbm,
        )

        pred_df = eval_df[["date", "symbol", "tradable_universe"]].copy()
        pred_df["ridge_score"] = outputs.ridge_pred.astype("float32")
        pred_df["lgbm_score"] = outputs.lgbm_pred.astype("float32")

        merged = merge_source.merge(pred_df, on=["date", "symbol", "tradable_universe"], how="inner")
        valid_merged = merged.loc[(merged["date"] >= valid_start_ts) & (merged["date"] < valid_end_ts)].copy()
        test_merged = merged.loc[(merged["date"] >= test_start_ts) & (merged["date"] < test_end_ts)].copy()
        if valid_merged.empty or test_merged.empty:
            cursor = cursor + pd.DateOffset(months=step_months)
            split_id += 1
            continue

        feature_importance = summarize_feature_importance(
            selected_features["feature"].tolist(),
            [outputs.ridge_coef],
            [outputs.lgbm_feature_importance],
        )
        feature_importance.to_csv(split_dir / "feature_importance_summary.csv", index=False)
        ridge_coefficients = summarize_ridge_coefficients(selected_features, outputs.ridge_coef)
        ridge_coefficients.to_csv(split_dir / "ridge_coefficients.csv", index=False)
        selected_feature_names = selected_features["feature"].tolist()
        summarize_feature_diagnostics(valid_merged, selected_feature_names, target_col).to_csv(
            split_dir / "feature_diagnostics_valid.csv", index=False
        )
        summarize_feature_diagnostics(test_merged, selected_feature_names, target_col).to_csv(
            split_dir / "feature_diagnostics_test.csv", index=False
        )

        split_metrics = {
            "window": {
                "train_start": train_start_ts.strftime("%Y-%m-%d"),
                "train_end": (train_end_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "valid_start": valid_start_ts.strftime("%Y-%m-%d"),
                "valid_end": (valid_end_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "test_start": test_start_ts.strftime("%Y-%m-%d"),
                "test_end": (test_end_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "selected_feature_count": len(selected_feature_names),
                "selected_features": selected_feature_names,
            },
            "valid": {
                "ridge": _run_strategy(valid_merged, target_col, "ridge_score", split_cfg, "valid_ridge"),
                "lgbm": _run_strategy(valid_merged, target_col, "lgbm_score", split_cfg, "valid_lgbm"),
            },
            "test": {
                "ridge": _run_strategy(test_merged, target_col, "ridge_score", split_cfg, "test_ridge"),
                "lgbm": _run_strategy(test_merged, target_col, "lgbm_score", split_cfg, "test_lgbm"),
            },
        }
        save_metrics(split_metrics, split_dir / "research_metrics.json")
        split_results.append({"split_id": split_id, **split_metrics})

        cursor = cursor + pd.DateOffset(months=step_months)
        split_id += 1

    if not split_results:
        raise ValueError("No factor-chain research splits were generated. Check the research window and window sizes.")

    save_metrics(
        {
            "research_start": research_start,
            "research_end": research_end,
            "train_months": train_months,
            "valid_months": valid_months,
            "test_months": test_months,
            "step_months": step_months,
            "factor_top_k": factor_top_k,
            "factor_min_rank_ic": factor_min_rank_ic,
            "factor_max_corr": factor_max_corr,
            "splits": split_results,
        },
        cfg.output_dir / "research_metrics.json",
    )
    split_ids = [split["split_id"] for split in split_results]
    _build_split_summary(split_results).to_csv(cfg.output_dir / "split_summary.csv", index=False)
    _build_feature_screening_summary(cfg.output_dir, split_ids).to_csv(
        cfg.output_dir / "feature_screening_summary.csv",
        index=False,
    )
    summarize_selected_feature_frequency(cfg.output_dir, split_ids).to_csv(
        cfg.output_dir / "selected_feature_frequency.csv",
        index=False,
    )
    summarize_ridge_coefficient_stability(cfg.output_dir, split_ids).to_csv(
        cfg.output_dir / "ridge_coefficient_stability.csv",
        index=False,
    )
    selected_frequency = _read_csv_or_empty(cfg.output_dir / "selected_feature_frequency.csv")
    ridge_stability = _read_csv_or_empty(cfg.output_dir / "ridge_coefficient_stability.csv")
    if "feature" not in selected_frequency.columns:
        selected_frequency = pd.DataFrame(columns=["feature"])
    if "feature" not in ridge_stability.columns:
        ridge_stability = pd.DataFrame(columns=["feature"])
    ridge_screen = factor_whitelist.merge(
        selected_frequency,
        on="feature",
        how="left",
    ).merge(
        ridge_stability,
        on="feature",
        how="left",
        suffixes=("", "_ridge"),
    )
    split_count = max(1, len(split_ids))
    ridge_screen["selection_rate"] = ridge_screen["selection_rate"].fillna(0.0)
    ridge_screen["pass_ridge_gate"] = (
        ridge_screen["selection_rate"] >= (
            cfg.train.factor_eval.ridge_min_selection_rate_multi_split
            if split_count > 1
            else cfg.train.factor_eval.ridge_min_selection_rate_single_split
        )
    ) & (
        ridge_screen["ridge_original_coef_cv"].fillna(0.0) <= cfg.train.factor_eval.ridge_max_original_coef_cv
    )
    if "ridge_original_abs_coef_mean" not in ridge_screen.columns:
        ridge_screen["ridge_original_abs_coef_mean"] = 0.0
    ridge_screen = ridge_screen.sort_values(
        ["pass_ridge_gate", "quality_score", "selection_rate", "ridge_original_abs_coef_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    ridge_screen.to_csv(cfg.output_dir / "ridge_screen.csv", index=False)

    lgbm_feature_pool = ridge_screen.loc[ridge_screen["pass_ridge_gate"]].copy()
    lgbm_feature_pool["final_status"] = "lgbm_candidate"
    lgbm_feature_pool.to_csv(cfg.output_dir / "lgbm_feature_pool.csv", index=False)


def run_single_factor_pipeline(
    cfg: AppConfig,
    research_start: str,
    research_end: str,
    bucket_count: int = 5,
    top_k: int = 20,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(
        cfg,
        cfg.output_dir / "run_config.json",
        extra={
            "command": "single-factor-run",
            "research_start": research_start,
            "research_end": research_end,
            "bucket_count": bucket_count,
            "top_k": top_k,
            "winsorize_lower_quantile": 0.01,
            "winsorize_upper_quantile": 0.99,
        },
    )

    panel = load_panel(
        cfg.data.input_path,
        index_code=cfg.data.index_code,
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date,
    )
    panel = add_features(panel)
    panel = add_labels(panel, cfg.labels.primary_horizon, cfg.labels.secondary_horizon)
    panel = apply_universe_filters(panel, cfg.data)
    panel = panel.loc[
        (panel["date"] >= pd.Timestamp(research_start)) & (panel["date"] <= pd.Timestamp(research_end))
    ].copy()

    features = cfg.train.features or FEATURE_COLUMNS
    target_col = f"future_return_{cfg.labels.primary_horizon}d"
    if panel.empty:
        raise ValueError("No data available for the requested single-factor window.")
    registry = build_factor_registry(features)
    registry.to_csv(cfg.output_dir / "factor_registry.csv", index=False)

    annual_rows: list[dict] = []
    years = sorted(panel["date"].dt.year.unique().tolist())
    for year in years:
        year_df = panel.loc[panel["date"].dt.year == year].copy()
        if year_df.empty:
            continue

        year_dir = cfg.output_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = summarize_feature_diagnostics(year_df, features, target_col)
        if diagnostics.empty:
            continue

        diagnostics["abs_mean_rank_ic"] = diagnostics["mean_rank_ic"].abs()
        diagnostics["direction"] = diagnostics["mean_rank_ic"].apply(lambda x: 1 if pd.isna(x) or x >= 0 else -1)
        diagnostics["sign_flipped"] = diagnostics["direction"].eq(-1)
        diagnostics = diagnostics.sort_values(
            ["abs_mean_rank_ic", "rank_ic_ir", "mean_top_bottom_spread"],
            ascending=False,
        ).reset_index(drop=True)
        diagnostics.to_csv(year_dir / "single_factor_summary.csv", index=False)

        selected = diagnostics.copy()
        if top_k > 0:
            selected = selected.head(top_k).copy()
        selected.to_csv(year_dir / "single_factor_selected.csv", index=False)

        year_report_rows: list[dict] = []
        for row in selected.to_dict(orient="records"):
            feature = row["feature"]
            direction = int(row["direction"])
            standardized_col = f"{feature}_sf_score"
            scored = winsorize_and_zscore_by_date(year_df, feature, standardized_col)
            if direction < 0:
                scored[standardized_col] = scored[standardized_col] * -1.0

            factor_dir = year_dir / feature
            factor_dir.mkdir(parents=True, exist_ok=True)

            ic_df, ic_summary, bucket_df = compute_signal_diagnostics(
                scored,
                score_col=standardized_col,
                target_col=target_col,
                bucket_count=bucket_count,
            )
            group_returns, group_metrics, group_summary = compute_single_factor_group_backtest(
                scored,
                score_col=standardized_col,
                target_col=target_col,
                bucket_count=bucket_count,
            )
            top_bucket = group_summary.get("top_bucket", f"Q{bucket_count}")
            top_bucket_portfolio = pd.DataFrame(
                {
                    "date": group_returns["date"],
                    "portfolio_ret": group_returns[top_bucket].fillna(0.0),
                }
            )
            top_bucket_portfolio["equity"] = (1.0 + top_bucket_portfolio["portfolio_ret"]).cumprod()
            official_benchmark = load_official_index_benchmark(split_cfg := replace(cfg, output_dir=factor_dir), top_bucket_portfolio["date"])
            _, top_bucket_excess_metrics = compare_to_official_index(top_bucket_portfolio, official_benchmark)

            ic_df.to_csv(factor_dir / "ic_timeseries.csv", index=False)
            bucket_df.to_csv(factor_dir / "bucket_returns.csv", index=False)
            group_returns.to_csv(factor_dir / "group_backtest_returns.csv", index=False)
            group_metrics.to_csv(factor_dir / "group_backtest_metrics.csv", index=False)
            save_json(ic_summary, factor_dir / "diagnostics.json")
            save_json(group_summary, factor_dir / "group_backtest_summary.json")
            save_json(top_bucket_excess_metrics, factor_dir / "top_bucket_official_index_excess.json")

            report_row = {
                "year": year,
                "feature": feature,
                "direction": direction,
                "sign_flipped": bool(row["sign_flipped"]),
                "mean_rank_ic": row["mean_rank_ic"],
                "rank_ic_ir": row["rank_ic_ir"],
                "positive_rank_ic_ratio": row["positive_rank_ic_ratio"],
                "mean_top_bottom_spread": row["mean_top_bottom_spread"],
                "long_short_annual_return_est": group_summary.get("long_short_annual_return_est"),
                "long_short_sharpe_est": group_summary.get("long_short_sharpe_est"),
                "top_bucket_annual_return_est": group_summary.get("top_bucket_annual_return_est"),
                "bottom_bucket_annual_return_est": group_summary.get("bottom_bucket_annual_return_est"),
                "top_bucket_official_index_excess_annual_return_est": top_bucket_excess_metrics.get("annual_return_est"),
                "top_bucket_official_index_excess_sharpe_est": top_bucket_excess_metrics.get("sharpe_est"),
                "bucket_return_spearman": group_summary.get("bucket_return_spearman"),
                "bucket_return_monotonic_increasing": group_summary.get("bucket_return_monotonic_increasing"),
                "bucket_return_monotonic_decreasing": group_summary.get("bucket_return_monotonic_decreasing"),
            }
            year_report_rows.append(report_row)
            annual_rows.append(report_row)

        pd.DataFrame(year_report_rows).sort_values(
            ["mean_rank_ic", "long_short_annual_return_est"],
            ascending=False,
        ).to_csv(year_dir / "single_factor_report.csv", index=False)

    annual_report = pd.DataFrame(annual_rows)
    if annual_report.empty:
        raise ValueError("No annual single-factor reports were generated for the requested window.")

    annual_report.to_csv(cfg.output_dir / "single_factor_annual_report.csv", index=False)
    evaluation, whitelist = _build_factor_evaluation(annual_report=annual_report, registry=registry, cfg=cfg)
    evaluation.to_csv(cfg.output_dir / "factor_evaluation.csv", index=False)
    whitelist.to_csv(cfg.output_dir / "factor_whitelist.csv", index=False)
    evaluation.to_csv(cfg.output_dir / "single_factor_stability.csv", index=False)


def run_family_lab_pipeline(
    cfg: AppConfig,
    family: str,
    research_start: str,
    research_end: str,
    bucket_count: int = 5,
) -> None:
    family_key = family.lower()
    family_features_map = {
        "sma": [
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
            "trend_5_13",
            "trend_5_20",
            "trend_8_16",
            "trend_8_21",
            "trend_13_34",
            "trend_16_48",
            "trend_20_60",
            "trend_21_55",
        ],
        "reversal": [
            "ret_1",
            "ret_3",
            "ret_5",
            "ret_10",
            "ret_20",
            "ret_60",
            "ret_reversal_5_20",
            "ret_reversal_10_60",
            "gap_1",
            "intraday_ret_1",
            "overnight_ret_5",
            "price_volume_div_5",
            "price_volume_div_20",
            "rebound_from_low_20",
        ],
    }
    if family_key not in family_features_map:
        raise ValueError(f"Unsupported factor family: {family}")

    cfg.train.features = family_features_map[family_key]
    run_single_factor_pipeline(
        cfg,
        research_start=research_start,
        research_end=research_end,
        bucket_count=bucket_count,
        top_k=0,
    )

    stability_path = cfg.output_dir / "single_factor_stability.csv"
    if not stability_path.exists():
        raise ValueError("single_factor_stability.csv was not generated by family lab.")

    stability = pd.read_csv(stability_path)
    registry = stability.copy()
    registry["family"] = family_key
    registry["status"] = "discard"
    registry["reason"] = "weak_recent_or_unstable"

    keep_mask = (
        registry["positive_top_bucket_excess_year_ratio"].fillna(0.0) >= 2.0 / 3.0
    ) & (
        registry["latest_top_bucket_official_index_excess_annual_return_est"].fillna(-1.0) > 0.0
    ) & (
        registry["latest_bucket_return_spearman"].fillna(0.0) >= 0.5
    ) & (
        registry["years_covered"].fillna(0) >= 4
    )
    recent_only_mask = (~keep_mask) & (
        registry["latest_top_bucket_official_index_excess_annual_return_est"].fillna(-1.0) > 0.0
    ) & (
        registry["latest_bucket_return_spearman"].fillna(0.0) >= 0.5
    )

    registry.loc[keep_mask, "status"] = "keep"
    registry.loc[keep_mask, "reason"] = "stable_excess_and_recent_effective"
    registry.loc[recent_only_mask, "status"] = "recent_only"
    registry.loc[recent_only_mask, "reason"] = "recent_effective_but_long_term_unstable"
    registry.loc[
        registry["latest_top_bucket_official_index_excess_annual_return_est"].fillna(-1.0) <= 0.0,
        "reason",
    ] = "recently_ineffective"
    registry.loc[
        registry["latest_bucket_return_spearman"].fillna(0.0) < 0.5,
        "reason",
    ] = "weak_bucket_ordering"

    registry = registry[[
        "family",
        "feature",
        "status",
        "reason",
        "years_covered",
        "mean_rank_ic_mean",
        "rank_ic_ir_mean",
        "top_bucket_official_index_excess_annual_return_est_mean",
        "positive_top_bucket_excess_year_ratio",
        "bucket_return_spearman_mean",
        "monotonic_year_ratio",
        "latest_mean_rank_ic",
        "latest_top_bucket_official_index_excess_annual_return_est",
        "latest_bucket_return_spearman",
    ]].sort_values(
        ["status", "latest_top_bucket_official_index_excess_annual_return_est", "bucket_return_spearman_mean"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    registry.to_csv(cfg.output_dir / "factor_family_registry.csv", index=False)


def ensure_default_dirs() -> None:
    Path("data").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
