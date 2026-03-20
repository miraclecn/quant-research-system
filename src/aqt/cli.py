from __future__ import annotations

import argparse
from pathlib import Path

from aqt.config import AppConfig
from aqt.data import export_panel
from aqt.pipeline import ensure_default_dirs, run_factor_chain_pipeline, run_family_lab_pipeline, run_pipeline, run_research_pipeline, run_single_factor_pipeline
from aqt.update import (
    build_prune_plan,
    execute_prune_plan,
    format_fina_indicator_rebuild_summary,
    format_index_daily_update_summary,
    format_index_weight_update_summary,
    format_prune_plan,
    format_update_summary,
    rebuild_fina_indicator,
    update_index_daily,
    update_index_weight,
    update_raw,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="A-share daily research starter")
    parser.add_argument(
        "command",
        choices=["run", "research-run", "factor-chain-run", "single-factor-run", "family-lab", "export-panel", "update-raw", "update-index-weight", "update-index-daily", "rebuild-fina-indicator", "prune-db", "daily-research"],
        help="Workflow command",
    )
    parser.add_argument("--input", dest="input_path", help="Input panel path")
    parser.add_argument("--output-dir", dest="output_dir", help="Output directory")
    parser.add_argument("--output", dest="output_path", help="Output panel path for export-panel")
    parser.add_argument("--execute", action="store_true", help="Apply destructive actions for prune-db")
    parser.add_argument("--lookback-days", type=int, default=7, help="Refresh overlap window for update-raw")
    parser.add_argument("--tushare-token", dest="tushare_token", help="Tushare token for update-raw")
    parser.add_argument("--limit-stocks", type=int, help="Limit stock count for full-table rebuild smoke tests")
    parser.add_argument("--index-code", dest="index_code", help="Index code for constituent membership, e.g. 000852.SH")
    parser.add_argument("--start-date", dest="start_date", help="Inclusive start date, e.g. 2021-01-01")
    parser.add_argument("--end-date", dest="end_date", help="Inclusive end date, e.g. 2025-12-31")
    parser.add_argument("--train-days", type=int, help="Rolling training window in trading days")
    parser.add_argument("--research-start", dest="research_start", help="Rolling research window start date for research-run")
    parser.add_argument("--research-end", dest="research_end", help="Rolling research window end date for research-run")
    parser.add_argument("--train-months", dest="train_months", type=int, help="Training window size in months for research-run")
    parser.add_argument("--valid-months", dest="valid_months", type=int, help="Validation window size in months for research-run")
    parser.add_argument("--test-months", dest="test_months", type=int, help="Test window size in months for research-run")
    parser.add_argument("--step-months", dest="step_months", type=int, help="Rolling step size in months for research-run; defaults to valid-months")
    parser.add_argument("--top-n", type=int, help="Number of names to hold at each rebalance")
    parser.add_argument("--rebalance-weekday", type=int, choices=range(5), help="0=Mon, 4=Fri")
    parser.add_argument("--lgbm-objective", choices=["lambdarank", "regression"], help="LightGBM objective; default is lambdarank")
    parser.add_argument("--lgbm-n-estimators", type=int, help="LightGBM number of boosting rounds")
    parser.add_argument("--lgbm-learning-rate", type=float, help="LightGBM learning rate")
    parser.add_argument("--lgbm-num-leaves", type=int, help="LightGBM num_leaves")
    parser.add_argument("--lgbm-max-depth", type=int, help="LightGBM max_depth; use -1 for no limit")
    parser.add_argument("--lgbm-min-child-samples", type=int, help="LightGBM min_child_samples")
    parser.add_argument("--lgbm-subsample", type=float, help="LightGBM row subsample ratio")
    parser.add_argument("--lgbm-colsample-bytree", type=float, help="LightGBM feature subsample ratio")
    parser.add_argument("--lgbm-reg-alpha", type=float, help="LightGBM L1 regularization")
    parser.add_argument("--lgbm-reg-lambda", type=float, help="LightGBM L2 regularization")
    parser.add_argument("--lgbm-rank-bins", type=int, help="Number of within-date relevance bins for lambdarank")
    parser.add_argument("--factor-top-k", type=int, default=10, help="Number of features kept after train-period single-factor screening")
    parser.add_argument("--factor-min-rank-ic", type=float, default=0.0, help="Minimum absolute train-period mean_rank_ic for feature selection")
    parser.add_argument("--factor-max-corr", type=float, default=0.9, help="Maximum absolute train-period pairwise correlation allowed among selected features")
    parser.add_argument("--factor-min-abs-rank-ic-ir", type=float, help="Minimum absolute mean rank ICIR for passing the single-factor gate")
    parser.add_argument("--factor-min-bucket-spearman", type=float, help="Minimum mean bucket Spearman for passing the single-factor gate")
    parser.add_argument("--factor-min-positive-top-bucket-excess-ratio", type=float, help="Minimum positive-year ratio of top-bucket benchmark excess")
    parser.add_argument("--factor-min-latest-top-bucket-excess", type=float, help="Minimum latest-year top-bucket official index excess annual return")
    parser.add_argument("--factor-core-quantile", type=float, help="Quantile threshold for tagging factors as core")
    parser.add_argument("--factor-candidate-quantile", type=float, help="Quantile threshold for tagging factors as candidate")
    parser.add_argument("--factor-fallback-top-n", type=int, help="Fallback number of top quality-score factors used when whitelist is empty")
    parser.add_argument("--ridge-min-selection-rate-multi-split", type=float, help="Minimum Ridge selection rate when there are multiple rolling splits")
    parser.add_argument("--ridge-min-selection-rate-single-split", type=float, help="Minimum Ridge selection rate when there is only one split")
    parser.add_argument("--ridge-max-original-coef-cv", type=float, help="Maximum Ridge original coefficient CV for passing the Ridge gate")
    parser.add_argument("--bucket-count", type=int, default=5, choices=[5, 10], help="Bucket count for single-factor group analysis")
    parser.add_argument("--single-factor-top-k", type=int, default=20, help="Number of top-ranked factors to export detailed single-factor reports for")
    parser.add_argument("--family", choices=["sma", "reversal"], help="Factor family name for family-lab")
    parser.add_argument("--neutralize-scores", action="store_true", help="Run additional industry and size neutralized score diagnostics")
    parser.add_argument("--no-neutralize-industry", action="store_true", help="Disable industry neutralization when --neutralize-scores is enabled")
    parser.add_argument("--no-neutralize-size", action="store_true", help="Disable size neutralization when --neutralize-scores is enabled")
    args = parser.parse_args()

    ensure_default_dirs()
    cfg = AppConfig()

    if args.input_path:
        cfg.data.input_path = Path(args.input_path)
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    if args.start_date:
        cfg.data.start_date = args.start_date
    if args.end_date:
        cfg.data.end_date = args.end_date
    if args.index_code:
        cfg.data.index_code = args.index_code
    if args.train_days is not None:
        cfg.train.train_days = args.train_days
    if args.top_n is not None:
        cfg.portfolio.top_n = args.top_n
    if args.rebalance_weekday is not None:
        cfg.portfolio.rebalance_weekday = args.rebalance_weekday
    if args.lgbm_objective:
        cfg.train.lgbm.objective = args.lgbm_objective
    if args.lgbm_n_estimators is not None:
        cfg.train.lgbm.n_estimators = args.lgbm_n_estimators
    if args.lgbm_learning_rate is not None:
        cfg.train.lgbm.learning_rate = args.lgbm_learning_rate
    if args.lgbm_num_leaves is not None:
        cfg.train.lgbm.num_leaves = args.lgbm_num_leaves
    if args.lgbm_max_depth is not None:
        cfg.train.lgbm.max_depth = args.lgbm_max_depth
    if args.lgbm_min_child_samples is not None:
        cfg.train.lgbm.min_child_samples = args.lgbm_min_child_samples
    if args.lgbm_subsample is not None:
        cfg.train.lgbm.subsample = args.lgbm_subsample
    if args.lgbm_colsample_bytree is not None:
        cfg.train.lgbm.colsample_bytree = args.lgbm_colsample_bytree
    if args.lgbm_reg_alpha is not None:
        cfg.train.lgbm.reg_alpha = args.lgbm_reg_alpha
    if args.lgbm_reg_lambda is not None:
        cfg.train.lgbm.reg_lambda = args.lgbm_reg_lambda
    if args.lgbm_rank_bins is not None:
        cfg.train.lgbm.rank_bins = args.lgbm_rank_bins
    if args.factor_min_abs_rank_ic_ir is not None:
        cfg.train.factor_eval.min_abs_rank_ic_ir = args.factor_min_abs_rank_ic_ir
    if args.factor_min_bucket_spearman is not None:
        cfg.train.factor_eval.min_bucket_return_spearman = args.factor_min_bucket_spearman
    if args.factor_min_positive_top_bucket_excess_ratio is not None:
        cfg.train.factor_eval.min_positive_top_bucket_excess_year_ratio = args.factor_min_positive_top_bucket_excess_ratio
    if args.factor_min_latest_top_bucket_excess is not None:
        cfg.train.factor_eval.min_latest_top_bucket_excess = args.factor_min_latest_top_bucket_excess
    if args.factor_core_quantile is not None:
        cfg.train.factor_eval.core_quantile = args.factor_core_quantile
    if args.factor_candidate_quantile is not None:
        cfg.train.factor_eval.candidate_quantile = args.factor_candidate_quantile
    if args.factor_fallback_top_n is not None:
        cfg.train.factor_eval.fallback_top_n = args.factor_fallback_top_n
    if args.ridge_min_selection_rate_multi_split is not None:
        cfg.train.factor_eval.ridge_min_selection_rate_multi_split = args.ridge_min_selection_rate_multi_split
    if args.ridge_min_selection_rate_single_split is not None:
        cfg.train.factor_eval.ridge_min_selection_rate_single_split = args.ridge_min_selection_rate_single_split
    if args.ridge_max_original_coef_cv is not None:
        cfg.train.factor_eval.ridge_max_original_coef_cv = args.ridge_max_original_coef_cv
    if args.neutralize_scores:
        cfg.train.neutralize_scores = True
    if args.no_neutralize_industry:
        cfg.train.neutralize_industry = False
    if args.no_neutralize_size:
        cfg.train.neutralize_size = False

    if args.command == "run":
        run_pipeline(cfg)
    elif args.command == "research-run":
        missing = [name for name, value in [
            ("research_start", args.research_start),
            ("research_end", args.research_end),
            ("train_months", args.train_months),
            ("valid_months", args.valid_months),
            ("test_months", args.test_months),
        ] if not value]
        if missing:
            raise ValueError(f"research-run requires: {', '.join(missing)}")
        run_research_pipeline(
            cfg,
            research_start=args.research_start,
            research_end=args.research_end,
            train_months=args.train_months,
            valid_months=args.valid_months,
            test_months=args.test_months,
            step_months=args.step_months,
        )
    elif args.command == "factor-chain-run":
        missing = [name for name, value in [
            ("research_start", args.research_start),
            ("research_end", args.research_end),
            ("train_months", args.train_months),
            ("valid_months", args.valid_months),
            ("test_months", args.test_months),
        ] if not value]
        if missing:
            raise ValueError(f"factor-chain-run requires: {', '.join(missing)}")
        run_factor_chain_pipeline(
            cfg,
            research_start=args.research_start,
            research_end=args.research_end,
            train_months=args.train_months,
            valid_months=args.valid_months,
            test_months=args.test_months,
            step_months=args.step_months,
            factor_top_k=args.factor_top_k,
            factor_min_rank_ic=args.factor_min_rank_ic,
            factor_max_corr=args.factor_max_corr,
        )
    elif args.command == "single-factor-run":
        missing = [name for name, value in [
            ("research_start", args.research_start),
            ("research_end", args.research_end),
        ] if not value]
        if missing:
            raise ValueError(f"single-factor-run requires: {', '.join(missing)}")
        run_single_factor_pipeline(
            cfg,
            research_start=args.research_start,
            research_end=args.research_end,
            bucket_count=args.bucket_count,
            top_k=args.single_factor_top_k,
        )
    elif args.command == "family-lab":
        missing = [name for name, value in [
            ("family", args.family),
            ("research_start", args.research_start),
            ("research_end", args.research_end),
        ] if not value]
        if missing:
            raise ValueError(f"family-lab requires: {', '.join(missing)}")
        run_family_lab_pipeline(
            cfg,
            family=args.family,
            research_start=args.research_start,
            research_end=args.research_end,
            bucket_count=args.bucket_count,
        )
    elif args.command == "export-panel":
        output_path = Path(args.output_path) if args.output_path else Path("data/daily_bars.parquet")
        input_path = cfg.data.input_path
        export_panel(
            input_path=input_path,
            index_code=cfg.data.index_code,
            output_path=output_path,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
        )
        print(output_path)
    elif args.command == "update-raw":
        db_path = cfg.data.input_path if cfg.data.input_path.suffix == ".duckdb" else Path("stock_data.duckdb")
        summary = update_raw(
            db_path=db_path,
            tushare_token=args.tushare_token,
            index_code=cfg.data.index_code,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
            lookback_days=args.lookback_days,
        )
        print(format_update_summary(summary))
    elif args.command == "prune-db":
        db_path = cfg.data.input_path if cfg.data.input_path.suffix == ".duckdb" else Path("stock_data.duckdb")
        plan = execute_prune_plan(db_path=db_path) if args.execute else build_prune_plan(db_path=db_path)
        print(format_prune_plan(plan))
    elif args.command == "rebuild-fina-indicator":
        db_path = cfg.data.input_path if cfg.data.input_path.suffix == ".duckdb" else Path("stock_data.duckdb")
        summary = rebuild_fina_indicator(
            db_path=db_path,
            tushare_token=args.tushare_token,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
            limit_stocks=args.limit_stocks,
        )
        print(format_fina_indicator_rebuild_summary(summary))
    elif args.command == "update-index-weight":
        db_path = cfg.data.input_path if cfg.data.input_path.suffix == ".duckdb" else Path("stock_data.duckdb")
        summary = update_index_weight(
            db_path=db_path,
            tushare_token=args.tushare_token,
            index_code=cfg.data.index_code,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
        )
        print(format_index_weight_update_summary(summary))
    elif args.command == "daily-research":
        db_path = cfg.data.input_path if cfg.data.input_path.suffix == ".duckdb" else Path("stock_data.duckdb")
        panel_path = Path(args.output_path) if args.output_path else Path("data/daily_bars.parquet")
        update_summary = update_raw(
            db_path=db_path,
            tushare_token=args.tushare_token,
            index_code=cfg.data.index_code,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
            lookback_days=args.lookback_days,
        )
        print(format_update_summary(update_summary))
        index_weight_summary = update_index_weight(
            db_path=db_path,
            tushare_token=args.tushare_token,
            index_code=cfg.data.index_code,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
        )
        print(format_index_weight_update_summary(index_weight_summary))
        index_daily_summary = update_index_daily(
            db_path=db_path,
            tushare_token=args.tushare_token,
            index_code=cfg.data.index_code,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
        )
        print(format_index_daily_update_summary(index_daily_summary))
        export_panel(
            input_path=db_path,
            index_code=cfg.data.index_code,
            output_path=panel_path,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
        )
        print(panel_path)
        cfg.data.input_path = panel_path
        run_pipeline(cfg)
    elif args.command == "update-index-daily":
        db_path = cfg.data.input_path if cfg.data.input_path.suffix == ".duckdb" else Path("stock_data.duckdb")
        summary = update_index_daily(
            db_path=db_path,
            tushare_token=args.tushare_token,
            index_code=cfg.data.index_code,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
        )
        print(format_index_daily_update_summary(summary))


if __name__ == "__main__":
    main()
