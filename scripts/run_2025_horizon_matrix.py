from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from aqt.backtest import select_rebalance_dates
from aqt.config import AppConfig
from aqt.data import load_panel
from aqt.features import add_features
from aqt.labels import add_labels
from aqt.models import fit_predict_models
from aqt.pipeline import _run_strategy
from aqt.research import save_json, summarize_feature_importance
from aqt.universe import apply_universe_filters


def _load_feature_pool(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        raise ValueError(f"feature column not found in {path}")
    features = [str(feature) for feature in df["feature"].dropna().tolist()]
    if not features:
        raise ValueError(f"no features found in {path}")
    return features


def _select_eval_rebalance_dates(
    modeling_dates: list[pd.Timestamp],
    weekday: int,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    rebalance_every_n_weeks: int,
) -> list[pd.Timestamp]:
    rebalance_dates = select_rebalance_dates(pd.Series(modeling_dates), weekday)
    rebalance_dates = pd.to_datetime(rebalance_dates)
    rebalance_dates = rebalance_dates[
        (rebalance_dates >= eval_start) & (rebalance_dates <= eval_end)
    ].reset_index(drop=True)
    step = max(1, int(rebalance_every_n_weeks))
    return rebalance_dates.iloc[::step].tolist()


def run_matrix(
    input_path: Path,
    whitelist_path: Path,
    output_dir: Path,
    start_date: str,
    end_date: str,
    eval_start: str,
    eval_end: str,
    cfg: AppConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or AppConfig()
    cfg.data.input_path = input_path
    features = _load_feature_pool(whitelist_path)

    panel = load_panel(
        input_path,
        index_code=cfg.data.index_code,
        start_date=start_date,
        end_date=end_date,
    )
    panel = add_features(panel, features)

    experiments = [
        {"name": "10d_weekly", "horizon": 10, "rebalance_every_n_weeks": 1},
        {"name": "20d_weekly", "horizon": 20, "rebalance_every_n_weeks": 1},
        {"name": "20d_biweekly", "horizon": 20, "rebalance_every_n_weeks": 2},
    ]

    results: list[dict] = []
    eval_start_ts = pd.Timestamp(eval_start)
    eval_end_ts = pd.Timestamp(eval_end)
    cache: dict[int, tuple[pd.DataFrame, list[np.ndarray], list[np.ndarray]]] = {}

    for experiment in experiments:
        horizon = int(experiment["horizon"])
        rebalance_every_n_weeks = int(experiment["rebalance_every_n_weeks"])
        experiment_dir = output_dir / experiment["name"]
        experiment_dir.mkdir(parents=True, exist_ok=True)

        labeled = add_labels(panel, primary_horizon=horizon, secondary_horizon=5)
        labeled = apply_universe_filters(labeled, cfg.data)

        target_col = f"future_return_{horizon}d"
        modeling = labeled.loc[labeled["tradable_universe"]].copy()
        modeling = modeling[
            ["date", "symbol", "tradable_universe", target_col, *features]
        ].dropna(subset=[target_col]).reset_index(drop=True)

        merge_source = labeled[
            [
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
        ].copy()

        unique_dates = sorted(modeling["date"].drop_duplicates())
        date_to_index = {date: idx for idx, date in enumerate(unique_dates)}
        rebalance_dates = _select_eval_rebalance_dates(
            modeling_dates=unique_dates,
            weekday=cfg.portfolio.rebalance_weekday,
            eval_start=eval_start_ts,
            eval_end=eval_end_ts,
            rebalance_every_n_weeks=rebalance_every_n_weeks,
        )

        cached = cache.get(horizon)
        ridge_importances: list = []
        lgbm_importances: list = []
        pred_df: pd.DataFrame
        if rebalance_every_n_weeks > 1 and cached is not None:
            pred_df, ridge_importances, lgbm_importances = cached
            pred_df = pred_df.loc[pred_df["date"].isin(rebalance_dates)].copy()
        else:
            predictions: list[pd.DataFrame] = []
            for rebalance_date in rebalance_dates:
                split_end = date_to_index.get(rebalance_date)
                if split_end is None or split_end < cfg.train.train_days:
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
                fold_pred["ridge_score"] = outputs.ridge_pred.astype("float32")
                fold_pred["lgbm_score"] = outputs.lgbm_pred.astype("float32")
                predictions.append(fold_pred)
                ridge_importances.append(outputs.ridge_coef)
                lgbm_importances.append(outputs.lgbm_feature_importance)

            if not predictions:
                raise ValueError(f"no prediction folds generated for {experiment['name']}")

            pred_df = pd.concat(predictions, ignore_index=True)
            pred_df = pred_df.drop_duplicates(subset=["date", "symbol"], keep="last")
            if rebalance_every_n_weeks == 1:
                cache[horizon] = (pred_df.copy(), ridge_importances.copy(), lgbm_importances.copy())
        merged = merge_source.merge(pred_df, on=["date", "symbol", "tradable_universe"], how="inner")

        experiment_cfg = replace(cfg, output_dir=experiment_dir)
        metrics = {
            "ridge": _run_strategy(merged, target_col, "ridge_score", experiment_cfg, "ridge"),
            "lgbm": _run_strategy(merged, target_col, "lgbm_score", experiment_cfg, "lgbm", write_default_alias=True),
        }
        save_json(metrics, experiment_dir / "metrics.json")

        feature_importance = summarize_feature_importance(features, ridge_importances, lgbm_importances)
        feature_importance.to_csv(experiment_dir / "feature_importance_summary.csv", index=False)

        results.append(
            {
                "experiment": experiment["name"],
                "horizon": horizon,
                "rebalance_every_n_weeks": rebalance_every_n_weeks,
                "prediction_dates": int(pred_df["date"].nunique()),
                "selected_feature_count": len(features),
                "ridge_annual_return_est": metrics["ridge"].get("annual_return_est"),
                "ridge_sharpe_est": metrics["ridge"].get("sharpe_est"),
                "ridge_max_drawdown": metrics["ridge"].get("max_drawdown"),
                "lgbm_annual_return_est": metrics["lgbm"].get("annual_return_est"),
                "lgbm_sharpe_est": metrics["lgbm"].get("sharpe_est"),
                "lgbm_max_drawdown": metrics["lgbm"].get("max_drawdown"),
                "lgbm_excess_annual_return_est": metrics["lgbm"].get("excess", {}).get("annual_return_est"),
                "lgbm_official_index_excess_annual_return_est": metrics["lgbm"].get("official_index_excess", {}).get("annual_return_est"),
            }
        )

    summary = pd.DataFrame(results).sort_values("experiment").reset_index(drop=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "run_context.json").write_text(
        json.dumps(
            {
                "input_path": str(input_path),
                "whitelist_path": str(whitelist_path),
                "features": features,
                "start_date": start_date,
                "end_date": end_date,
                "eval_start": eval_start,
                "eval_end": eval_end,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2025 horizon/rebalance comparison from a fixed factor whitelist.")
    parser.add_argument("--input", default="data/daily_bars.parquet")
    parser.add_argument("--whitelist", default="outputs/single-factor-2024-formal/factor_whitelist.csv")
    parser.add_argument("--output-dir", default="outputs/2025-horizon-matrix")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--eval-start", default="2025-01-01")
    parser.add_argument("--eval-end", default="2025-12-31")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU device for LightGBM training")
    args = parser.parse_args()

    cfg = AppConfig()
    if args.use_gpu:
        cfg.train.lgbm.device = "gpu"
        cfg.train.lgbm.force_col_wise = False

    summary = run_matrix(
        input_path=Path(args.input),
        whitelist_path=Path(args.whitelist),
        output_dir=Path(args.output_dir),
        start_date=args.start_date,
        end_date=args.end_date,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        cfg=cfg,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
