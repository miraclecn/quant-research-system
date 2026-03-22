from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from aqt.config import LightGBMConfig


@dataclass(slots=True)
class ModelOutputs:
    ridge_pred: np.ndarray
    lgbm_pred: np.ndarray
    ridge_coef: np.ndarray
    lgbm_feature_importance: np.ndarray


@dataclass(slots=True)
class LGBMOutputs:
    pred: np.ndarray
    feature_importance: np.ndarray
    best_iteration: int | None
    best_score: dict[str, dict[str, float]]


def fit_predict_ridge(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    target_col: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = train_df[features].fillna(0.0)
    y_train = train_df[target_col]
    x_test = test_df[features].fillna(0.0)

    ridge = Ridge(alpha=1.0, random_state=random_state)
    ridge.fit(x_train, y_train)
    ridge_pred = ridge.predict(x_test)
    ridge_coef = np.asarray(ridge.coef_, dtype=float)
    return ridge_pred, ridge_coef


def fit_predict_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    target_col: str,
    random_state: int,
    lgbm_cfg: LightGBMConfig,
    valid_df: pd.DataFrame | None = None,
) -> ModelOutputs:
    ridge_pred, ridge_coef = fit_predict_ridge(
        train_df=train_df,
        test_df=test_df,
        features=features,
        target_col=target_col,
        random_state=random_state,
    )

    lgbm_outputs = fit_predict_lgbm(
        train_df=train_df,
        test_df=test_df,
        features=features,
        target_col=target_col,
        random_state=random_state,
        lgbm_cfg=lgbm_cfg,
        valid_df=valid_df,
    )

    return ModelOutputs(
        ridge_pred=ridge_pred,
        lgbm_pred=lgbm_outputs.pred,
        ridge_coef=ridge_coef,
        lgbm_feature_importance=lgbm_outputs.feature_importance,
    )


def fit_predict_lgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    target_col: str,
    random_state: int,
    lgbm_cfg: LightGBMConfig,
    valid_df: pd.DataFrame | None = None,
) -> LGBMOutputs:
    x_train = train_df[features].fillna(0.0)
    y_train = train_df[target_col]
    x_test = test_df[features].fillna(0.0)

    rank_metric_spec = (
        ["ndcg@20", "ndcg@50"]
        if lgbm_cfg.metric in (None, "ndcg@20")
        else lgbm_cfg.metric
    )
    if not isinstance(rank_metric_spec, list):
        rank_metric_spec = [rank_metric_spec]
    rank_eval_at = tuple(
        int(str(metric).split("@", 1)[1])
        for metric in rank_metric_spec
        if isinstance(metric, str) and metric.startswith("ndcg@")
    )

    lgbm_params = {
        "objective": lgbm_cfg.objective,
        **(
            {
                "metric": "ndcg",
                "lambdarank_truncation_level": lgbm_cfg.lambdarank_truncation_level,
            }
            if lgbm_cfg.objective == "lambdarank"
            else {}
        ),
        "learning_rate": lgbm_cfg.learning_rate,
        "num_leaves": lgbm_cfg.num_leaves,
        "max_depth": lgbm_cfg.max_depth,
        "min_child_samples": lgbm_cfg.min_child_samples,
        "subsample": lgbm_cfg.subsample,
        "colsample_bytree": lgbm_cfg.colsample_bytree,
        "reg_alpha": lgbm_cfg.reg_alpha,
        "reg_lambda": lgbm_cfg.reg_lambda,
        "random_state": random_state,
        "n_jobs": -1,
        "force_col_wise": lgbm_cfg.force_col_wise,
        "verbosity": lgbm_cfg.verbosity,
        "device": lgbm_cfg.device,
    }

    min_boost_round = max(0, int(lgbm_cfg.min_boost_round))
    total_boost_round = max(1, int(lgbm_cfg.n_estimators))
    warmup_boost_round = min(min_boost_round, total_boost_round)
    remaining_boost_round = max(0, total_boost_round - warmup_boost_round)

    if lgbm_cfg.objective == "lambdarank":
        rank_train_df = train_df.sort_values(["date", "symbol"]).reset_index(drop=True)
        x_train = rank_train_df[features].fillna(0.0)
        rank_labels, rank_group = _build_lambdarank_labels(rank_train_df, target_col, lgbm_cfg.rank_bins)
        fit_metric = "ndcg" if rank_eval_at else rank_metric_spec
        has_valid = valid_df is not None and not valid_df.empty
        if has_valid:
            rank_valid_df = valid_df.sort_values(["date", "symbol"]).reset_index(drop=True)
            x_valid = rank_valid_df[features].fillna(0.0)
            valid_labels, valid_group = _build_lambdarank_labels(rank_valid_df, target_col, lgbm_cfg.rank_bins)
        if not has_valid:
            lgbm = lgb.LGBMRanker(**{**lgbm_params, "n_estimators": total_boost_round})
            lgbm.fit(x_train, rank_labels, group=rank_group)
        else:
            warmup_params = {**lgbm_params, "n_estimators": warmup_boost_round}
            lgbm = lgb.LGBMRanker(**warmup_params)
            lgbm.fit(x_train, rank_labels, group=rank_group)
        if has_valid and remaining_boost_round > 0:
            fit_kwargs = {
                "eval_set": [(x_valid, valid_labels)],
                "eval_group": [valid_group],
                "eval_metric": fit_metric,
                **({"eval_at": rank_eval_at} if rank_eval_at else {}),
                "callbacks": [lgb.early_stopping(lgbm_cfg.early_stopping_rounds, verbose=False)],
                "init_model": lgbm.booster_,
            }
            lgbm = lgb.LGBMRanker(**{**lgbm_params, "n_estimators": remaining_boost_round})
            lgbm.fit(x_train, rank_labels, group=rank_group, **fit_kwargs)
    else:
        has_valid = valid_df is not None and not valid_df.empty
        if has_valid:
            x_valid = valid_df[features].fillna(0.0)
            y_valid = valid_df[target_col]
        if not has_valid:
            lgbm = lgb.LGBMRegressor(**{**lgbm_params, "n_estimators": total_boost_round})
            lgbm.fit(x_train, y_train)
        else:
            warmup_params = {**lgbm_params, "n_estimators": warmup_boost_round}
            lgbm = lgb.LGBMRegressor(**warmup_params)
            lgbm.fit(x_train, y_train)
        if has_valid and remaining_boost_round > 0:
            fit_kwargs = {
                "eval_set": [(x_valid, y_valid)],
                "eval_metric": lgbm_cfg.metric,
                "callbacks": [lgb.early_stopping(lgbm_cfg.early_stopping_rounds, verbose=False)],
                "init_model": lgbm.booster_,
            }
            lgbm = lgb.LGBMRegressor(**{**lgbm_params, "n_estimators": remaining_boost_round})
            lgbm.fit(x_train, y_train, **fit_kwargs)
    best_iteration = getattr(lgbm, "best_iteration_", None)
    lgbm_pred = lgbm.predict(x_test, num_iteration=best_iteration if best_iteration and best_iteration > 0 else None)

    return LGBMOutputs(
        pred=lgbm_pred,
        feature_importance=np.asarray(lgbm.feature_importances_, dtype=float),
        best_iteration=best_iteration,
        best_score=getattr(lgbm, "best_score_", {}) or {},
    )


def _build_lambdarank_labels(
    train_df: pd.DataFrame,
    target_col: str,
    rank_bins: int,
) -> tuple[np.ndarray, list[int]]:
    def to_relevance(group: pd.Series) -> pd.Series:
        non_null = group.dropna()
        if non_null.empty:
            return pd.Series(np.zeros(len(group), dtype=np.int32), index=group.index)

        bins = max(2, min(rank_bins, int(non_null.nunique())))
        if bins <= 1:
            return pd.Series(np.zeros(len(group), dtype=np.int32), index=group.index)

        ranked = non_null.rank(method="first")
        labels = pd.qcut(ranked, q=bins, labels=False, duplicates="drop")
        out = pd.Series(np.zeros(len(group), dtype=np.int32), index=group.index)
        out.loc[non_null.index] = labels.astype(np.int32)
        return out

    relevance = (
        train_df.groupby("date", sort=True)[target_col]
        .transform(to_relevance)
        .astype(np.int32)
        .to_numpy()
    )
    group_sizes = train_df.groupby("date", sort=True).size().astype(int).tolist()
    return relevance, group_sizes
