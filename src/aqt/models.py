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


def fit_predict_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    target_col: str,
    random_state: int,
    lgbm_cfg: LightGBMConfig,
) -> ModelOutputs:
    x_train = train_df[features].fillna(0.0)
    y_train = train_df[target_col]
    x_test = test_df[features].fillna(0.0)

    ridge = Ridge(alpha=1.0, random_state=random_state)
    ridge.fit(x_train, y_train)
    ridge_pred = ridge.predict(x_test)

    lgbm_params = {
        "objective": lgbm_cfg.objective,
        "n_estimators": lgbm_cfg.n_estimators,
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
    }

    if lgbm_cfg.objective == "lambdarank":
        rank_train_df = train_df.sort_values(["date", "symbol"]).reset_index(drop=True)
        x_train = rank_train_df[features].fillna(0.0)
        rank_labels, rank_group = _build_lambdarank_labels(rank_train_df, target_col, lgbm_cfg.rank_bins)
        lgbm = lgb.LGBMRanker(**lgbm_params)
        lgbm.fit(x_train, rank_labels, group=rank_group)
    else:
        lgbm = lgb.LGBMRegressor(**lgbm_params)
        lgbm.fit(x_train, y_train)
    lgbm_pred = lgbm.predict(x_test)

    return ModelOutputs(
        ridge_pred=ridge_pred,
        lgbm_pred=lgbm_pred,
        ridge_coef=np.asarray(ridge.coef_, dtype=float),
        lgbm_feature_importance=np.asarray(lgbm.feature_importances_, dtype=float),
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
