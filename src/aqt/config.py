from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class LightGBMConfig:
    objective: str = "lambdarank"
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.5
    reg_lambda: float = 1.0
    rank_bins: int = 20
    force_col_wise: bool = True
    verbosity: int = -1


@dataclass(slots=True)
class DataConfig:
    input_path: Path = Path("data/daily_bars.parquet")
    index_code: str = "000852.SH"
    start_date: str | None = None
    end_date: str | None = None
    min_listed_days: int = 250
    min_amount_20d: float = 30_000_000.0


@dataclass(slots=True)
class LabelConfig:
    primary_horizon: int = 10
    secondary_horizon: int = 5


@dataclass(slots=True)
class PortfolioConfig:
    top_n: int = 12
    rebalance_weekday: int = 4
    fee_rate: float = 0.0003
    slippage_rate: float = 0.0007
    max_weight: float = 0.12


@dataclass(slots=True)
class TrainConfig:
    train_days: int = 504
    valid_days: int = 126
    min_train_rows: int = 20_000
    features: list[str] = field(default_factory=list)
    random_state: int = 42
    lgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    neutralize_scores: bool = False
    neutralize_industry: bool = True
    neutralize_size: bool = True
    factor_eval: "FactorEvalConfig" = field(default_factory=lambda: FactorEvalConfig())


@dataclass(slots=True)
class FactorEvalWeights:
    abs_rank_ic_ir_mean: float = 0.30
    abs_mean_rank_ic_mean: float = 0.20
    bucket_return_spearman_mean: float = 0.15
    positive_top_bucket_excess_year_ratio: float = 0.15
    latest_top_bucket_official_index_excess_annual_return_est: float = 0.10
    monotonic_year_ratio: float = 0.05
    positive_ic_year_ratio: float = 0.05
    mean_rank_ic_std_penalty: float = 0.10


@dataclass(slots=True)
class FactorEvalConfig:
    min_years_required_single_window: int = 1
    min_years_required_multi_window: int = 2
    max_years_required: int = 4
    min_abs_rank_ic_ir: float = 0.10
    min_bucket_return_spearman: float = 0.20
    min_positive_top_bucket_excess_year_ratio: float = 0.50
    min_latest_top_bucket_excess: float = 0.0
    core_quantile: float = 0.80
    candidate_quantile: float = 0.50
    fallback_top_n: int = 20
    feature_batch_size: int = 16
    ridge_min_selection_rate_multi_split: float = 0.50
    ridge_min_selection_rate_single_split: float = 1.0
    ridge_max_original_coef_cv: float = 1.5
    weights: FactorEvalWeights = field(default_factory=FactorEvalWeights)


@dataclass(slots=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: Path = Path("outputs")
