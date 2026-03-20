from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

REQUIRED_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
OPTIONAL_DEFAULTS = {
    "turnover_rate": 0.0,
    "float_mv": 0.0,
    "index_weight": 0.0,
    "industry": "UNKNOWN",
    "is_st": False,
    "listed_days": 9999,
    "is_paused": False,
    "is_limit_up": False,
    "is_limit_down": False,
    "in_universe": True,
}
FUNDAMENTAL_COLUMNS = [
    "pe_ttm",
    "pb",
    "ps_ttm",
    "dv_ttm",
    "eps",
    "bps",
    "cfps",
    "ocfps",
    "roe",
    "roe_dt",
    "roa",
    "roic",
    "grossprofit_margin",
    "netprofit_margin",
    "debt_to_assets",
    "current_ratio",
    "quick_ratio",
    "profit_dedt",
    "basic_eps_yoy",
    "dt_eps_yoy",
    "netprofit_yoy",
    "dt_netprofit_yoy",
    "ocf_yoy",
    "tr_yoy",
    "or_yoy",
    "q_sales_yoy",
    "q_op_qoq",
]
FINA_INDICATOR_PANEL_COLUMNS = [col for col in FUNDAMENTAL_COLUMNS if col not in {"pe_ttm", "pb", "ps_ttm", "dv_ttm"}]
DEFAULT_DUCKDB_PATH = Path("stock_data.duckdb")


def _normalize_panel(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover_rate",
        "float_mv",
        "index_weight",
        "listed_days",
    ] + [col for col in FUNDAMENTAL_COLUMNS if col in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    float_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover_rate",
        "float_mv",
        "index_weight",
    ] + [col for col in FUNDAMENTAL_COLUMNS if col in df.columns]
    for col in float_cols:
        df[col] = df[col].astype("float32")
    df["listed_days"] = df["listed_days"].fillna(9999).astype("int32")

    bool_cols = ["is_st", "is_paused", "is_limit_up", "is_limit_down", "in_universe"]
    for col in bool_cols:
        df[col] = df[col].fillna(False).astype(bool)

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype("category")
    if "industry" in df.columns:
        df["industry"] = df["industry"].astype("category")

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def _apply_date_filters(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    out = df
    if start_date:
        out = out.loc[out["date"] >= pd.Timestamp(start_date)]
    if end_date:
        out = out.loc[out["date"] <= pd.Timestamp(end_date)]
    out = out.reset_index(drop=True)
    for col in ["symbol", "industry"]:
        if col in out.columns and pd.api.types.is_categorical_dtype(out[col]):
            out[col] = out[col].cat.remove_unused_categories()
    return out


def _load_duckdb_panel(path: Path, index_code: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    filters = []
    params: list[str] = []
    if start_date:
        filters.append("strptime(k.trade_date, '%Y%m%d') >= CAST(? AS DATE)")
        params.append(start_date)
    if end_date:
        filters.append("strptime(k.trade_date, '%Y%m%d') <= CAST(? AS DATE)")
        params.append(end_date)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    with duckdb.connect(str(path), read_only=True) as con:
        has_index_weight = bool(
            con.execute(
                "SELECT COUNT(*) > 0 FROM duckdb_tables() WHERE schema_name = 'main' AND table_name = 'index_weight'"
            ).fetchone()[0]
        )
        has_fina_indicator_clean = bool(
            con.execute(
                "SELECT COUNT(*) > 0 FROM duckdb_tables() WHERE schema_name = 'main' AND table_name = 'fina_indicator_clean'"
            ).fetchone()[0]
        )
        weight_cte = ""
        weight_select = "0.0 AS index_weight,\n        FALSE AS in_index_weight,\n        FALSE AS has_index_weight_table,"
        weight_join = ""
        if has_index_weight:
            weight_cte = f"""
weights AS (
    SELECT
        con_code AS symbol,
        strptime(trade_date, '%Y%m%d') AS weight_date,
        weight
    FROM index_weight
    WHERE index_code = '{index_code}'
),
weight_dates AS (
    SELECT DISTINCT weight_date
    FROM weights
),
"""
            weight_select = "w.weight AS index_weight,\n        (w.symbol IS NOT NULL) AS in_index_weight,\n        TRUE AS has_index_weight_table,"
            weight_join = """
    LEFT JOIN LATERAL (
        SELECT
            weight_date
        FROM weight_dates AS wd
        WHERE wd.weight_date <= strptime(k.trade_date, '%Y%m%d')
        ORDER BY wd.weight_date DESC
        LIMIT 1
    ) AS snap ON TRUE
    LEFT JOIN weights AS w
        ON w.symbol = k.ts_code AND w.weight_date = snap.weight_date
"""

        fina_select = ",\n        ".join(f"fi.{col} AS {col}" for col in FINA_INDICATOR_PANEL_COLUMNS)
        fina_join = ""
        if has_fina_indicator_clean:
            fina_join = f"""
    LEFT JOIN LATERAL (
        SELECT
            {", ".join(f"fic.{col} AS {col}" for col in FINA_INDICATOR_PANEL_COLUMNS)}
        FROM fina_indicator_clean AS fic
        WHERE fic.ts_code = k.ts_code
          AND strptime(fic.ann_date, '%Y%m%d') <= strptime(k.trade_date, '%Y%m%d')
        ORDER BY strptime(fic.ann_date, '%Y%m%d') DESC, strptime(fic.end_date, '%Y%m%d') DESC
        LIMIT 1
    ) AS fi ON TRUE
"""
        else:
            fina_select = ",\n        ".join(f"CAST(NULL AS DOUBLE) AS {col}" for col in FINA_INDICATOR_PANEL_COLUMNS)

        cte_prefix = weight_cte + "\n" if weight_cte else ""
        panel_from = "base"
        panel_select = fina_select
        panel_join = ""
        if has_fina_indicator_clean:
            panel_from = "base AS b"
            panel_select = ",\n        ".join(f"fi.{col} AS {col}" for col in FINA_INDICATOR_PANEL_COLUMNS)
            panel_join = fina_join.replace("k.ts_code", "b.symbol").replace("k.trade_date", "strftime(b.date, '%Y%m%d')")

        query = f"""
WITH
{cte_prefix}base AS (
    SELECT
        strptime(k.trade_date, '%Y%m%d') AS date,
        k.ts_code AS symbol,
        k.open,
        k.high,
        k.low,
        k.close,
        k.vol * 100.0 AS volume,
        k.amount * 1000.0 AS amount,
        db.turnover_rate,
        db.pe_ttm,
        db.pb,
        db.ps_ttm,
        db.dv_ttm,
        db.circ_mv * 10000.0 AS float_mv,
        COALESCE(sb.industry, 'UNKNOWN') AS industry,
        COALESCE(sb.name, '') AS name,
        {weight_select}
        CASE
            WHEN sb.list_date IS NULL THEN 9999
            ELSE GREATEST(date_diff('day', strptime(sb.list_date, '%Y%m%d'), strptime(k.trade_date, '%Y%m%d')), 0)
        END AS listed_days,
        row_number() OVER (PARTITION BY k.trade_date ORDER BY db.circ_mv DESC NULLS LAST, k.ts_code) AS float_mv_rank
    FROM daily_kline AS k
    LEFT JOIN daily_basic AS db
        ON k.ts_code = db.ts_code AND k.trade_date = db.trade_date
    LEFT JOIN stock_basic AS sb
        ON k.ts_code = sb.ts_code
{weight_join}
    {where_clause}
),
panel AS (
    SELECT
        b.date,
        b.symbol,
        b.open,
        b.high,
        b.low,
        b.close,
        b.volume,
        b.amount,
        b.turnover_rate,
        b.pe_ttm,
        b.pb,
        b.ps_ttm,
        b.dv_ttm,
        b.float_mv,
        b.industry,
        b.name,
        b.index_weight,
        b.in_index_weight,
        b.has_index_weight_table,
        {panel_select},
        b.listed_days,
        b.float_mv_rank
    FROM {panel_from}
{panel_join}
)
SELECT
    date,
    symbol,
    open,
    high,
    low,
    close,
    volume,
    amount,
    COALESCE(turnover_rate, 0.0) AS turnover_rate,
    COALESCE(float_mv, 0.0) AS float_mv,
    industry,
    COALESCE(index_weight, 0.0) AS index_weight,
    {", ".join(FUNDAMENTAL_COLUMNS)},
    name LIKE 'ST%%' OR name LIKE '*ST%%' OR name LIKE 'S*ST%%' AS is_st,
    listed_days,
    FALSE AS is_paused,
    CASE
        WHEN symbol LIKE '300%%' OR symbol LIKE '688%%' THEN (close / NULLIF(open, 0.0) - 1.0) >= 0.195
        ELSE (close / NULLIF(open, 0.0) - 1.0) >= 0.098
    END AS is_limit_up,
    CASE
        WHEN symbol LIKE '300%%' OR symbol LIKE '688%%' THEN (close / NULLIF(open, 0.0) - 1.0) <= -0.195
        ELSE (close / NULLIF(open, 0.0) - 1.0) <= -0.098
    END AS is_limit_down,
    CASE
        WHEN has_index_weight_table THEN in_index_weight
        ELSE float_mv_rank BETWEEN 801 AND 1800
    END AS in_universe
FROM panel
"""
        df = con.execute(query, params).df()
    return _normalize_panel(df)


def resolve_input_path(path: Path) -> Path:
    candidates = [path]
    if path.suffix == ".parquet":
        candidates.append(path.with_suffix(".csv"))
        candidates.append(path.with_suffix(".duckdb"))
    elif path.suffix == ".csv":
        candidates.append(path.with_suffix(".parquet"))
        candidates.append(path.with_suffix(".duckdb"))
    elif path.suffix == ".duckdb":
        candidates.extend([path.with_suffix(".parquet"), path.with_suffix(".csv")])
    else:
        candidates.extend([path.with_suffix(".parquet"), path.with_suffix(".csv"), path.with_suffix(".duckdb")])

    default_candidates = [Path("data/daily_bars.parquet"), Path("data/daily_bars.csv"), DEFAULT_DUCKDB_PATH]
    for candidate in default_candidates:
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Input data not found. Tried: {tried}")


def load_panel(
    path: Path,
    index_code: str = "000852.SH",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    path = resolve_input_path(path)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".duckdb":
        return _load_duckdb_panel(path, index_code=index_code, start_date=start_date, end_date=end_date)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")

    df = _normalize_panel(df)
    return _apply_date_filters(df, start_date=start_date, end_date=end_date)


def export_panel(
    input_path: Path,
    output_path: Path,
    index_code: str = "000852.SH",
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path:
    panel = load_panel(input_path, index_code=index_code, start_date=start_date, end_date=end_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    return output_path
