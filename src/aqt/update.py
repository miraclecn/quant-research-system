from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


@dataclass(slots=True)
class TableStatus:
    table: str
    min_date: str | None
    max_date: str | None
    row_count: int


@dataclass(slots=True)
class UpdateSummary:
    db_path: Path
    index_code: str
    start_date: str
    end_date: str
    trade_dates: int
    daily_kline_before: TableStatus
    daily_kline_after: TableStatus
    daily_basic_before: TableStatus
    daily_basic_after: TableStatus
    stock_basic_before: TableStatus
    stock_basic_after: TableStatus
    index_weight_before: TableStatus
    index_weight_after: TableStatus
    index_daily_before: TableStatus
    index_daily_after: TableStatus
    fetched_daily_kline_rows: int
    fetched_daily_basic_rows: int
    fetched_stock_basic_rows: int
    fetched_index_weight_rows: int
    fetched_index_daily_rows: int


@dataclass(slots=True)
class IndexWeightUpdateSummary:
    db_path: Path
    index_code: str
    start_date: str
    end_date: str
    before: TableStatus
    after: TableStatus
    fetched_rows: int


@dataclass(slots=True)
class IndexDailyUpdateSummary:
    db_path: Path
    index_code: str
    start_date: str
    end_date: str
    before: TableStatus
    after: TableStatus
    fetched_rows: int


@dataclass(slots=True)
class PruneRecommendation:
    table: str
    action: str
    reason: str
    row_count: int


@dataclass(slots=True)
class PrunePlan:
    db_path: Path
    keep: list[PruneRecommendation]
    drop: list[PruneRecommendation]


def _fetch_status(con: duckdb.DuckDBPyConnection, table: str, date_col: str | None) -> TableStatus:
    table_exists = con.execute(
        "SELECT COUNT(*) FROM duckdb_tables() WHERE schema_name = 'main' AND table_name = ?",
        [table],
    ).fetchone()[0]
    if not table_exists:
        return TableStatus(table=table, min_date=None, max_date=None, row_count=0)

    if date_col is None:
        row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return TableStatus(table=table, min_date=None, max_date=None, row_count=int(row_count))

    min_date, max_date, row_count = con.execute(
        f"SELECT MIN({date_col}), MAX({date_col}), COUNT(*) FROM {table}"
    ).fetchone()
    return TableStatus(
        table=table,
        min_date=str(min_date) if min_date is not None else None,
        max_date=str(max_date) if max_date is not None else None,
        row_count=int(row_count),
    )


def _today_ymd() -> str:
    return pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d")


def _lookback_start(max_date: str | None, lookback_days: int) -> str | None:
    if max_date is None:
        return None
    return (pd.Timestamp(max_date) - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")


def _resolve_update_window(
    daily_kline_before: TableStatus,
    daily_basic_before: TableStatus,
    start_date: str | None,
    end_date: str | None,
    lookback_days: int,
) -> tuple[str, str]:
    resolved_end = (end_date or _today_ymd()).replace("-", "")
    if start_date:
        return start_date.replace("-", ""), resolved_end

    candidates = [
        candidate
        for candidate in [
            _lookback_start(daily_kline_before.max_date, lookback_days),
            _lookback_start(daily_basic_before.max_date, lookback_days),
        ]
        if candidate is not None
    ]
    if candidates:
        return min(candidates), resolved_end
    return "20140101", resolved_end


def _normalize_dataframe(df: pd.DataFrame, ordered_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=ordered_columns)
    out = df.copy()
    for col in ordered_columns:
        if col not in out.columns:
            out[col] = None
    out = out[ordered_columns]
    return out.drop_duplicates().reset_index(drop=True)


def _get_tushare_pro(token: str):
    import tushare as ts

    return ts.pro_api(token)


def _fetch_trade_dates(pro, start_date: str, end_date: str) -> list[str]:
    cal = pro.trade_cal(exchange="", start_date=start_date, end_date=end_date, is_open="1")
    if cal is None or cal.empty:
        return []
    return sorted(cal["cal_date"].astype(str).unique().tolist())


def _call_tushare(api_call, min_interval_sec: float = 0.35):
    last_called_at = getattr(_call_tushare, "_last_called_at", None)
    if last_called_at is not None:
        elapsed = time.monotonic() - last_called_at
        if elapsed < min_interval_sec:
            time.sleep(min_interval_sec - elapsed)

    while True:
        try:
            result = api_call()
            _call_tushare._last_called_at = time.monotonic()
            return result
        except Exception as exc:
            message = str(exc)
            if "每分钟最多访问该接口200次" not in message:
                raise
            time.sleep(60)
            _call_tushare._last_called_at = time.monotonic()


def _fetch_daily_kline(pro, trade_dates: list[str]) -> pd.DataFrame:
    frames = []
    for trade_date in trade_dates:
        df = _call_tushare(lambda: pro.daily(trade_date=trade_date))
        if df is not None and not df.empty:
            frames.append(df)
    ordered_columns = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    ]
    if not frames:
        return pd.DataFrame(columns=ordered_columns)
    merged = pd.concat(frames, ignore_index=True)
    return _normalize_dataframe(merged, ordered_columns)


def _fetch_daily_basic(pro, trade_dates: list[str]) -> pd.DataFrame:
    frames = []
    fields = ",".join(
        [
            "ts_code",
            "trade_date",
            "close",
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "pe",
            "pe_ttm",
            "pb",
            "ps",
            "ps_ttm",
            "dv_ratio",
            "dv_ttm",
            "total_share",
            "float_share",
            "free_share",
            "total_mv",
            "circ_mv",
        ]
    )
    for trade_date in trade_dates:
        df = _call_tushare(lambda trade_date=trade_date: pro.daily_basic(trade_date=trade_date, fields=fields))
        if df is not None and not df.empty:
            frames.append(df)
    ordered_columns = fields.split(",")
    if not frames:
        return pd.DataFrame(columns=ordered_columns)
    merged = pd.concat(frames, ignore_index=True)
    return _normalize_dataframe(merged, ordered_columns)


def _fetch_stock_basic(pro) -> pd.DataFrame:
    fields = "ts_code,symbol,name,area,industry,list_date,delist_date,is_hs"
    frames = []
    for list_status in ["L", "D", "P"]:
        df = _call_tushare(lambda list_status=list_status: pro.stock_basic(exchange="", list_status=list_status, fields=fields))
        if df is not None and not df.empty:
            frames.append(df)
    ordered_columns = fields.split(",")
    if not frames:
        return pd.DataFrame(columns=ordered_columns)
    merged = pd.concat(frames, ignore_index=True)
    return _normalize_dataframe(merged, ordered_columns)


def _month_windows(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    months = pd.period_range(start=start_ts.to_period("M"), end=end_ts.to_period("M"), freq="M")
    windows: list[tuple[str, str]] = []
    for month in months:
        month_start = month.start_time.strftime("%Y%m%d")
        month_end = month.end_time.strftime("%Y%m%d")
        windows.append((month_start, month_end))
    return windows


def _fetch_index_weight(pro, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    frames = []
    ordered_columns = ["index_code", "con_code", "trade_date", "weight"]
    for month_start, month_end in _month_windows(start_date, end_date):
        df = _call_tushare(
            lambda month_start=month_start, month_end=month_end: pro.index_weight(
                index_code=index_code,
                start_date=month_start,
                end_date=month_end,
            )
        )
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=ordered_columns)
    merged = pd.concat(frames, ignore_index=True)
    return _normalize_dataframe(merged, ordered_columns)


def _fetch_index_daily(pro, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    fields = "ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount"
    df = _call_tushare(
        lambda: pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date, fields=fields)
    )
    ordered_columns = fields.split(",")
    if df is None or df.empty:
        return pd.DataFrame(columns=ordered_columns)
    return _normalize_dataframe(df, ordered_columns)


def _ensure_raw_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
CREATE TABLE IF NOT EXISTS daily_kline (
    ts_code VARCHAR,
    trade_date VARCHAR,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    pre_close DOUBLE,
    change DOUBLE,
    pct_chg DOUBLE,
    vol DOUBLE,
    amount DOUBLE
)
"""
    )
    con.execute(
        """
CREATE TABLE IF NOT EXISTS daily_basic (
    ts_code VARCHAR,
    trade_date VARCHAR,
    close DOUBLE,
    turnover_rate DOUBLE,
    turnover_rate_f DOUBLE,
    volume_ratio DOUBLE,
    pe DOUBLE,
    pe_ttm DOUBLE,
    pb DOUBLE,
    ps DOUBLE,
    ps_ttm DOUBLE,
    dv_ratio DOUBLE,
    dv_ttm DOUBLE,
    total_share DOUBLE,
    float_share DOUBLE,
    free_share DOUBLE,
    total_mv DOUBLE,
    circ_mv DOUBLE
)
"""
    )
    con.execute(
        """
CREATE TABLE IF NOT EXISTS stock_basic (
    ts_code VARCHAR,
    symbol VARCHAR,
    name VARCHAR,
    area VARCHAR,
    industry VARCHAR,
    list_date VARCHAR,
    delist_date VARCHAR,
    is_hs VARCHAR
)
"""
    )
    con.execute(
        """
CREATE TABLE IF NOT EXISTS index_weight (
    index_code VARCHAR,
    con_code VARCHAR,
    trade_date VARCHAR,
    weight DOUBLE
)
"""
    )
    con.execute(
        """
CREATE TABLE IF NOT EXISTS index_daily (
    ts_code VARCHAR,
    trade_date VARCHAR,
    close DOUBLE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    pre_close DOUBLE,
    change DOUBLE,
    pct_chg DOUBLE,
    vol DOUBLE,
    amount DOUBLE
)
"""
    )


def _upsert_frame(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
    key_columns: list[str],
) -> None:
    if df.empty:
        return
    con.register("incoming_frame", df)
    columns = df.columns.tolist()
    on_clause = " AND ".join(f"target.{col} = source.{col}" for col in key_columns)
    update_columns = [col for col in columns if col not in key_columns]
    update_clause = ", ".join(f"{col} = source.{col}" for col in update_columns)
    insert_columns = ", ".join(columns)
    insert_values = ", ".join(f"source.{col}" for col in columns)
    con.execute(
        f"""
MERGE INTO {table} AS target
USING incoming_frame AS source
ON {on_clause}
WHEN MATCHED THEN UPDATE SET {update_clause}
WHEN NOT MATCHED THEN INSERT ({insert_columns}) VALUES ({insert_values})
"""
    )
    con.unregister("incoming_frame")


def _replace_stock_basic(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    con.execute("DELETE FROM stock_basic")
    if df.empty:
        return
    con.register("incoming_stock_basic", df)
    con.execute("INSERT INTO stock_basic SELECT * FROM incoming_stock_basic")
    con.unregister("incoming_stock_basic")


def _replace_index_weight(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
) -> None:
    _upsert_frame(con, "index_weight", df, key_columns=["index_code", "con_code", "trade_date"])


def _replace_index_daily(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
) -> None:
    _upsert_frame(con, "index_daily", df, key_columns=["ts_code", "trade_date"])


def update_raw(
    db_path: Path,
    tushare_token: str | None = None,
    index_code: str = "000852.SH",
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int = 7,
) -> UpdateSummary:
    token = tushare_token or os.getenv("TUSHARE_TOKEN")
    if not token:
        raise ValueError("Missing Tushare token. Pass --tushare-token or set TUSHARE_TOKEN.")

    with duckdb.connect(str(db_path)) as con:
        _ensure_raw_tables(con)
        daily_kline_before = _fetch_status(con, "daily_kline", "trade_date")
        daily_basic_before = _fetch_status(con, "daily_basic", "trade_date")
        stock_basic_before = _fetch_status(con, "stock_basic", None)
        index_weight_before = _fetch_status(con, "index_weight", "trade_date")
        index_daily_before = _fetch_status(con, "index_daily", "trade_date")

    resolved_start, resolved_end = _resolve_update_window(
        daily_kline_before=daily_kline_before,
        daily_basic_before=daily_basic_before,
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days,
    )

    pro = _get_tushare_pro(token)
    trade_dates = _fetch_trade_dates(pro, start_date=resolved_start, end_date=resolved_end)
    daily_kline_df = _fetch_daily_kline(pro, trade_dates)
    daily_basic_df = _fetch_daily_basic(pro, trade_dates)
    stock_basic_df = _fetch_stock_basic(pro)
    index_weight_df = _fetch_index_weight(pro, index_code=index_code, start_date=resolved_start, end_date=resolved_end)
    index_daily_df = _fetch_index_daily(pro, index_code=index_code, start_date=resolved_start, end_date=resolved_end)

    with duckdb.connect(str(db_path)) as con:
        _ensure_raw_tables(con)
        _upsert_frame(con, "daily_kline", daily_kline_df, key_columns=["ts_code", "trade_date"])
        _upsert_frame(con, "daily_basic", daily_basic_df, key_columns=["ts_code", "trade_date"])
        _replace_stock_basic(con, stock_basic_df)
        _replace_index_weight(con, index_weight_df)
        _replace_index_daily(con, index_daily_df)
        daily_kline_after = _fetch_status(con, "daily_kline", "trade_date")
        daily_basic_after = _fetch_status(con, "daily_basic", "trade_date")
        stock_basic_after = _fetch_status(con, "stock_basic", None)
        index_weight_after = _fetch_status(con, "index_weight", "trade_date")
        index_daily_after = _fetch_status(con, "index_daily", "trade_date")

    return UpdateSummary(
        db_path=db_path,
        index_code=index_code,
        start_date=resolved_start,
        end_date=resolved_end,
        trade_dates=len(trade_dates),
        daily_kline_before=daily_kline_before,
        daily_kline_after=daily_kline_after,
        daily_basic_before=daily_basic_before,
        daily_basic_after=daily_basic_after,
        stock_basic_before=stock_basic_before,
        stock_basic_after=stock_basic_after,
        index_weight_before=index_weight_before,
        index_weight_after=index_weight_after,
        index_daily_before=index_daily_before,
        index_daily_after=index_daily_after,
        fetched_daily_kline_rows=len(daily_kline_df),
        fetched_daily_basic_rows=len(daily_basic_df),
        fetched_stock_basic_rows=len(stock_basic_df),
        fetched_index_weight_rows=len(index_weight_df),
        fetched_index_daily_rows=len(index_daily_df),
    )


def update_index_weight(
    db_path: Path,
    tushare_token: str | None = None,
    index_code: str = "000852.SH",
    start_date: str | None = None,
    end_date: str | None = None,
) -> IndexWeightUpdateSummary:
    token = tushare_token or os.getenv("TUSHARE_TOKEN")
    if not token:
        raise ValueError("Missing Tushare token. Pass --tushare-token or set TUSHARE_TOKEN.")

    resolved_start = (start_date or "20150101").replace("-", "")
    resolved_end = (end_date or _today_ymd()).replace("-", "")

    with duckdb.connect(str(db_path)) as con:
        _ensure_raw_tables(con)
        before = _fetch_status(con, "index_weight", "trade_date")

    pro = _get_tushare_pro(token)
    index_weight_df = _fetch_index_weight(pro, index_code=index_code, start_date=resolved_start, end_date=resolved_end)

    with duckdb.connect(str(db_path)) as con:
        _ensure_raw_tables(con)
        _replace_index_weight(con, index_weight_df)
        after = _fetch_status(con, "index_weight", "trade_date")

    return IndexWeightUpdateSummary(
        db_path=db_path,
        index_code=index_code,
        start_date=resolved_start,
        end_date=resolved_end,
        before=before,
        after=after,
        fetched_rows=len(index_weight_df),
    )


def update_index_daily(
    db_path: Path,
    tushare_token: str | None = None,
    index_code: str = "000852.SH",
    start_date: str | None = None,
    end_date: str | None = None,
) -> IndexDailyUpdateSummary:
    token = tushare_token or os.getenv("TUSHARE_TOKEN")
    if not token:
        raise ValueError("Missing Tushare token. Pass --tushare-token or set TUSHARE_TOKEN.")

    resolved_start = (start_date or "20150101").replace("-", "")
    resolved_end = (end_date or _today_ymd()).replace("-", "")

    with duckdb.connect(str(db_path)) as con:
        _ensure_raw_tables(con)
        before = _fetch_status(con, "index_daily", "trade_date")

    pro = _get_tushare_pro(token)
    index_daily_df = _fetch_index_daily(pro, index_code=index_code, start_date=resolved_start, end_date=resolved_end)

    with duckdb.connect(str(db_path)) as con:
        _ensure_raw_tables(con)
        _replace_index_daily(con, index_daily_df)
        after = _fetch_status(con, "index_daily", "trade_date")

    return IndexDailyUpdateSummary(
        db_path=db_path,
        index_code=index_code,
        start_date=resolved_start,
        end_date=resolved_end,
        before=before,
        after=after,
        fetched_rows=len(index_daily_df),
    )


def format_update_summary(summary: UpdateSummary) -> str:
    lines = [
        f"db_path={summary.db_path}",
        f"index_code={summary.index_code}",
        f"window start={summary.start_date} end={summary.end_date} trade_dates={summary.trade_dates}",
        (
            "daily_kline "
            f"before_max={summary.daily_kline_before.max_date} "
            f"after_max={summary.daily_kline_after.max_date} "
            f"fetched_rows={summary.fetched_daily_kline_rows}"
        ),
        (
            "daily_basic "
            f"before_max={summary.daily_basic_before.max_date} "
            f"after_max={summary.daily_basic_after.max_date} "
            f"fetched_rows={summary.fetched_daily_basic_rows}"
        ),
        (
            "stock_basic "
            f"before_rows={summary.stock_basic_before.row_count} "
            f"after_rows={summary.stock_basic_after.row_count} "
            f"fetched_rows={summary.fetched_stock_basic_rows}"
        ),
        (
            "index_weight "
            f"before_max={summary.index_weight_before.max_date} "
            f"after_max={summary.index_weight_after.max_date} "
            f"fetched_rows={summary.fetched_index_weight_rows}"
        ),
        (
            "index_daily "
            f"before_max={summary.index_daily_before.max_date} "
            f"after_max={summary.index_daily_after.max_date} "
            f"fetched_rows={summary.fetched_index_daily_rows}"
        ),
    ]
    return "\n".join(lines)


def format_index_weight_update_summary(summary: IndexWeightUpdateSummary) -> str:
    lines = [
        f"db_path={summary.db_path}",
        f"index_code={summary.index_code}",
        f"window start={summary.start_date} end={summary.end_date}",
        (
            "index_weight "
            f"before_max={summary.before.max_date} "
            f"after_max={summary.after.max_date} "
            f"fetched_rows={summary.fetched_rows}"
        ),
    ]
    return "\n".join(lines)


def format_index_daily_update_summary(summary: IndexDailyUpdateSummary) -> str:
    lines = [
        f"db_path={summary.db_path}",
        f"index_code={summary.index_code}",
        f"window start={summary.start_date} end={summary.end_date}",
        (
            "index_daily "
            f"before_max={summary.before.max_date} "
            f"after_max={summary.after.max_date} "
            f"fetched_rows={summary.fetched_rows}"
        ),
    ]
    return "\n".join(lines)


def _table_row_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    counts: dict[str, int] = {}
    table_names = [
        row[0]
        for row in con.execute(
            "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main' ORDER BY table_name"
        ).fetchall()
    ]
    for table_name in table_names:
        try:
            counts[table_name] = int(con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
        except duckdb.Error:
            continue
    try:
        view_names = [
            row[0]
            for row in con.execute(
                "SELECT view_name FROM duckdb_views() WHERE schema_name = 'main' ORDER BY view_name"
            ).fetchall()
        ]
    except duckdb.Error:
        view_names = []
    for view_name in view_names:
        if view_name.startswith("duckdb_") or view_name.startswith("pragma_") or view_name.startswith("sqlite_"):
            continue
        try:
            counts[view_name] = int(con.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0])
        except duckdb.Error:
            counts[view_name] = -1
    return counts


def _object_type_map(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    object_types: dict[str, str] = {}
    for table_name, in con.execute(
        "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main'"
    ).fetchall():
        object_types[table_name] = "table"
    try:
        for view_name, in con.execute("SELECT view_name FROM duckdb_views() WHERE schema_name = 'main'").fetchall():
            object_types[view_name] = "view"
    except duckdb.Error:
        pass
    return object_types


def build_prune_plan(db_path: Path) -> PrunePlan:
    keep_reasons = {
        "daily_kline": "raw daily price table used to rebuild research panels",
        "daily_basic": "raw daily basic table used to rebuild research panels",
        "stock_basic": "raw stock master table used for industry and listing metadata",
        "index_weight": "raw index constituent and weight history for true benchmark membership",
        "index_daily": "raw official index daily bar history for benchmark comparison",
        "fina_indicator": "raw financial indicator table with future factor expansion value",
        "margin_detail": "raw margin trading table with possible alpha value",
        "top_inst": "raw institutional activity table with possible event alpha value",
        "stock_zt_pool": "raw limit-up pool table with event-study value",
        "financial_news": "raw news crawl table; tiny now but potentially accumulative",
    }
    drop_reasons = {
        "kline_clean": "redundant clean copy derivable from daily_kline",
        "daily_basic_clean": "redundant clean copy derivable from daily_basic",
        "kline_raw": "extra daily kline copy; daily_kline should be the canonical raw table",
        "daily_basic_raw": "extra daily basic copy; daily_basic should be the canonical raw table",
        "stock_info_raw": "duplicative stock master data; stock_basic is sufficient",
        "alpha158_factor": "derived factor cache, not raw data",
        "alpha158_factor_clean": "derived factor cache, not raw data",
        "alpha158_factor_clean_v2": "derived experiment table, not raw data",
        "alpha158_factor_final": "derived experiment table, not raw data",
        "factor_data": "derived research panel, not raw data",
        "fundamental_clean": "derived clean financial table, not raw data",
        "fundamental_raw": "likely duplicative with fina_indicator; keep one canonical raw financial source",
        "board_concept_kline": "adjacent market data, not needed for the current raw core",
        "board_industry_kline": "adjacent market data, not needed for the current raw core",
        "stock_board_concept": "adjacent mapping table, not needed for the current raw core",
        "stock_board_industry": "adjacent mapping table, not needed for the current raw core",
        "fund_etf_spot": "separate fund dataset, not needed for the current stock raw core",
        "stock_fund_flow_rank": "presentation/ranking table, not canonical raw history",
        "stock_fund_flow": "empty table with no retained value in current state",
    }

    with duckdb.connect(str(db_path), read_only=True) as con:
        counts = _table_row_counts(con)

    keep = [
        PruneRecommendation(table=table, action="keep", reason=reason, row_count=counts.get(table, 0))
        for table, reason in keep_reasons.items()
        if table in counts
    ]
    drop = [
        PruneRecommendation(table=table, action="drop", reason=reason, row_count=counts.get(table, 0))
        for table, reason in drop_reasons.items()
        if table in counts
    ]

    uncategorized = sorted(set(counts) - {item.table for item in keep} - {item.table for item in drop})
    for table in uncategorized:
        keep.append(
            PruneRecommendation(
                table=table,
                action="review",
                reason="uncategorized table; keep for manual review before any deletion",
                row_count=counts[table],
            )
        )

    keep.sort(key=lambda item: (item.action, item.table))
    drop.sort(key=lambda item: item.table)
    return PrunePlan(db_path=db_path, keep=keep, drop=drop)


def format_prune_plan(plan: PrunePlan) -> str:
    lines = [f"db_path={plan.db_path}", "[keep]"]
    lines.extend(
        f"{item.table}\trows={item.row_count}\treason={item.reason}"
        for item in sorted(plan.keep, key=lambda item: item.table)
    )
    lines.append("[drop]")
    lines.extend(
        f"{item.table}\trows={item.row_count}\treason={item.reason}"
        for item in sorted(plan.drop, key=lambda item: item.table)
    )
    return "\n".join(lines)


def execute_prune_plan(db_path: Path) -> PrunePlan:
    plan = build_prune_plan(db_path)
    if not plan.drop:
        return plan

    with duckdb.connect(str(db_path)) as con:
        object_types = _object_type_map(con)
        for item in plan.drop:
            object_type = object_types.get(item.table, "table")
            if object_type == "view":
                con.execute(f"DROP VIEW IF EXISTS {item.table}")
            else:
                con.execute(f"DROP TABLE IF EXISTS {item.table}")

    return build_prune_plan(db_path)
