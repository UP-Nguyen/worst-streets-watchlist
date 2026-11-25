"""NYC DOT automated traffic-count extraction and aggregation.

These data are sampled 15-minute counts. They are useful as a supplemental
exposure source, but they are not Annual Average Daily Traffic (AADT).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .api import (
    TRAFFIC_COUNTS_ID,
    cache_key,
    fetch_socrata,
    load_cache,
    normalize_columns,
    normalize_segment_id,
    save_cache,
)

TRAFFIC_FIELDS = [
    "requestid",
    "boro",
    "yr",
    "m",
    "d",
    "hh",
    "mm",
    "vol",
    "segmentid",
    "street",
    "fromst",
    "tost",
    "direction",
]


def _two_digit_year(year: int) -> int:
    return year % 100


def fetch_automated_traffic_counts(
    *,
    years: Iterable[int] | None = None,
    refresh: bool = False,
    cache_dir: str | Path = "data/cache",
    page_size: int = 50_000,
) -> pd.DataFrame:
    """Fetch current NYC DOT ATR counts, optionally limited to calendar years."""
    years = sorted(set(int(y) for y in years)) if years else None
    where = None
    if years:
        yy = [_two_digit_year(y) for y in years]
        where = "yr in (" + ",".join(map(str, yy)) + ")"

    key = cache_key("traffic_counts", years=years or "all")
    if not refresh:
        cached = load_cache(key, cache_dir=cache_dir)
        if cached is not None:
            print(f"Using cached traffic-count snapshot: {key}")
            return cached

    df = fetch_socrata(
        TRAFFIC_COUNTS_ID,
        where=where,
        select=TRAFFIC_FIELDS,
        order="yr,m,d,requestid,hh,mm",
        page_size=page_size,
    )
    df = normalize_columns(df)
    for col in ["requestid", "yr", "m", "d", "hh", "mm", "vol"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "segmentid" in df:
        df["segmentid"] = normalize_segment_id(df["segmentid"])
    save_cache(df, key, cache_dir=cache_dir, metadata={"dataset_id": TRAFFIC_COUNTS_ID, "years": years})
    return df


def aggregate_traffic_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Turn 15-minute ATR samples into per-segment observed daily-volume summaries.

    A request/day is treated as one sample day. Summing its 15-minute volumes
    gives an observed daily total when the request covers a full day. The
    output keeps observation counts so analysts can inspect coverage rather
    than treating the result as AADT.
    """
    work = df.copy()
    required = {"requestid", "segmentid", "yr", "m", "d", "vol"}
    missing = required - set(work.columns)
    if missing:
        raise KeyError(f"Traffic data missing required columns: {sorted(missing)}")

    # NYC data store year as two digits. Interpret 00-79 as 2000s.
    full_year = 2000 + pd.to_numeric(work["yr"], errors="coerce")
    work["count_date"] = pd.to_datetime(
        {"year": full_year, "month": work["m"], "day": work["d"]},
        errors="coerce",
    )
    work["vol"] = pd.to_numeric(work["vol"], errors="coerce")

    request_day = (
        work.dropna(subset=["segmentid", "count_date", "vol"])
        .groupby(["segmentid", "requestid", "count_date"], as_index=False)
        .agg(
            observed_volume=("vol", "sum"),
            intervals_observed=("vol", "count"),
        )
    )
    summary = (
        request_day.groupby("segmentid", as_index=False)
        .agg(
            traffic_sample_days=("count_date", "nunique"),
            traffic_requests=("requestid", "nunique"),
            mean_observed_daily_volume=("observed_volume", "mean"),
            median_observed_daily_volume=("observed_volume", "median"),
            latest_traffic_count_date=("count_date", "max"),
            median_intervals_per_request=("intervals_observed", "median"),
        )
    )
    return summary
