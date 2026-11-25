"""HTTP, pagination, normalization, and local snapshot-cache helpers.

Treat public APIs as the authoritative source while
keeping local timestamped snapshots so analyses can be reproduced without
re-downloading large datasets on every notebook run.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

NYC_DOMAIN = "https://data.cityofnewyork.us"
CRASHES_ID = "h9gi-nx95"
PERSON_ID = "f55k-p6yu"
VEHICLES_ID = "bm4k-52h4"
LION_ID = "2v4z-66xt"  # NYC Open Data catalog identifier
LION_ARCGIS_URL = (
    "https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/ArcGIS/rest/services/"
    "LION/FeatureServer/0/query"
)
TRAFFIC_COUNTS_ID = "7ym2-wayt"

NYSDOT_AADT_URL = (
    "https://gis.dot.ny.gov/hostingny/rest/services/"
    "Roadways/Traffic_Monitoring/FeatureServer/1/query"
)

DEFAULT_CACHE_DIR = Path(os.getenv("WSW_CACHE_DIR", "data/cache"))


def make_session(app_token: str | None = None) -> requests.Session:
    """Create a requests Session with retries and an optional Socrata token."""
    token = app_token or os.getenv("SOCRATA_APP_TOKEN")
    session = requests.Session()
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "worst-streets-watchlist/0.1"})
    if token:
        session.headers["X-App-Token"] = token
    return session


def snake_case(value: str) -> str:
    """Convert API column names to predictable snake_case, including acronyms."""
    value = str(value)
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^0-9A-Za-z]+", "_", value)
    return value.strip("_").lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized column names."""
    out = df.copy()
    out.columns = [snake_case(c) for c in out.columns]
    return out


def normalize_segment_id(series: pd.Series) -> pd.Series:
    """Normalize LION SegmentID values to seven-character zero-filled strings."""
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    return numeric.astype("string").str.zfill(7)


def _cache_key(prefix: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return f"{prefix}_{digest}"


def cache_key(prefix: str, **kwargs: Any) -> str:
    """Public deterministic cache-key helper."""
    return _cache_key(prefix, kwargs)


def save_cache(
    obj: Any,
    name: str,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist an object as pickle plus JSON metadata with retrieval timestamp."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / f"{name}.pkl"
    meta_path = cache_dir / f"{name}.meta.json"

    with data_path.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "cache_name": name,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_type": type(obj).__name__,
    }
    if hasattr(obj, "shape"):
        meta["shape"] = list(obj.shape)
    if metadata:
        meta.update(metadata)
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return data_path


def load_cache(
    name: str,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> Any | None:
    """Load a cached object, returning None when no snapshot exists."""
    data_path = Path(cache_dir) / f"{name}.pkl"
    if not data_path.exists():
        return None
    with data_path.open("rb") as fh:
        return pickle.load(fh)


def fetch_socrata(
    dataset_id: str,
    *,
    where: str | None = None,
    select: str | Iterable[str] | None = None,
    order: str | None = None,
    page_size: int = 50_000,
    session: requests.Session | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch a Socrata dataset into a DataFrame with pagination.

    Parameters are passed as SoQL query parameters. For large mutable datasets,
    use a deterministic ``order`` to reduce the chance of page drift while the
    source is being updated.
    """
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    session = session or make_session()
    url = f"{NYC_DOMAIN}/resource/{dataset_id}.json"
    if isinstance(select, (list, tuple, set)):
        select = ",".join(select)

    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        params: dict[str, Any] = {"$limit": page_size, "$offset": offset}
        if where:
            params["$where"] = where
        if select:
            params["$select"] = select
        if order:
            params["$order"] = order

        response = session.get(url, params=params, timeout=120)
        response.raise_for_status()
        rows = response.json()
        if not rows:
            break

        frames.append(pd.DataFrame(rows))
        offset += len(rows)
        if verbose:
            print(f"{dataset_id}: fetched {offset:,} rows")
        if len(rows) < page_size:
            break

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_socrata_geojson(
    dataset_id: str,
    *,
    where: str | None = None,
    select: str | Iterable[str] | None = None,
    order: str | None = None,
    page_size: int = 10_000,
    crs: str = "EPSG:4326",
    session: requests.Session | None = None,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """Fetch a Socrata geospatial dataset as a paginated GeoDataFrame."""
    session = session or make_session()
    url = f"{NYC_DOMAIN}/resource/{dataset_id}.geojson"
    if isinstance(select, (list, tuple, set)):
        select = ",".join(select)

    frames: list[gpd.GeoDataFrame] = []
    offset = 0
    while True:
        params: dict[str, Any] = {"$limit": page_size, "$offset": offset}
        if where:
            params["$where"] = where
        if select:
            params["$select"] = select
        if order:
            params["$order"] = order

        response = session.get(url, params=params, timeout=180)
        response.raise_for_status()
        payload = response.json()
        features = payload.get("features", [])
        if not features:
            break

        page = gpd.GeoDataFrame.from_features(features, crs=crs)
        frames.append(page)
        offset += len(page)
        if verbose:
            print(f"{dataset_id}: fetched {offset:,} geospatial rows")
        if len(page) < page_size:
            break

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=crs)


def fetch_arcgis_geojson(
    query_url: str,
    *,
    where: str = "1=1",
    out_fields: str = "*",
    bbox: tuple[float, float, float, float] | None = None,
    bbox_crs: int = 4326,
    out_crs: int = 4326,
    page_size: int = 2_000,
    order_by: str = "OBJECTID",
    session: requests.Session | None = None,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """Fetch an ArcGIS FeatureServer query endpoint as paginated GeoJSON."""
    session = session or make_session()
    frames: list[gpd.GeoDataFrame] = []
    offset = 0

    while True:
        params: dict[str, Any] = {
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "true",
            "outSR": out_crs,
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "orderByFields": order_by,
            "f": "geojson",
        }
        if bbox is not None:
            params.update(
                {
                    "geometry": ",".join(map(str, bbox)),
                    "geometryType": "esriGeometryEnvelope",
                    "inSR": bbox_crs,
                    "spatialRel": "esriSpatialRelIntersects",
                }
            )

        response = session.get(query_url, params=params, timeout=180)
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"ArcGIS query error: {payload['error']}")
        features = payload.get("features", [])
        if not features:
            break

        page = gpd.GeoDataFrame.from_features(features, crs=f"EPSG:{out_crs}")
        frames.append(page)
        offset += len(page)
        if verbose:
            print(f"ArcGIS: fetched {offset:,} features")
        if len(page) < page_size:
            break

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{out_crs}")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=f"EPSG:{out_crs}")
