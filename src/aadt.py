"""NYSDOT Annual Average Daily Traffic (AADT) extraction and cleaning."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from .api import (
    NYSDOT_AADT_URL,
    cache_key,
    fetch_arcgis_geojson,
    load_cache,
    normalize_columns,
    save_cache,
)

# NYSDOT Region 11 is New York City: Bronx, Kings, New York, Queens, Richmond.
NYC_DOT_REGION = 11

# Retained for backwards compatibility ? this didn't work before but just in case
NYC_BBOX = (-74.30, 40.45, -73.65, 40.95)


def _candidate_query_urls() -> list[str]:
    """Return equivalent NYSDOT query endpoints, preferring MapServer first.

    The Traffic Monitoring service publishes both FeatureServer and MapServer
    views. In practice the MapServer query endpoint can be more stable for
    read-only spatial extracts, so use it first and fall back to the configured
    FeatureServer URL.
    """
    mapserver_url = NYSDOT_AADT_URL.replace("/FeatureServer/", "/MapServer/")
    urls = [mapserver_url, NYSDOT_AADT_URL]
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(urls))


def fetch_aadt(
    *,
    bbox: tuple[float, float, float, float] = NYC_BBOX,
    refresh: bool = False,
    cache_dir: str | Path = "data/cache",
    page_size: int = 500,
) -> gpd.GeoDataFrame:
    """Fetch current NYSDOT AADT line features for New York City.

    The NYSDOT layer already contains ``DOT_RegionNumber`` and Region 11 is
    exactly New York City. Querying that integer field is simpler and more
    robust than asking the ArcGIS server to evaluate a five-value CountyFIPS
    ``IN (...)`` expression or reproject a WGS84 bounding box. This gave us
    500 errors

    ``bbox`` is retained for notebook compatibility. It is applied locally
    after the API fetch, not sent to NYSDOT.
    """
    where = f"DOT_RegionNumber = {NYC_DOT_REGION} AND AADT > 0"

    key = cache_key(
        "nysdot_aadt",
        dot_region=NYC_DOT_REGION,
        bbox=bbox,
    )

    if not refresh:
        cached = load_cache(key, cache_dir=cache_dir)
        if cached is not None:
            print(f"Using cached AADT snapshot: {key}")
            return cached

    # A smaller page size is intentionally used here. Although the service
    # advertises a 2,000-record maximum, requesting 2,000 polyline features and
    # server-side reprojection to WGS84 can trigger intermittent HTTP 500s.
    effective_page_size = min(page_size, 500)

    last_error: Exception | None = None
    gdf: gpd.GeoDataFrame | None = None

    for query_url in _candidate_query_urls():
        try:
            print(f"Fetching AADT from: {query_url}")
            gdf = fetch_arcgis_geojson(
                query_url,
                where=where,
                out_fields="*",
                bbox=None,
                out_crs=4326,
                page_size=effective_page_size,
                order_by="OBJECTID ASC",
            )
            if not gdf.empty:
                break
        except Exception as exc:  # surface one endpoint failure, then try the other
            last_error = exc
            print(f"AADT endpoint failed: {exc}")
            gdf = None

    if gdf is None:
        raise RuntimeError(
            "NYSDOT AADT could not be fetched from either the MapServer or "
            "FeatureServer query endpoint."
        ) from last_error

    gdf = prepare_aadt(gdf, combined_directions_only=True)

    # Preserve the old bbox argument's semantics, but clip locally after the
    # Region 11 fetch succeeds.
    if bbox is not None and not gdf.empty:
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")

        minx, miny, maxx, maxy = bbox
        bbox_geom = box(minx, miny, maxx, maxy)
        gdf = gdf[gdf.geometry.intersects(bbox_geom)].copy()

    save_cache(
        gdf,
        key,
        cache_dir=cache_dir,
        metadata={
            "source": NYSDOT_AADT_URL,
            "dot_region": NYC_DOT_REGION,
            "bbox": bbox,
        },
    )
    return gdf


def prepare_aadt(
    gdf: gpd.GeoDataFrame,
    *,
    combined_directions_only: bool = True,
) -> gpd.GeoDataFrame:
    """Normalize AADT schema and retain current/combined-direction observations."""
    out = normalize_columns(gdf)
    if out.crs is None:
        out = out.set_crs("EPSG:4326")

    numeric = [
        "aadt",
        "aadt_last_act",
        "calculation_year",
        "year_last_act",
        "speed",
        "functional_class",
        "federal_direction",
        "su_aadt",
        "cu_aadt",
        "avg_truck_percent",
        "dot_region_number",
    ]
    for col in numeric:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "retired" in out:
        out = out[~out["retired"].astype("string").str.upper().eq("Y")].copy()

    if combined_directions_only and "federal_direction" in out:
        combined = out["federal_direction"].isna() | out["federal_direction"].isin([0, 9])
        # Use combined-direction records if supplied; do not accidentally return
        # zero rows if a future schema omits them.
        if combined.any():
            out = out[combined].copy()

    if "aadt" not in out:
        raise KeyError(f"NYSDOT response does not contain AADT. Columns: {list(out.columns)}")

    out = out[out["aadt"].gt(0) & out.geometry.notna()].copy()
    return gpd.GeoDataFrame(out, geometry="geometry", crs=out.crs)
