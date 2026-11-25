"""NYSDOT Annual Average Daily Traffic (AADT) extraction and cleaning.

This version uses NYSDOT's Roadway Data Viewer AADT layer (layer 5), which is
more stable for programmatic roadway-inventory downloads than the older
Traffic Monitoring AADT layer.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from .api import (
    cache_key,
    fetch_arcgis_geojson,
    load_cache,
    normalize_columns,
    save_cache,
)


# NYSDOT Roadway Data Viewer -> AADT layer (5)
# Layer docs:
# https://gis.dot.ny.gov/hostingny/rest/services/Roadways/RoadwayDataViewer/MapServer/5
NYSDOT_AADT_URLS = (
    "https://gis.dot.ny.gov/hostingny/rest/services/Roadways/RoadwayDataViewer/MapServer/5/query",
    "https://gis.dot.ny.gov/hostingny/rest/services/Roadways/RoadwayDataViewer/FeatureServer/5/query",
)

# Broad bounding box covering all five NYC boroughs. no longer send this
# envelope to NYSDOT. It is applied locally after the statewide API pull.
NYC_BBOX = (-74.30, 40.45, -73.65, 40.95)


def fetch_aadt(
    *,
    bbox: tuple[float, float, float, float] = NYC_BBOX,
    refresh: bool = False,
    cache_dir: str | Path = "data/cache",
    page_size: int = 5_000,
) -> gpd.GeoDataFrame:
    """Fetch current NYSDOT AADT roadway segments for NYC.

    The NYSDOT Roadway Data Viewer AADT service is pulled statewide with a
    deliberately simple ``where=1=1`` query, then filtered locally to Region 11
    (NYC) and clipped to ``bbox``. This avoids server-side SQL/spatial filters
    that have intermittently returned HTTP 500 errors on NYSDOT services.
    """
    key = cache_key(
        "nysdot_aadt_roadway_viewer",
        bbox=bbox,
        source="roadway_data_viewer_layer_5",
    )

    if not refresh:
        cached = load_cache(key, cache_dir=cache_dir)
        if cached is not None:
            print(f"Using cached AADT snapshot: {key}")
            return cached

    gdf = None
    last_error = None

    # Keep the request as simple as possible. We filter to NYC locally after
    # download. Layer 5 advertises a max record count of 20,000; using 5,000
    # per page keeps payloads manageable while still being efficient.
    for query_url in NYSDOT_AADT_URLS:
        try:
            print(f"Fetching AADT from: {query_url}")
            candidate = fetch_arcgis_geojson(
                query_url,
                where="1=1",
                out_fields="*",
                bbox=None,
                out_crs=4326,
                page_size=min(page_size, 20_000),
                order_by="OBJECTID ASC",
            )
            if candidate is not None and not candidate.empty:
                gdf = candidate
                break
        except Exception as exc:  # pragma: no cover - depends on upstream API
            last_error = exc
            print(f"AADT endpoint failed: {exc}")

    if gdf is None or gdf.empty:
        raise RuntimeError(
            "NYSDOT AADT could not be fetched from either the Roadway Data "
            "Viewer MapServer or FeatureServer endpoint."
        ) from last_error

    gdf = prepare_aadt(gdf)

    # Preserve the old bbox argument's behavior, but clip locally in WGS84.
    if bbox is not None and not gdf.empty:
        xmin, ymin, xmax, ymax = bbox
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf_4326 = gdf.to_crs("EPSG:4326")
        gdf_4326 = gdf_4326.cx[xmin:xmax, ymin:ymax].copy()
        gdf = gpd.GeoDataFrame(gdf_4326, geometry="geometry", crs="EPSG:4326")

    save_cache(
        gdf,
        key,
        cache_dir=cache_dir,
        metadata={
            "source": NYSDOT_AADT_URLS[0],
            "bbox": bbox,
            "nyc_filter": "DOT_Region_Number == 11 (applied locally)",
            "aadt_field": "AADT_Current_Estimate",
        },
    )
    return gdf


def prepare_aadt(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalize Roadway Data Viewer AADT schema and retain NYC observations."""
    out = normalize_columns(gdf)

    if out.crs is None:
        out = out.set_crs("EPSG:4326")

    # Normalize the Roadway Data Viewer field names into the names expected by
    # the rest of the project.
    numeric = [
        "aadt_current_estimate",
        "aadt_last_actual",
        "aadt_last_actual_count_year",
        "posted_speed_limit_mph",
        "aadt_single_unit",
        "aadt_combination_unit",
        "dot_region_number",
    ]
    for col in numeric:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Region 11 is New York City. Apply this locally rather than in ArcGIS SQL.
    if "dot_region_number" in out.columns:
        region = pd.to_numeric(out["dot_region_number"], errors="coerce")
        out = out[region.eq(11)].copy()

    # Create compatibility aliases used elsewhere in the pipeline.
    if "aadt_current_estimate" in out.columns:
        out["aadt"] = out["aadt_current_estimate"]
    elif "aadt_last_actual" in out.columns:
        out["aadt"] = out["aadt_last_actual"]
    else:
        raise KeyError(
            "NYSDOT Roadway Data Viewer response does not contain an AADT "
            f"field. Columns: {list(out.columns)}"
        )

    if "aadt_last_actual" in out.columns:
        out["aadt_last_act"] = out["aadt_last_actual"]

    if "aadt_last_actual_count_year" in out.columns:
        out["year_last_act"] = out["aadt_last_actual_count_year"]

    if "posted_speed_limit_mph" in out.columns:
        out["speed"] = out["posted_speed_limit_mph"]

    if "aadt_single_unit" in out.columns:
        out["su_aadt"] = out["aadt_single_unit"]

    if "aadt_combination_unit" in out.columns:
        out["cu_aadt"] = out["aadt_combination_unit"]

    # Drop invalid/nonpositive AADT and null geometries.
    out["aadt"] = pd.to_numeric(out["aadt"], errors="coerce")
    out = out[out["aadt"].gt(0) & out.geometry.notna()].copy()

    return gpd.GeoDataFrame(out, geometry="geometry", crs=out.crs)
