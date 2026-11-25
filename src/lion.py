"""LION street-network extraction and preparation."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from .api import (
    LION_ARCGIS_URL,
    LION_ID,
    cache_key,
    fetch_arcgis_geojson,
    load_cache,
    normalize_columns,
    normalize_segment_id,
    save_cache,
)

NYC_PROJECTED_CRS = "EPSG:2263"

LION_FIELDS = [
    "OBJECTID",
    "Street",
    "FeatureTyp",
    "SegmentTyp",
    "NonPed",
    "TrafDir",
    "SegmentID",
    "NodeIDFrom",
    "NodeIDTo",
    "RW_TYPE",
    "PhysicalID",
    "Status",
    "StreetWidth_Min",
    "StreetWidth_Max",
    "BikeLane",
    "ACTIVE_FLAG",
    "Number_Travel_Lanes",
    "Number_Park_Lanes",
    "Number_Total_Lanes",
    "POSTED_SPEED",
    "TRUCK_ROUTE_TYPE",
]


def fetch_lion(
    *,
    refresh: bool = False,
    cache_dir: str | Path = "data/cache",
    page_size: int = 2_000,
) -> gpd.GeoDataFrame:
    """Fetch the current DCP LION layer from its ArcGIS REST service."""
    key = cache_key("lion", dataset=LION_ID)
    if not refresh:
        cached = load_cache(key, cache_dir=cache_dir)
        if cached is not None:
            print(f"Using cached LION snapshot: {key}")
            return cached

    lion = fetch_arcgis_geojson(
        LION_ARCGIS_URL,
        where="1=1",
        out_fields=",".join(LION_FIELDS),
        out_crs=4326,
        page_size=min(page_size, 2_000),
        order_by="OBJECTID",
    )
    lion = normalize_columns(lion)
    lion = lion.rename(columns={"segment_id": "segmentid"})
    if "segmentid" not in lion.columns:
        raise KeyError(f"LION response does not contain segmentid. Columns: {list(lion.columns)}")
    lion["segmentid"] = normalize_segment_id(lion["segmentid"])
    lion = lion[lion.geometry.notna() & lion["segmentid"].notna()].copy()
    save_cache(
        lion,
        key,
        cache_dir=cache_dir,
        metadata={"dataset_id": LION_ID, "source": LION_ARCGIS_URL},
    )
    return lion


def prepare_lion_segments(
    lion: gpd.GeoDataFrame,
    *,
    projected_crs: str = NYC_PROJECTED_CRS,
    street_only: bool = True,
) -> gpd.GeoDataFrame:
    """Collapse duplicate LION records to one physical SegmentID.

    LION can include multiple records for one physical segment (e.g. alternate
    address records or borough-boundary representation). Dissolving prevents
    one-to-many explosions later in the pipeline.
    """
    gdf = normalize_columns(lion)
    gdf = gdf.rename(columns={"segment_id": "segmentid"})
    if "segmentid" not in gdf.columns:
        raise KeyError("Expected segmentid in LION data")
    gdf["segmentid"] = normalize_segment_id(gdf["segmentid"])
    gdf = gdf[gdf["segmentid"].notna() & gdf.geometry.notna()].copy()

    # LION includes railroads, shorelines, census boundaries, paper streets,
    # ferries and other linear features. For the roadway-safety prototype,
    # retain physically existing street-like features when FeatureTyp exists.
    # 0=street, 6=private street, A=alley, C=CCO/public-use street area.
    if street_only and "feature_typ" in gdf.columns:
        gdf = gdf[gdf["feature_typ"].astype("string").isin(["0", "6", "A", "C"])].copy()

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(projected_crs)

    # Preserve a compact set of useful attributes when present.
    preferred = [
        "segmentid",
        "street",
        "feature_typ",
        "segment_typ",
        "non_ped",
        "node_id_from",
        "node_id_to",
        "rw_type",
        "traf_dir",
        "bike_lane",
        "physical_id",
        "streetwidt",
        "streetwidth",
        "street_width_min",
        "street_width_max",
        "num_travel_lanes",
        "number_travel_lanes",
        "number_park_lanes",
        "number_total_lanes",
        "posted_speed",
        "truck_route_type",
        "active_flag",
        "status",
        "geometry",
    ]
    keep = [c for c in preferred if c in gdf.columns]
    gdf = gdf[keep].copy()

    for col in [
        "street_width_min", "street_width_max", "number_travel_lanes",
        "number_park_lanes", "number_total_lanes", "posted_speed", "physical_id",
    ]:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    attrs = [c for c in gdf.columns if c not in {"segmentid", "geometry"}]
    agg = {c: "first" for c in attrs}
    dissolved = gdf.dissolve(by="segmentid", aggfunc=agg, as_index=False)
    dissolved = gpd.GeoDataFrame(dissolved, geometry="geometry", crs=projected_crs)
    return dissolved
