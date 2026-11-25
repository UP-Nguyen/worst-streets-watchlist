"""Spatial/relational joins and exposure metrics for the watchlist pipeline."""
from __future__ import annotations

import re
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd

from .api import normalize_segment_id
from .lion import NYC_PROJECTED_CRS

HARM_COLUMNS = [
    "crashes",
    "people_injured",
    "people_killed",
    "pedestrians_injured",
    "pedestrians_killed",
    "cyclists_injured",
    "cyclists_killed",
    "motorists_injured",
    "motorists_killed",
]


def join_crashes_to_lion(
    crashes: gpd.GeoDataFrame,
    lion_segments: gpd.GeoDataFrame,
    *,
    max_distance_ft: float = 75,
    projected_crs: str = NYC_PROJECTED_CRS,
) -> gpd.GeoDataFrame:
    """Assign each geocoded crash to its nearest LION segment.

    Ties can occur at intersections; this function deterministically keeps the
    nearest/lowest SegmentID for the corridor prototype. Intersection analysis
    should use LION nodes as a separate model rather than interpreting this
    forced assignment as an intersection classification.
    """
    crash = crashes.to_crs(projected_crs).copy()
    lion = lion_segments.to_crs(projected_crs).copy()
    lion["segmentid"] = normalize_segment_id(lion["segmentid"])

    joined = gpd.sjoin_nearest(
        crash,
        lion[["segmentid", "geometry"]],
        how="left",
        max_distance=max_distance_ft,
        distance_col="lion_distance_ft",
    )
    joined = joined.sort_values(
        ["collision_id", "lion_distance_ft", "segmentid"],
        na_position="last",
    ).drop_duplicates(subset="collision_id", keep="first")
    return gpd.GeoDataFrame(joined, geometry="geometry", crs=projected_crs)


def aggregate_crashes_by_segment(crash_lion: gpd.GeoDataFrame) -> pd.DataFrame:
    """Aggregate matched crashes to one row per SegmentID."""
    work = crash_lion[crash_lion["segmentid"].notna()].copy()
    for col in [
        "number_of_persons_injured",
        "number_of_persons_killed",
        "number_of_pedestrians_injured",
        "number_of_pedestrians_killed",
        "number_of_cyclist_injured",
        "number_of_cyclist_killed",
        "number_of_motorist_injured",
        "number_of_motorist_killed",
    ]:
        if col not in work:
            work[col] = 0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)

    return (
        work.groupby("segmentid", as_index=False)
        .agg(
            crashes=("collision_id", "nunique"),
            people_injured=("number_of_persons_injured", "sum"),
            people_killed=("number_of_persons_killed", "sum"),
            pedestrians_injured=("number_of_pedestrians_injured", "sum"),
            pedestrians_killed=("number_of_pedestrians_killed", "sum"),
            cyclists_injured=("number_of_cyclist_injured", "sum"),
            cyclists_killed=("number_of_cyclist_killed", "sum"),
            motorists_injured=("number_of_motorist_injured", "sum"),
            motorists_killed=("number_of_motorist_killed", "sum"),
            mean_crash_to_lion_ft=("lion_distance_ft", "mean"),
            max_crash_to_lion_ft=("lion_distance_ft", "max"),
        )
    )


def attach_crash_metrics(
    lion_segments: gpd.GeoDataFrame,
    segment_crashes: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Left-join crash aggregates onto every LION segment and fill no-crash rows with zero."""
    lion = lion_segments.copy()
    lion["segmentid"] = normalize_segment_id(lion["segmentid"])
    metrics = segment_crashes.copy()
    metrics["segmentid"] = normalize_segment_id(metrics["segmentid"])

    result = lion.merge(metrics, on="segmentid", how="left", validate="1:1")
    for col in HARM_COLUMNS:
        if col in result:
            result[col] = result[col].fillna(0)
    return gpd.GeoDataFrame(result, geometry="geometry", crs=lion.crs)


def attach_traffic_summary(
    segments: gpd.GeoDataFrame,
    traffic_summary: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Ordinary SegmentID join for NYC DOT traffic-count summaries."""
    left = segments.copy()
    right = traffic_summary.copy()
    left["segmentid"] = normalize_segment_id(left["segmentid"])
    right["segmentid"] = normalize_segment_id(right["segmentid"])
    out = left.merge(right, on="segmentid", how="left", validate="1:1")
    return gpd.GeoDataFrame(out, geometry="geometry", crs=segments.crs)


def normalize_street_name(value: object) -> str | None:
    """Coarse street-name normalizer used only as AADT match QA, not as a join key."""
    if value is None or pd.isna(value):
        return None
    text = re.sub(r"[^A-Z0-9 ]+", " ", str(value).upper())
    replacements = {
        r"\bSTREET\b": "ST",
        r"\bAVENUE\b": "AVE",
        r"\bBOULEVARD\b": "BLVD",
        r"\bROAD\b": "RD",
        r"\bPARKWAY\b": "PKWY",
        r"\bHIGHWAY\b": "HWY",
        r"\bEXPRESSWAY\b": "EXPWY",
        r"\bPLACE\b": "PL",
        r"\bDRIVE\b": "DR",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return " ".join(text.split()) or None


def join_aadt_to_lion(
    segments: gpd.GeoDataFrame,
    aadt: gpd.GeoDataFrame,
    *,
    max_distance_ft: float = 100,
    projected_crs: str = NYC_PROJECTED_CRS,
) -> gpd.GeoDataFrame:
    """Attach the nearest NYSDOT AADT roadway to each LION segment.

    Distance and optional street-name agreement are retained as QA variables.
    This is a correspondence step between two different roadway networks, not
    a guaranteed exact identifier join.
    """
    left = segments.to_crs(projected_crs).copy()
    right = aadt.to_crs(projected_crs).copy()
    left["segmentid"] = normalize_segment_id(left["segmentid"])

    desired = [
        "aadt",
        "roadway_name",
        "route_number",
        "speed",
        "functional_class",
        "calculation_year",
        "year_last_act",
        "aadt_last_act",
        "su_aadt",
        "cu_aadt",
        "avg_truck_percent",
        "rc_station",
        "geometry",
    ]
    cols = [c for c in desired if c in right.columns]
    if "geometry" not in cols:
        cols.append("geometry")

    joined = gpd.sjoin_nearest(
        left,
        right[cols],
        how="left",
        max_distance=max_distance_ft,
        distance_col="aadt_distance_ft",
        rsuffix="nysdot",
    )

    # Resolve equidistant candidates deterministically, preferring newer AADT
    # records where the service supplies a year.
    sort_cols = ["segmentid", "aadt_distance_ft"]
    ascending = [True, True]
    for col in ["calculation_year", "year_last_act"]:
        if col in joined:
            joined[col] = pd.to_numeric(joined[col], errors="coerce")
            sort_cols.append(col)
            ascending.append(False)
    joined = joined.sort_values(sort_cols, ascending=ascending, na_position="last")
    joined = joined.drop_duplicates(subset="segmentid", keep="first")

    if "street" in joined and "roadway_name" in joined:
        joined["lion_street_normalized"] = joined["street"].map(normalize_street_name)
        joined["aadt_roadway_normalized"] = joined["roadway_name"].map(normalize_street_name)
        joined["aadt_name_match"] = (
            joined["lion_street_normalized"].notna()
            & joined["aadt_roadway_normalized"].notna()
            & joined["lion_street_normalized"].eq(joined["aadt_roadway_normalized"])
        )
    else:
        joined["aadt_name_match"] = False

    dist = joined["aadt_distance_ft"]
    name_match = joined["aadt_name_match"].fillna(False)
    joined["aadt_match_confidence"] = np.select(
        [
            joined.get("aadt", pd.Series(index=joined.index)).isna(),
            dist.le(20) | (dist.le(50) & name_match),
            dist.le(50) | (dist.le(100) & name_match),
            dist.le(max_distance_ft),
        ],
        ["none", "high", "medium", "low"],
        default="none",
    )
    return gpd.GeoDataFrame(joined, geometry="geometry", crs=projected_crs)


def add_exposure_metrics(
    segments: gpd.GeoDataFrame,
    *,
    start_date: str,
    end_date: str,
) -> gpd.GeoDataFrame:
    """Calculate segment VMT and harm rates using AADT.

    AADT is vehicle exposure only. These rates should not be interpreted as
    pedestrian- or cyclist-exposure-adjusted risk.
    """
    out = segments.to_crs(NYC_PROJECTED_CRS).copy()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    study_days = (end - start).days + 1

    out["length_ft"] = out.geometry.length
    out["length_miles"] = out["length_ft"] / 5280.0
    out["study_days"] = study_days
    out["aadt"] = pd.to_numeric(out.get("aadt"), errors="coerce")
    out["vehicle_miles_traveled"] = out["aadt"] * study_days * out["length_miles"]

    denominator = out["vehicle_miles_traveled"].where(out["vehicle_miles_traveled"] > 0)
    rate_map = {
        "crashes": "crashes_per_100m_vmt",
        "people_injured": "injuries_per_100m_vmt",
        "people_killed": "deaths_per_100m_vmt",
        "pedestrians_injured": "ped_injuries_per_100m_vmt",
        "cyclists_injured": "cyclist_injuries_per_100m_vmt",
    }
    for source, target in rate_map.items():
        if source in out:
            out[target] = pd.to_numeric(out[source], errors="coerce") / denominator * 100_000_000

    out["rate_eligible"] = (
        out["aadt"].gt(0)
        & out["length_miles"].gt(0)
        & out.get("aadt_match_confidence", "none").isin(["high", "medium"])
    )
    return out


def add_rank_comparison(
    segments: gpd.GeoDataFrame,
    *,
    raw_metric: str = "people_injured",
    adjusted_metric: str = "injuries_per_100m_vmt",
) -> gpd.GeoDataFrame:
    """Add raw and exposure-adjusted ranks; positive change means rank rises after adjustment."""
    out = segments.copy()
    eligible = out.get("rate_eligible", pd.Series(True, index=out.index)).fillna(False)
    out["raw_rank"] = pd.to_numeric(out[raw_metric], errors="coerce").rank(method="min", ascending=False)
    out["adjusted_rank"] = np.nan
    out.loc[eligible, "adjusted_rank"] = pd.to_numeric(
        out.loc[eligible, adjusted_metric], errors="coerce"
    ).rank(method="min", ascending=False)
    out["rank_change"] = out["raw_rank"] - out["adjusted_rank"]
    return gpd.GeoDataFrame(out, geometry="geometry", crs=segments.crs)
