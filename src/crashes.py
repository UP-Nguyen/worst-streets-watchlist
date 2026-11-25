"""NYC Motor Vehicle Collisions extraction and cleaning."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .api import (
    CRASHES_ID,
    PERSON_ID,
    VEHICLES_ID,
    cache_key,
    fetch_socrata,
    load_cache,
    normalize_columns,
    save_cache,
)

CRASH_FIELDS = [
    "collision_id",
    "crash_date",
    "crash_time",
    "borough",
    "zip_code",
    "latitude",
    "longitude",
    "on_street_name",
    "cross_street_name",
    "off_street_name",
    "number_of_persons_injured",
    "number_of_persons_killed",
    "number_of_pedestrians_injured",
    "number_of_pedestrians_killed",
    "number_of_cyclist_injured",
    "number_of_cyclist_killed",
    "number_of_motorist_injured",
    "number_of_motorist_killed",
    "contributing_factor_vehicle_1",
]

CRASH_NUMERIC_FIELDS = [
    "latitude",
    "longitude",
    "number_of_persons_injured",
    "number_of_persons_killed",
    "number_of_pedestrians_injured",
    "number_of_pedestrians_killed",
    "number_of_cyclist_injured",
    "number_of_cyclist_killed",
    "number_of_motorist_injured",
    "number_of_motorist_killed",
]

PERSON_FIELDS = [
    "unique_id",
    "collision_id",
    "crash_date",
    "person_id",
    "person_type",
    "person_injury",
    "person_age",
    "bodily_injury",
    "ped_location",
    "ped_action",
    "ped_role",
]

VEHICLE_FIELDS = [
    "unique_id",
    "collision_id",
    "crash_date",
    "vehicle_id",
    "vehicle_type",
    "vehicle_make",
    "vehicle_model",
    "vehicle_year",
    "travel_direction",
    "vehicle_occupants",
    "pre_crash",
    "point_of_impact",
    "contributing_factor_1",
    "contributing_factor_2",
]


def _date_string(value: str | date | datetime | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _date_where(start_date, end_date) -> str:
    """Build half-open Socrata date filter [start_date, end_date + 1 day)."""
    start = pd.Timestamp(start_date).normalize()
    end_exclusive = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
    return (
        f"crash_date >= '{start:%Y-%m-%d}T00:00:00.000' "
        f"AND crash_date < '{end_exclusive:%Y-%m-%d}T00:00:00.000'"
    )


def clean_crashes(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    if df.empty:
        return df
    if "collision_id" in df:
        df["collision_id"] = df["collision_id"].astype("string")
    if "crash_date" in df:
        df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")
    for col in CRASH_NUMERIC_FIELDS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # A small number of historical records use zero coordinates; those are not
    # valid NYC locations and should not enter spatial matching.
    if {"latitude", "longitude"}.issubset(df.columns):
        df.loc[(df["latitude"] == 0) | (df["longitude"] == 0), ["latitude", "longitude"]] = pd.NA
    return df


def fetch_crashes(
    start_date: str,
    end_date: str,
    *,
    refresh: bool = False,
    cache_dir: str | Path = "data/cache",
    page_size: int = 50_000,
) -> pd.DataFrame:
    """Fetch crash events for an inclusive date range."""
    key = cache_key("crashes", start=_date_string(start_date), end=_date_string(end_date))
    if not refresh:
        cached = load_cache(key, cache_dir=cache_dir)
        if cached is not None:
            print(f"Using cached crashes snapshot: {key}")
            return cached

    df = fetch_socrata(
        CRASHES_ID,
        where=_date_where(start_date, end_date),
        select=CRASH_FIELDS,
        order="crash_date,collision_id",
        page_size=page_size,
    )
    df = clean_crashes(df)
    save_cache(
        df,
        key,
        cache_dir=cache_dir,
        metadata={"dataset_id": CRASHES_ID, "start_date": start_date, "end_date": end_date},
    )
    return df


def crashes_to_geodataframe(df: pd.DataFrame, *, drop_missing: bool = True) -> gpd.GeoDataFrame:
    """Convert crash coordinates to an EPSG:4326 point GeoDataFrame."""
    work = df.copy()
    if drop_missing:
        work = work.dropna(subset=["latitude", "longitude"])
    gdf = gpd.GeoDataFrame(
        work,
        geometry=gpd.points_from_xy(work["longitude"], work["latitude"]),
        crs="EPSG:4326",
    )
    return gdf


def fetch_people(
    start_date: str,
    end_date: str,
    *,
    refresh: bool = False,
    cache_dir: str | Path = "data/cache",
    page_size: int = 50_000,
) -> pd.DataFrame:
    """Optional person-level MVC enrichment."""
    key = cache_key("people", start=_date_string(start_date), end=_date_string(end_date))
    if not refresh:
        cached = load_cache(key, cache_dir=cache_dir)
        if cached is not None:
            print(f"Using cached people snapshot: {key}")
            return cached

    df = fetch_socrata(
        PERSON_ID,
        where=_date_where(start_date, end_date),
        select=PERSON_FIELDS,
        order="crash_date,collision_id,unique_id",
        page_size=page_size,
    )
    df = normalize_columns(df)
    if "collision_id" in df:
        df["collision_id"] = df["collision_id"].astype("string")
    if "person_age" in df:
        df["person_age"] = pd.to_numeric(df["person_age"], errors="coerce")
    save_cache(df, key, cache_dir=cache_dir, metadata={"dataset_id": PERSON_ID})
    return df


def aggregate_people_by_crash(people: pd.DataFrame) -> pd.DataFrame:
    """Collapse person-level records to exactly one row per collision."""
    df = people.copy()
    if df.empty:
        return pd.DataFrame(columns=["collision_id"])
    injury = df.get("person_injury", pd.Series(index=df.index, dtype="string")).astype("string").str.upper()
    ptype = df.get("person_type", pd.Series(index=df.index, dtype="string")).astype("string").str.upper()
    age = pd.to_numeric(df.get("person_age"), errors="coerce") if "person_age" in df else pd.Series(index=df.index, dtype=float)

    df["is_injured"] = injury.eq("INJURED")
    df["is_killed"] = injury.eq("KILLED")
    df["is_pedestrian"] = ptype.eq("PEDESTRIAN")
    df["injured_child"] = df["is_injured"] & age.between(0, 17)
    df["injured_older_adult"] = df["is_injured"] & age.ge(65)

    return (
        df.groupby("collision_id", dropna=False)
        .agg(
            people_records=("unique_id", "count"),
            people_injured=("is_injured", "sum"),
            people_killed=("is_killed", "sum"),
            pedestrian_people=("is_pedestrian", "sum"),
            injured_children=("injured_child", "sum"),
            injured_older_adults=("injured_older_adult", "sum"),
        )
        .reset_index()
    )


def fetch_vehicles(
    start_date: str,
    end_date: str,
    *,
    refresh: bool = False,
    cache_dir: str | Path = "data/cache",
    page_size: int = 50_000,
) -> pd.DataFrame:
    """Optional vehicle-level MVC enrichment."""
    key = cache_key("vehicles", start=_date_string(start_date), end=_date_string(end_date))
    if not refresh:
        cached = load_cache(key, cache_dir=cache_dir)
        if cached is not None:
            print(f"Using cached vehicles snapshot: {key}")
            return cached

    df = fetch_socrata(
        VEHICLES_ID,
        where=_date_where(start_date, end_date),
        select=VEHICLE_FIELDS,
        order="crash_date,collision_id,unique_id",
        page_size=page_size,
    )
    df = normalize_columns(df)
    if "collision_id" in df:
        df["collision_id"] = df["collision_id"].astype("string")
    save_cache(df, key, cache_dir=cache_dir, metadata={"dataset_id": VEHICLES_ID})
    return df


def aggregate_vehicles_by_crash(vehicles: pd.DataFrame) -> pd.DataFrame:
    """Collapse vehicle-level records to one row per collision."""
    df = vehicles.copy()
    if df.empty:
        return pd.DataFrame(columns=["collision_id"])
    vehicle_type = df.get("vehicle_type", pd.Series(index=df.index, dtype="string")).astype("string").str.upper()
    df["heavy_vehicle"] = vehicle_type.str.contains(
        r"TRUCK|TRACTOR|SEMI|BUS|DUMP|DELIVERY", regex=True, na=False
    )
    return (
        df.groupby("collision_id", dropna=False)
        .agg(
            vehicle_records=("unique_id", "count"),
            vehicle_count=("vehicle_id", "nunique"),
            heavy_vehicle_involved=("heavy_vehicle", "max"),
        )
        .reset_index()
    )
