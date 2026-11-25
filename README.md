# Worst Streets Watchlist

NYU CUSP Capstone project for developing a reproducible method to identify NYC street segments with high traffic harm, compare raw harm with vehicle-exposure-adjusted rates, and ultimately support diagnosis, intervention prioritization, and a "cost of doing nothing" analysis.

The repository is **API-first**: public NYC Open Data, NYC DCP ArcGIS REST, and NYSDOT services are the authoritative inputs. Local files under `data/cache/` are timestamped snapshots used only to make notebook runs reproducible and to avoid repeatedly downloading large datasets.

## Current pipeline

```text
NYPD Motor Vehicle Collisions API
            |
            v
     geocoded crash points
            |
            | nearest spatial match
            v
     DCP LION SegmentID  <-------- NYC DOT ATR counts (direct SegmentID join)
            |
            | aggregate crash harm to one row / segment
            v
   segment-level harm table
            |
            | nearest roadway-network correspondence + QA
            v
      NYSDOT AADT lines
            |
            v
 raw harm vs. vehicle-exposure-adjusted harm
```

The first pass analytical comparison is intentionally simple (get it working first)

- **Raw harm:** where did the most crashes/injuries/deaths occur?
- **Traffic-adjusted harm:** where is harm high relative to vehicle-miles traveled (VMT)?
- **Rank change:** which segments become much more concerning after accounting for motor-vehicle volume?

This is an exploratory network-screening step, not  the final Watchlist score. 
Final prioritization should add statistical stabilization 
?(for example Empirical Bayes / safety performance functions), pedestrian and cyclist exposure, street-design factors, treatment history, and uncertainty.

## Data sources

| Source | Dataset | Role |
|---|---|---|
| NYC Open Data / NYPD | Motor Vehicle Collisions - Crashes (`h9gi-nx95`) | Crash events and injury/fatality outcomes |
| NYC Open Data / NYPD | MVC Person (`f55k-p6yu`) | Optional person-level enrichment |
| NYC Open Data / NYPD | MVC Vehicles (`bm4k-52h4`) | Optional vehicle-level enrichment |
| NYC DCP ArcGIS REST / NYC Open Data catalog | LION (catalog `2v4z-66xt`) | Street-network spine and SegmentID |
| NYC Open Data / DOT | Automated Traffic Volume Counts (`7ym2-wayt`) | Supplemental sampled traffic counts; **not AADT** |
| NYSDOT ArcGIS REST | Traffic Monitoring / AADT Feature Layer | Annual Average Daily Traffic used for VMT exposure |

Source pages:

- https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95
- https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Person/f55k-p6yu
- https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Vehicles/bm4k-52h4
- https://www.nyc.gov/content/planning/pages/resources/datasets/lion
- https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/ArcGIS/rest/services/LION/FeatureServer/0
- https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt
- https://gis.dot.ny.gov/hostingny/rest/services/Roadways/Traffic_Monitoring/FeatureServer/1

## Repository structure

```text
worst-streets-watchlist/
├── src/
│   ├── api.py          # HTTP sessions, pagination, schema normalization, cache snapshots
│   ├── crashes.py      # crash/person/vehicle API pulls and crash cleaning
│   ├── lion.py         # LION API pull and one-row-per-SegmentID preparation
│   ├── traffic.py      # NYC DOT sampled traffic counts and daily summaries
│   ├── aadt.py         # NYSDOT AADT ArcGIS pull and cleaning
│   └── joins.py        # crash->LION, AADT->LION, aggregation, VMT, rankings
├── notebooks/
│   ├── 01_pull_live_data.ipynb
│   ├── 02_crash_lion_join.ipynb
│   ├── 03_aadt_join.ipynb
│   └── 04_raw_vs_exposure.ipynb
├── data/
│   ├── cache/          # ignored local API snapshots + retrieval metadata
│   └── outputs/        # ignored derived CSV / GeoJSON outputs
├── tests/
├── requirements.txt
└── README.md
```

## Setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

### Optional Socrata app token

NYC Open Data can be queried without a token, but a free Socrata application token is recommended for larger pulls.

```bash
export SOCRATA_APP_TOKEN="your_token_here"
```

Or copy `.env.example` to `.env` and export the value in your shell. The code never commits tokens.

## Analysis period vs. live data

The API source remains live, but a published analysis should normally use complete calendar years for reproducibility. The notebooks default to:

```python
ANALYSIS_START = "2021-01-01"
ANALYSIS_END = "2025-12-31"
```

Change these values for another study period. For a live dashboard, set `ANALYSIS_END` to today's date, but label partial-year comparisons clearly.

## Run the notebooks in order

### 1. `01_pull_live_data.ipynb`

Pulls and caches:

- crash events for the analysis period;
- the current LION street network from DCP's ArcGIS REST layer (filtered to physically existing street-like features for the roadway prototype);
- NYSDOT AADT for the NYC bounding box;
- optional recent NYC DOT automated traffic counts.

Set:

```python
REFRESH = True
```

to force a new API snapshot. Leave it `False` to reuse the timestamped local cache.

Person and vehicle tables are supported by `src/crashes.py`, but are intentionally optional because the crash table already contains aggregate pedestrian/cyclist/motorist injury and fatality counts needed for the initial segment analysis.

### 2. `02_crash_lion_join.ipynb`

- converts crash coordinates to points;
- projects both datasets to NYC State Plane (`EPSG:2263`);
- assigns each crash to the nearest LION segment within a QA threshold;
- aggregates to one row per `SegmentID`;
- reports unmatched crashes and match distances.

**Intersection caveat:** a crash located exactly at an intersection may be equidistant from multiple street segments. The corridor prototype deterministically assigns it to one segment. A future intersection model should use LION `NodeID` rather than treating this forced segment assignment as an intersection classification.

### 3. `03_aadt_join.ipynb`

- matches each LION segment to the nearest NYSDOT AADT line;
- retains distance and street-name agreement as QA fields;
- labels AADT matches `high`, `medium`, `low`, or `none` confidence;
- computes segment length and study-period VMT;
- calculates harm rates per 100 million VMT.

The AADT join is a **network correspondence**, not a shared-ID join. NYSDOT and LION are different roadway networks, so ambiguous/parallel-road matches must be inspected before publication.

### 4. `04_raw_vs_exposure.ipynb`

Compares:

- raw injury ranking;
- injuries per 100M VMT ranking;
- rank change after vehicle exposure adjustment.

It exports a compact CSV to `data/outputs/raw_vs_exposure.csv` and includes exploratory maps.

## Core formulas

For a LION segment:

```text
study-period VMT = AADT x number of study days x segment length (miles)
```

```text
injury rate per 100M VMT = injuries / study-period VMT x 100,000,000
```

The same denominator can be used for crash and fatality rates.

## Important methodological caveats

1. **AADT adjusts for motor-vehicle exposure only.** It is not a pedestrian- or cyclist-exposure denominator. Pedestrian demand/counts and bicycle counts should eventually be modeled separately.
2. **Raw rates are unstable for short / low-exposure segments.** One crash on a tiny segment can create an extreme rate. Do not turn the exploratory rate ranking directly into the final Watchlist. Add minimum-data rules and preferably Empirical Bayes / safety-performance-function methods.
3. **AADT coverage is not uniform across every local NYC street.** Segments without reliable AADT remain valid for raw-harm analysis but should not receive a vehicle-exposure-adjusted ranking.
4. **Automated Traffic Volume Counts are samples, not annual averages.** `src/traffic.py` therefore labels them as observed sample-day volume rather than pretending they are AADT.
5. **Some DOT datasets reference older LION vintages.** For example, the automated traffic-count metadata describes `SegmentID` against LION version 14. Always report join coverage when attaching historical/current DOT datasets to the latest LION release.
6. **NYPD collision data are preliminary and can be amended.** Cached snapshots and explicit analysis dates make results reproducible.

## Next methodological extensions

Once the baseline pipeline is stable,  next steps?

- pedestrian demand and direct pedestrian/bicycle counts;
- bike facilities, speed limits, truck routes, bus lanes, LPIs, turn-calming and other Vision Zero treatments;
- matched "safer twin" comparisons;
- expected-vs-observed harm / Empirical Bayes network screening;
- treatment-gap analysis;
- intervention Crash Modification Factors;
- five-year expected cost of inaction and benefit-cost analysis.

## Development / testing

The tests use synthetic geometries and do not require network access:

```bash
python -m unittest discover -s tests -v
```