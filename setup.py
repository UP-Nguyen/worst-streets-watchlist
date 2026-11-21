import os
import requests
import pandas as pd
import geopandas as gpd
from io import StringIO

NYC_DOMAIN = "https://data.cityofnewyork.us"

CRASHES_ID = "h9gi-nx95"
PERSON_ID = "f55k-p6yu"
VEHICLES_ID = "bm4k-52h4"
LION_ID = "2v4z-66xt"

TRAFFIC_COUNTS_ID = "7ym2-wayt"

NYSDOT_AADT_URL = (
    "https://gis.dot.ny.gov/hostingny/rest/services/"
    "Roadways/Traffic_Monitoring/FeatureServer/1/query"
)

# TODO:
# create a free Socrata app token and put it in environment.
SOCRATA_APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN")

HEADERS = {}
if SOCRATA_APP_TOKEN:
    HEADERS["X-App-Token"] = SOCRATA_APP_TOKEN