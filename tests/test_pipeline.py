import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from src.api import normalize_segment_id
from src.joins import (
    add_exposure_metrics,
    add_rank_comparison,
    aggregate_crashes_by_segment,
    attach_crash_metrics,
    join_aadt_to_lion,
    join_crashes_to_lion,
)


### fake tests! does the pipeline work 

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.lion = gpd.GeoDataFrame(
            {
                "segmentid": ["0000001", "0000002"],
                "street": ["MAIN STREET", "SECOND AVENUE"],
            },
            geometry=[
                LineString([(0, 0), (1000, 0)]),
                LineString([(0, 500), (1000, 500)]),
            ],
            crs="EPSG:2263",
        )

        self.crashes = gpd.GeoDataFrame(
            {
                "collision_id": ["a", "b", "c"],
                "number_of_persons_injured": [2, 1, 1],
                "number_of_persons_killed": [0, 0, 1],
                "number_of_pedestrians_injured": [1, 0, 0],
                "number_of_pedestrians_killed": [0, 0, 1],
                "number_of_cyclist_injured": [0, 1, 0],
                "number_of_cyclist_killed": [0, 0, 0],
                "number_of_motorist_injured": [1, 0, 1],
                "number_of_motorist_killed": [0, 0, 0],
            },
            geometry=[Point(100, 5), Point(200, 10), Point(100, 510)],
            crs="EPSG:2263",
        )

    def test_segment_id_normalization(self):
        result = normalize_segment_id(pd.Series([1, "23", None]))
        self.assertEqual(result.iloc[0], "0000001")
        self.assertEqual(result.iloc[1], "0000023")
        self.assertTrue(pd.isna(result.iloc[2]))

    def test_crash_join_and_aggregation(self):
        joined = join_crashes_to_lion(self.crashes, self.lion, max_distance_ft=50)
        self.assertEqual(joined["segmentid"].notna().sum(), 3)
        agg = aggregate_crashes_by_segment(joined)
        first = agg.loc[agg["segmentid"] == "0000001"].iloc[0]
        self.assertEqual(first["crashes"], 2)
        self.assertEqual(first["people_injured"], 3)

        segments = attach_crash_metrics(self.lion, agg)
        self.assertEqual(len(segments), 2)

    def test_aadt_and_rates(self):
        aadt = gpd.GeoDataFrame(
            {
                "aadt": [10_000, 2_000],
                "roadway_name": ["MAIN ST", "SECOND AVE"],
                "calculation_year": [2025, 2025],
            },
            geometry=[
                LineString([(0, 5), (1000, 5)]),
                LineString([(0, 505), (1000, 505)]),
            ],
            crs="EPSG:2263",
        )
        segments = self.lion.copy()
        segments["crashes"] = [10, 5]
        segments["people_injured"] = [10, 5]
        segments["people_killed"] = [0, 0]
        segments["pedestrians_injured"] = [0, 0]
        segments["cyclists_injured"] = [0, 0]

        matched = join_aadt_to_lion(segments, aadt, max_distance_ft=50)
        self.assertTrue((matched["aadt_match_confidence"] == "high").all())
        rated = add_exposure_metrics(matched, start_date="2025-01-01", end_date="2025-12-31")
        self.assertTrue((rated["vehicle_miles_traveled"] > 0).all())
        self.assertTrue((rated["injuries_per_100m_vmt"] > 0).all())

        ranked = add_rank_comparison(rated)
        self.assertIn("rank_change", ranked.columns)


if __name__ == "__main__":
    unittest.main()
