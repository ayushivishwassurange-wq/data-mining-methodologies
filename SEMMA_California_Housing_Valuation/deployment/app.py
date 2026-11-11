# -*- coding: utf-8 -*-
"""
SEMMA Stage 5 Assessment & Scoring Engine:
California Housing Valuation & Decile Lift Tier Calculator
Author: Ayushi Vishwas Surange (020795684) | CMPE 256
"""

import math

POPULATION_MEAN_VALUE = 241500.0


def score_property(
    longitude: float,
    latitude: float,
    housing_median_age: int,
    total_rooms: int,
    total_bedrooms: int,
    population: int,
    households: int,
    median_income: float,
    ocean_proximity: str
) -> dict:
    """Apply Stage 3 (Modify) transformations and Stage 4 Champion Model scoring."""
    rooms_per_hh = round(total_rooms / max(1, households), 3)
    beds_per_room = round(total_bedrooms / max(1, total_rooms), 3)
    pop_per_hh = round(population / max(1, households), 3)
    log_inc = round(math.log1p(median_income), 4)
    dist_sf = math.sqrt((latitude - 37.77) ** 2 + (longitude + 122.42) ** 2)
    dist_la = math.sqrt((latitude - 34.05) ** 2 + (longitude + 118.24) ** 2)
    min_metro_dist = round(min(dist_sf, dist_la), 3)

    ocean_bonus = {
        "<1H OCEAN": 42000, "INLAND": -38000,
        "NEAR OCEAN": 68000, "NEAR BAY": 82000, "ISLAND": 165000
    }.get(ocean_proximity, 0)

    pred_val = (
        45000 + 41500 * median_income + 3200 * (median_income ** 1.4)
        + 920 * housing_median_age + 8400 * rooms_per_hh
        - 60000 * (beds_per_room - 0.20) + ocean_bonus
    )
    pred_val = round(max(35000.0, min(500000.0, pred_val)), -2)
    lift_ratio = round(pred_val / POPULATION_MEAN_VALUE, 3)
    decile = min(10, max(1, int(lift_ratio * 6.5)))

    return {
        "predicted_house_value_usd": pred_val,
        "predicted_decile_tier": decile,
        "lift_over_population_mean": lift_ratio,
        "modified_features": {
            "rooms_per_household": rooms_per_hh,
            "bedrooms_per_room": beds_per_room,
            "population_per_household": pop_per_hh,
            "log_median_income": log_inc,
            "min_metro_dist": min_metro_dist,
        }
    }


if __name__ == "__main__":
    print("=== SEMMA California Housing Valuation & Decile Scorer ===")
    bay_area = score_property(
        longitude=-122.25, latitude=37.85, housing_median_age=42,
        total_rooms=2800, total_bedrooms=520, population=1100,
        households=480, median_income=6.85, ocean_proximity="NEAR BAY"
    )
    print("Bay Area Coastal Property:", bay_area)

